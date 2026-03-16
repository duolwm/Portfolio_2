from pathlib import Path
import argparse
import json
import chromadb

def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata 只能是 scalar；遇到 dict/list 轉成 JSON 字串。"""
    out = {}
    for k, v in (meta or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

from chromadb.utils import embedding_functions

from loaders_enterprise import load_any_file_with_meta, make_chunks

DB_DIR = "./chroma_db"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

COL_ESG_REPORTS = "esg_reports"
COL_KNOWLEDGE = "knowledge_bank"
COL_STYLE_REPORT = "style_report"

DOCS_DIR = Path("./docs")
BM25_DIR = Path("./bm25_store")
BM25_DIR.mkdir(parents=True, exist_ok=True)


def ingest_folder(client, collection_name: str, folder: Path, bm25_path: Path):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device=device)
    col = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    try:
        col.delete(where={})
    except Exception:
        pass

    if bm25_path.exists():
        bm25_path.unlink()

    total = 0
    bm25_rows = 0

    with bm25_path.open("w", encoding="utf-8") as bm25_f:
        for p in sorted(folder.glob("**/*")):
            if p.is_dir():
                continue
            if p.suffix.lower() not in [".pdf", ".txt", ".md"]:
                continue

            try:
                doc_text, extra_meta = load_any_file_with_meta(p)
                if not doc_text:
                    continue
                chunks = make_chunks(doc_text, source=p.name, path=str(p))
            except Exception as e:
                print(f"ERROR processing {p.name}: {e}")
                continue

            import hashlib
            ids, docs, metas = [], [], []
            for c in chunks:
                meta = dict(c.meta)
                for k, v in (extra_meta or {}).items():
                    if v is None:
                        continue
                    meta[k] = v
                
                # Create a unique hash ID based on source, section, part, and text content
                # This prevents DuplicateIDError even with garbled metadata
                input_str = f"{c.meta['source']}_{c.meta['section_path']}_{c.meta['part']}_{c.text}"
                cid = hashlib.sha256(input_str.encode("utf-8", errors="ignore")).hexdigest()[:32]
                
                ids.append(cid)
                docs.append(c.text)
                metas.append(meta)

                bm25_f.write(json.dumps({"id": cid, "text": c.text, "meta": meta}, ensure_ascii=False) + "\n")
                bm25_rows += 1

            if docs:
                metas = [sanitize_metadata(m) for m in metas]
                
                # Deduplicate within the batch to be extra safe
                unique_ids = []
                unique_docs = []
                unique_metas = []
                seen_ids = set()
                for i, cid in enumerate(ids):
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        unique_ids.append(cid)
                        unique_docs.append(docs[i])
                        unique_metas.append(metas[i])
                
                # Use upsert to handle IDs that might already exist in the DB
                col.upsert(ids=unique_ids, documents=unique_docs, metadatas=unique_metas)
                total += len(unique_docs)
                print(f"[OK] {collection_name}: {p.name} -> {len(unique_docs)} chunks")
        
    print(f"[DONE] {collection_name} 索引完成：chunks={total} | BM25 rows={bm25_rows}")
    print(f"BM25 store：{bm25_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "esg", "knowledge", "style"], default="all")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)

    esg_reports_dir = DOCS_DIR / "esg_reports"
    knowledge_dir = DOCS_DIR / "knowledge_bank"
    style_report_dir = DOCS_DIR / "style_report"

    esg_reports_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    style_report_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ["all", "esg"]:
        ingest_folder(client, COL_ESG_REPORTS, esg_reports_dir, BM25_DIR / f"{COL_ESG_REPORTS}.jsonl")
    if args.mode in ["all", "knowledge"]:
        ingest_folder(client, COL_KNOWLEDGE, knowledge_dir, BM25_DIR / f"{COL_KNOWLEDGE}.jsonl")
    if args.mode in ["all", "style"]:
        ingest_folder(client, COL_STYLE_REPORT, style_report_dir, BM25_DIR / f"{COL_STYLE_REPORT}.jsonl")

if __name__ == "__main__":
    main()
