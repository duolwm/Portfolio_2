import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    meta: Dict


YAML_FM_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

def parse_yaml_front_matter(raw: str) -> tuple[dict, str]:
    """若檔案有 YAML front matter（--- ... ---），回傳 (meta, body_raw)"""
    m = YAML_FM_RE.match(raw)
    if not m:
        return {}, raw
    fm = m.group(0)
    body = raw[len(fm):]
    meta = {}
    try:
        import yaml
        lines = fm.splitlines()
        if len(lines) >= 3 and lines[0].strip() == "---":
            inner = "\n".join(lines[1:-1])
            meta = yaml.safe_load(inner) or {}
    except Exception:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), body
    
MULTI_NL_RE = re.compile(r"\n{3,}")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)(\{[^}]*\})?")
MD_ATTR_RE = re.compile(r"\{#[^}]+\}|\{\.unnumbered[^}]*\}|\{\.unlisted[^}]*\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# ESG reports might not have these specific noise lines, but we can keep the function structure
NOISE_LINES = []

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = MULTI_NL_RE.sub("\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def strip_noise_lines(text: str) -> str:
    for n in NOISE_LINES:
        text = text.replace(n, "")
    return text


def strip_markdown_attributes(text: str) -> str:
    return MD_ATTR_RE.sub("", text)


def strip_images(text: str) -> str:
    return MD_IMAGE_RE.sub("", text)


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    text = "\n".join(pages)
    text = strip_noise_lines(text)
    return normalize_text(text)


def split_with_heading_paths(text: str) -> List[Tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("NO_HEADING", text)]

    current = {}
    sections = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        current[level] = title
        for lv in list(current.keys()):
            if lv > level:
                del current[lv]

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        path_parts = []
        for lv in range(1, level + 1):
            if lv in current:
                path_parts.append(f"H{lv}:{current[lv]}")
        heading_path = " > ".join(path_parts) if path_parts else f"H{level}:{title}"
        sections.append((heading_path, body))

    return sections if sections else [("NO_HEADING", text)]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 180) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def make_chunks(doc_text: str, source: str, path: str,
                chunk_size: int = 1000, overlap: int = 180) -> List[Chunk]:
    sections = split_with_heading_paths(doc_text)
    out: List[Chunk] = []
    for heading_path, sec_body in sections:
        parts = chunk_text(sec_body, chunk_size=chunk_size, overlap=overlap)
        for i, p in enumerate(parts):
            out.append(Chunk(
                text=p,
                meta={
                    "source": source,
                    "section_path": heading_path,
                    "part": i,
                    "path": path
                }
            ))
    return out


def load_any_file_with_meta(path: Path) -> tuple[str, dict]:
    """讀取檔案並回傳（純文字內容, 由 YAML front matter 解析出的 meta）"""
    suffix = path.suffix.lower()
    extra_meta: dict = {}
    
    if suffix == ".pdf":
        return pdf_to_text(path), extra_meta

    if suffix in [".md", ".txt"]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = parse_yaml_front_matter(raw)
        extra_meta = fm
        body = strip_images(strip_markdown_attributes(body))
        body = HTML_TAG_RE.sub(" ", body)
        body = strip_noise_lines(body)
        return normalize_text(body), extra_meta

    # Fallback
    raw = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = parse_yaml_front_matter(raw)
    extra_meta = fm
    body = strip_images(strip_markdown_attributes(body))
    body = HTML_TAG_RE.sub(" ", body)
    body = strip_noise_lines(body)
    return normalize_text(body), extra_meta
