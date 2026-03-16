# 作品集：ESG RAG 企業永續報告書自動化檢索與查核生成系統

---

## 專案概述 (Project Overview)

本專案建構一個專為**處理並解析長篇 ESG (環境、社會、公司治理) 永續報告書**而設計的 RAG (Retrieval-Augmented Generation) 系統。本系統著重於「事實準確性 (Factuality)」與「專業寫作風格 (Style)」的平衡。

透過混用本地端的向量檢索 (Vector Search) 與關鍵字稀疏檢索 (BM25)，搭配開源的繁體中文語言模型，此系統能夠從企業年報中，精準抽取碳排放、員工權益、治理表現等具體數據，並輔助生成符合企業管理需求的專業文稿。

**核心技術與工具：**

- **LLM Engine**: Local Ollama (支援 **Streaming 串流輸出**，防止大模型讀取超時)
- **Vector Database**: ChromaDB (以 `BAAI/bge-small-zh-v1.5` 為 Embedding 模型)
- **Keyword Search**: BM25 (Okapi)
- **Reranker**: BAAI/bge-reranker-base (提升檢索精確度)
- **Web UI**: Streamlit
- **Document Processing**: `pathlib`, `json`, `python-docx`, `pypdf`
- **External Data**: `requests`, `BeautifulSoup` (爬蟲擴充知識庫)

---

## 硬體配置與模型建議 (Hardware Recommendations)

針對本地端執行 LLM，本系統建議如下配置：

- **GPU (推薦)**: NVIDIA RTX 3060/4060/5060 系列 (8GB VRAM 以上)
    - **8GB VRAM**: 建議運行 `qwen2.5:7b` 或 `llama3.1:8b`。使用 14B 模型尚可，但若使用 32B 以上模型會因顯存溢出至系統 RAM 而變得極慢。
    - **16GB VRAM+**: 可流暢運行 `qwen2.5:14b` 或 `qwen2.5:32b` 以獲得更穩定的邏輯抽取能力。
- **RAM**: 16GB 以上。

---

## 系統架構理念 (Architecture Design)

在處理 ESG 數據時，LLM 最大的隱患在於「幻覺 (Hallucination)」——即捏造不存在的數據。本系統採用 **Fact-then-Generate (先抽事實，後再生成)** 的三階段管線設計：

1. **混合檢索 (Hybrid Search) 搭配 HyDE**：
   系統會針對使用者的提問，先利用 LLM 去生成一段**假想的專業摘要 (HyDE)**，然後拿這段摘要去向量庫做語意搜尋，同時併用 BM25 關鍵字檢索確保年份與量化數據的命中率，最後以 **CrossEncoder Reranker** 重新排序，選出最相關的片段。

2. **本機事實抽取 (Streaming Fact Extraction)**：
   本系統已優化為 **Streaming 串流模式**，即使在一般家用 GPU 上運行較大的模型，也能有效避免 HTTP ReadTimeout 的問題。LLM 扮演「資料校對員」，嚴格地將文本轉化為**條列式的客觀事實 (Facts)**，並強迫附加來源引注。

3. **第三方查核與最終文稿生成**：
   系統產出事實後，建議透過 Stage 2 將 Prompt 貼入聯網強大模型（如 ChatGPT）進行二次事實核對，最後再套入專業顧問的寫作「風格樣本 (Style Report)」，產出 Markdown 報告並提供 `.docx` 下載。

---

## 核心模組實作展示

### 1. 混合檢索器 (Hybrid Searcher)

結合向量相似度分佈與 BM25 關鍵字頻率得分，並加上權重參數。

```python
def merge_hybrid(vec_hits, bm25_hits, vec_weight: float = 0.65, bm25_weight: float = 0.35):
    # 對 vector scores 與 bm25 scores 進行正規化後加權合併
    # 確保兩者在同一數量級上進行比較
    pass
```

### 2. 外部知識庫擴充 (Knowledge Crawler)

系統不只依賴 PDF，更設計了自動化爬蟲動態更新 `knowledge_bank`，抓取中立機構的 ESG 專有名詞定義。

```python
# snippet from crawler_knowledge.py
def crawl_wiki(keyword):
    url = f"https://zh.wikipedia.org/wiki/{keyword}"
    # ... 抓取內容並存入 docs/knowledge_bank/
```

---

## 挑戰與學習 (Challenges & Takeaways)

1. **穩定品質的權衡：**
   透過實測發現，雖然 7B 模型生成極快，但在「遵循複雜指令」與「數據抽取完整度」上，14B 以上模型有質的飛躍。專案中加入了**硬體自適應提示**，引導使用者在效能與精確度之間取得平衡。

2. **解決長文超時 (ReadTimeout)：**
   原本非串流 (Non-streaming) 的 API 調用在等待大模型推理時極易中斷。改採 `requests` 的串流處理後，不僅解決了 900s 的超時報錯，也大幅提升了使用者介面的響應體驗。

3. **防範 LLM 幻覺的成本交換：**
   透過「混合檢索 -> 重排序 -> 本機事實抽取 -> 外部事實查核」的流程，雖然增加了操作步驟，但這是在企業應用端不可妥協的代價。**數據的正確性遠比回應的迅速性更為重要**。

4. **Chunking 策略的優化：**
   優化了針對長篇 PDF 的文本段落正規化，能更好地保留「標題與分層資訊」作為 ChromaDB 的 Metadata，防止檢索結果斷章取義。
