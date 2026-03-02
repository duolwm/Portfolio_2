# 作品集：ESG RAG 企業永續報告書自動化檢索與查核生成系統

---

## 專案概述 (Project Overview)

本專案建構一個專為**處理並解析長篇 ESG (環境、社會、公司治理) 永續報告書**而設計的 RAG (Retrieval-Augmented Generation) 系統。本系統著重於「事實準確性 (Factuality)」與「專業寫作風格 (Style)」融合。

透過混用本地端的向量檢索 (Vector Search) 與關鍵字稀疏檢索 (BM25)，搭配開源的繁體中文語言模型 (Llama-3-Taiwan)，此系統能夠從企業年報中，精準抽取碳排放、員工權益、治理表現等具體數據，並輔助生成符合企業管理需求的專業文稿。

**核心技術與工具：**

- **LLM Engine**: Local Ollama (Llama-3-Taiwan-8b-instruct)
- **Vector Database**: ChromaDB (以 `BAAI/bge-small-zh-v1.5` 為 Embedding 模型)
- **Keyword Search**: BM25 (Okapi)
- **Web UI**: Streamlit
- **Document Processing**: `pathlib`, `json`, `BeautifulSoup` (爬蟲擴充知識庫)

---

## 系統架構理念 (Architecture Design)

在處理 ESG 數據時，LLM 最大的隱患在於「幻覺 (Hallucination)」——即捏造不存在的減碳目標或數據。為了解決這點，本系統採用 **Fact-then-Generate (先抽事實，後再生成)** 的三階段管線設計：

1. **混合檢索 (Hybrid Search) 搭配 HyDE**：
   系統會針對使用者的提問（如：「台積電 2023 年的水資源管理策略」），先利用 LLM 去生成一段**假想的專業摘要 (HyDE - Hypothetical Document Embeddings)**，然後拿這段摘要去向量庫做語意搜尋，同時併用 BM25 關鍵字檢索確保專有名詞的命中率，最後以 **CrossEncoder Reranker** 重新排序，選出最相關的片段。

2. **本機事實抽取 (Fact Extraction)**：
   不讓 LLM 直接在一大堆檢索文本裡長篇大論，而是約束它只准做「資料校對員」，嚴格地將文本轉化為**條列式的客觀事實 (Facts)**，並且每條後面強制附加來源文件的引註。

3. **第三方查核與最終文稿生成**：
   系統產出 JSON 格式的 Facts 供外部強效模型（如 ChatGPT）覆核（可選步驟），確認無誤後，最後再套入專業顧問的寫作「風格樣本 (Style Report)」，撰寫出一篇極具邏輯、段落分明且附檔可供下載 `.docx` 的永續分析報告。

---

## 核心模組實作展示

### 1. 混合檢索器 (Hybrid Searcher)

結合向量相似度分佈與 BM25 關鍵字頻率得分，並加上權重參數。

```python
# python snippet: 混合檢索 (Hybrid Search) 分數正規化與權重合併
def merge_hybrid(vec_hits, bm25_hits, vec_weight: float = 0.65, bm25_weight: float = 0.35):
    # 此處示範核心概念，詳見實際完整代碼
    # 1. 對 vector scores 正規化至 0~1
    # 2. 對 bm25 scores 正規化至 0~1
    # 3. 分別按 vec_weight + bm25_weight 加權後，排序選出 Top K 段落
    pass
```

### 2. 外部知識庫擴充 (Knowledge Crawler)

系統不只依賴使用者餵進來的 PDF，更設計了一套自動化爬蟲，動態建構並更新 `knowledge_bank`。

```python
import requests
from bs4 import BeautifulSoup
import os

# 自動抓取如維基百科等中立機構的 ESG 專有名詞定義
url = "https://zh.wikipedia.org/wiki/碳足跡"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
content = soup.select_one("div.mw-parser-output").get_text(strip=True)

# 存入知識庫給 ChromaDB 進行向量化 (ingest.py)
with open("docs/knowledge_bank/碳足跡.txt", "w", encoding="utf-8") as f:
    f.write(content)
```

### 3. Streamlit 使用者介面

設計了直觀的三段式介面，讓非工程人員也能流暢地產出並稽核報告：

- **階段①**：檢索並由本地 Ollama 抽取事實。
- **階段②**：將事實轉換為嚴格的查核 Prompt JSON 格式。
- **階段③**：套用專業語氣樣本 (Style Snippets) 生成最終 Markdown 文章，並提供一鍵下載為 Word `.docx`。

---

## 挑戰與學習 (Challenges & Takeaways)

1. **Chunking 策略的優化：**
   原先簡單的根據字數切分，常切斷上下文（尤其是表格與分段列表）。後來改用 `loaders_enterprise.py`，優化了針對大檔案（如 200 頁以上的 PDF）的文本段落正規化，能更好地保留「章節資訊」作為 ChromaDB 的 Metadata。

2. **防範 LLM 幻覺的成本交換：**
   導入 HyDE 和 Fact-Extraction 雖然顯著增加了回應的時間（延遲了約 5-10 秒），但這是在企業應用端不可妥協的代價。**數據的正確性遠比回應的迅速性更為重要**。

3. **解決中文編碼與爬蟲適應性：**
   一開始直接讓系統抓取外部 ESG 諮詢公司網站時遇到阻擋或網站架構更迭 (產生 404 Error)，最終將動態知識庫的來源移往更穩定的維基百科詞條等標竿網站，並設計獨立的 `crawler_knowledge.py` 與主系統 `app.py` 解耦。

---

