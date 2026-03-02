import os
import requests
from bs4 import BeautifulSoup
import subprocess
import time
import json
import sys

# 1. 指定存放路徑
KNOWLEDGE_DIR = "./docs/knowledge_bank"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 預定義的爬蟲目標 (大部份企業編製 ESG 報告時常用到的準則與網站)
TARGETS = [
    {
        "name": "GRI_全球報告倡議組織",
        "url": "https://zh.wikipedia.org/wiki/%E5%85%A8%E7%90%83%E5%A0%B1%E5%91%8A%E5%80%A1%E8%AD%B0%E7%B5%84%E7%B9%94", 
        "selector": "div.mw-parser-output",
        "source": "Wikipedia"
    },
    {
        "name": "TCFD_氣候相關財務揭露",
        "url": "https://zh.wikipedia.org/wiki/%E6%B0%A3%E5%80%99%E7%9B%B8%E9%97%9C%E8%B2%A1%E5%8B%99%E6%8F%AD%E9%9C%B2%E5%B7%A5%E4%BD%9C%E5%B0%8F%E7%B5%84",
        "selector": "div.mw-parser-output",
        "source": "Wikipedia"
    },
    {
         "name": "ESG_環境社會和公司治理",
         "url": "https://zh.wikipedia.org/zh-tw/%E7%8E%AF%E5%A2%83%E3%80%81%E7%A4%BE%E4%BC%9A%E5%92%8C%E5%85%AC%E5%8F%B8%E6%B2%BB%E7%90%86",
         "selector": "div.mw-parser-output",
         "source": "Wikipedia"
    },
    {
         "name": "金管會_上市上櫃公司永續發展路徑圖",
         "url": "https://www.sfb.gov.tw/ch/home.jsp?id=1010&parentpath=0,4&mcustomize=multimessages_view.jsp&dataserno=202203030005",
         "selector": "div.page_content",
         "source": "金管會"
    },
    {
         "name": "溫室氣體盤查與碳足跡",
         "url": "https://zh.wikipedia.org/wiki/%E7%A2%B3%E8%B6%B3%E8%BF%B9",
         "selector": "div.mw-parser-output",
         "source": "Wikipedia"
    },
    {
         "name": "碳中和與淨零排放",
         "url": "https://zh.wikipedia.org/wiki/%E7%A2%B3%E4%B8%AD%E5%92%8C",
         "selector": "div.mw-parser-output",
         "source": "Wikipedia"
    }
]

def fetch_and_save(target):
    print(f"🔄 正在抓取: {target['name']} ({target['url']})")
    try:
        response = requests.get(target["url"], headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8' # Ensure correct encoding
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 尋找目標內容區塊
        content_div = soup.select_one(target["selector"])
        
        if content_div:
            # 清理一些不必要的標籤
            for script in content_div(["script", "style", "table", "sup"]): # Remove tables & sups from Wiki for cleaner text
                script.decompose()
            
            content = content_div.get_text(separator="\n\n", strip=True)
            
            file_path = os.path.join(KNOWLEDGE_DIR, f"{target['name']}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"TITLE: {target['name']}\n")
                f.write(f"SOURCE: {target['source']}\n")
                f.write(f"URL: {target['url']}\n\n")
                f.write(content)
                
            print(f"✅ 已成功儲存：{file_path}\n")
        else:
             print(f"❌ 找不到目標內容區塊: {target['name']}\n")
             
    except Exception as e:
        print(f"❌ 抓取失敗: {target['name']}, 錯誤: {e}\n")
    
    # 避免請求過於頻繁
    time.sleep(5)

if __name__ == "__main__":
    print("🚀 開始建立企業常用 ESG 知識庫...\n")
    
    # 清除舊的直接寫死的 JSON 檔案
    old_glossary = os.path.join(KNOWLEDGE_DIR, "ESG_Common_Glossary.json")
    if os.path.exists(old_glossary):
        os.remove(old_glossary)
        print("🗑️ 已刪除舊版的寫死名詞字典 ESG_Common_Glossary.json")
    
    # 抓取外部準則文章與詞彙
    for target in TARGETS:
        fetch_and_save(target)
    
    print("-" * 40)
    print("🔄 正在更新 knowledge_bank 向量與關鍵字索引...")
    try:
        # 取得目前執行此腳本的 python 執行檔路徑 (.venv 中的 python)
        python_exe = sys.executable
        subprocess.run([python_exe, "ingest.py", "--mode", "knowledge"], check=True)
        print("🎉 知識庫更新完成！請重啟 app.py 以套用最新的 Knowledge Bank。")
    except subprocess.CalledProcessError as e:
        print(f"❌ 知識庫更新失敗: {e}")
    except FileNotFoundError:
        print("❌ 找不到 ingest.py，請確認此腳本與 ingest.py 位於同一目錄下。")
