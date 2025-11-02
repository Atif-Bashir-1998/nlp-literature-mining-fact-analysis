import os
import re
import json
import time
import requests
from xml.etree import ElementTree as ET
from rapidfuzz import fuzz

# ------------------------------
# CONFIGURATION
# ------------------------------
QUERIES = [
    "Explainable Recommender Systems",
    "explainable recommendation",
    "explainable recommender",
    "explanation recommendation system"
]

OUTPUT_DIR = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\papers_explainable_rs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FETCH = 2000       # collect up to 200 papers per query
MAX_DOWNLOAD = 100    # download top 100
FUZZY_THRESHOLD = 63
BASE_URL = "https://export.arxiv.org/api/query"


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def clean_filename(text):
    text = re.sub(r"[\\/*?\"<>|:]", "_", text)
    return text.strip().replace("\n", " ")[:80]


def fetch_arxiv_results(query, max_results=1000):
    """Fetch arXiv metadata safely (no pagination errors)."""
    results = []
    start = 0
    batch_size = 100

    while len(results) < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(BASE_URL, params=params, timeout=20)
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch results for '{query}' (status {response.status_code})")
            break

        root = ET.fromstring(response.text)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if not entries:
            break  # stop when no more results

        for entry in entries:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            pdf_url = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib["href"]
                    break
            year = entry.find("{http://www.w3.org/2005/Atom}published").text[:4]
            results.append({
                "title": title,
                "abstract": summary,
                "year": year,
                "pdf_url": pdf_url
            })

        if len(entries) < batch_size:
            break  # fewer entries means last page
        start += batch_size
        time.sleep(1)  # polite delay

    return results


def download_pdf(url, path):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"⚠️ Download failed for {url}: {e}")
        return False


# ------------------------------
# MAIN SCRIPT
# ------------------------------
def main():
    all_papers = {}
    print("🔍 Stage 1: Fetching candidate papers...")

    for query in QUERIES:
        print(f"\n🔎 Searching for: '{query}'")
        results = fetch_arxiv_results(query, max_results=MAX_FETCH)
        for paper in results:
            if paper["pdf_url"] and paper["pdf_url"] not in all_papers:
                all_papers[paper["pdf_url"]] = paper

    print(f"\n✅ Collected {len(all_papers)} unique papers total.")

    # ------------------------------
    # FILTER STAGE
    # ------------------------------
    print(f"\n🎯 Stage 2: Filtering papers by relevance ('Explainable Recommender Systems')...")
    target_phrase = ["explainable recommender system","explainable recommender","explainable recommendation","xai recommender", "interpretable recommendation"]
    filtered = []

    for paper in all_papers.values():
        text = f"{paper['title']} {paper['abstract']}".lower()
        for syname in target_phrase:
            score = fuzz.partial_ratio(syname, text)
            if score >= FUZZY_THRESHOLD:
                filtered.append(paper)
                break

    print(f"✅ {len(filtered)} papers matched the explainable recommender topic.")

    # ------------------------------
    # DOWNLOAD STAGE
    # ------------------------------
    print(f"\n⬇️ Stage 3: Downloading top {min(len(filtered), MAX_DOWNLOAD)} relevant PDFs...")
    downloaded = []
    for i, paper in enumerate(filtered[:MAX_DOWNLOAD], start=1):
        safe_title = clean_filename(paper["title"])
        filename = f"{i:03d}_{safe_title}.pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)

        ok = download_pdf(paper["pdf_url"], filepath)
        if ok:
            paper["id"] = f"paper_{i}"
            paper["file"] = filename
            downloaded.append(paper)
            print(f"✅ Saved {filename}")

    # ------------------------------
    # SAVE INDEX
    # ------------------------------
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 Done! Downloaded {len(downloaded)} highly relevant papers.")
    print(f"📁 Metadata saved to: {index_path}")


if __name__ == "__main__":
    main()
