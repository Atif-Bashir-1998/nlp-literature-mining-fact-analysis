import os
import json
import requests
from rapidfuzz import fuzz

# ================= Configuration =================
API_KEY = ""
QUERY_VARIATIONS = [
    "Explainable Recommender Systems",
    "XAI Recommender",
    "Interpretable Recommendation",
    "Explainable Collaborative Filtering"
]
NUM_PAPERS = 100
SAVE_DIR = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\papers_explainable_rs"  # Change path as needed
INDEX_FILE = "index.json"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= Helper Functions =================
def search_papers(query, limit=50):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,authors,abstract,year,externalIds,url"
    headers = {"x-api-key": API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("data", [])
    return []

def get_pdf_url(paper):
    arxiv_id = paper.get("externalIds", {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return paper.get("url")  # fallback to Semantic Scholar URL

def sanitize_filename(s):
    return "".join(c if c.isalnum() or c in "_- " else "_" for c in s)

# ================= Main Loop =================
index_list = []
paper_count = 0
seen_titles = set()

for query in QUERY_VARIATIONS:
    papers = search_papers(query, limit=50)
    for paper in papers:
        if paper_count >= NUM_PAPERS:
            break

        title = paper.get("title", "")
        title_lower = title.lower()

        # Fuzzy match with lowercase, reduced threshold
        max_ratio = max(fuzz.token_sort_ratio(title_lower, q.lower()) for q in QUERY_VARIATIONS)
        if max_ratio < 40:  # Reduced threshold
            continue

        if title_lower in seen_titles:
            continue  # Skip duplicates
        seen_titles.add(title_lower)

        pdf_url = get_pdf_url(paper)
        if not pdf_url:
            print(f"Skipping {title}, PDF not found.")
            continue

        paper_count += 1
        local_filename = f"{paper_count:03d}_{sanitize_filename(title)}.pdf"
        local_path = os.path.join(SAVE_DIR, local_filename)

        # Download PDF
        try:
            r = requests.get(pdf_url, stream=True)
            if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded ({paper_count}/{NUM_PAPERS}): {title}")
            else:
                print(f"Skipping {title}, PDF not accessible.")
                paper_count -= 1
                continue
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            paper_count -= 1
            continue

        authors = ", ".join([a.get("name") for a in paper.get("authors", [])])
        index_list.append({
            "title": title,
            "authors": authors,
            "abstract": paper.get("abstract", ""),
            "year": str(paper.get("year", "")),
            "pdf_url": pdf_url,
            "id": f"paper_{paper_count}",
            "file": local_filename
        })

# Save index.json
with open(INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(index_list, f, ensure_ascii=False, indent=2)

print(f"\nDone! {len(index_list)} papers saved to {SAVE_DIR}. Index written to {INDEX_FILE}.")

