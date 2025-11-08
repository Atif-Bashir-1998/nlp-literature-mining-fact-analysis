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
base_dir = os.path.dirname(__file__)
SAVE_DIR = os.path.join(base_dir, "papers_explainable_rs")
INDEX_FILE = "index.json"
OUTPUT_DIR = os.path.join(base_dir, "paper_tei_files")
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ================= Main Loop: Download PDFs =================
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

# ================= Main Loop: Process PDFs with GROBID =================
with open(INDEX_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

for i, paper in enumerate(papers, start=1):
    pdf_file = os.path.join(SAVE_DIR, paper["file"])

    if not os.path.exists(pdf_file):
        print(f"Skipping {paper['title']} — file not found: {pdf_file}")
        continue

    print(f"📄 Processing [{i}/{len(papers)}]: {paper['file']}")

    with open(pdf_file, "rb") as f:
        files = {'input': f}
        data = {'teiCoordinates': 'true'}
        response = requests.post(GROBID_URL, files=files, data=data)

    if response.status_code == 200:
        output_path = os.path.join(OUTPUT_DIR, f"{paper['id']}_output.tei.xml")
        with open(output_path, "wb") as out_f:
            out_f.write(response.content)
        print(f" Saved TEI: {output_path}")
    else:
        print(f" Error ({response.status_code}) for {paper['file']}: {response.text}")

print(" All done. TEI files are ready in 'paper_tei_files'.")
