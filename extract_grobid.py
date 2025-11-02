import json
import os
import requests

# === CONFIG ===
index_path = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\index.json"
pdf_dir = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\papers_explainable_rs"
output_dir = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\paper_tei_files"
grobid_url = "http://localhost:8070/api/processFulltextDocument"

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# === LOAD INDEX.JSON ===
with open(index_path, "r", encoding="utf-8") as f:
    papers = json.load(f)

# === PROCESS EACH PAPER ===
for i, paper in enumerate(papers, start=1):
    pdf_file = os.path.join(pdf_dir, paper["file"])

    if not os.path.exists(pdf_file):
        print(f"⚠️ Skipping {paper['title']} — file not found: {pdf_file}")
        continue

    print(f"📄 Processing [{i}/{len(papers)}]: {paper['file']}")

    with open(pdf_file, "rb") as f:
        files = {'input': f}
        data = {'teiCoordinates': 'true'}
        response = requests.post(grobid_url, files=files, data=data)

    if response.status_code == 200:
        output_path = os.path.join(output_dir, f"{paper['id']}_output.tei.xml")
        with open(output_path, "wb") as out_f:
            out_f.write(response.content)
        print(f"✅ Saved TEI: {output_path}")
    else:
        print(f"❌ Error ({response.status_code}) for {paper['file']}: {response.text}")

print("🏁 All done.")
