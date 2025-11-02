import os
import re
import json
from bs4 import BeautifulSoup
from collections import defaultdict

# --- Folder containing TEI files ---
tei_folder = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\paper_tei_files"  # ✅ change to your folder

# --- Define extraction helpers ---
def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def section_title(sec):
    head = sec.find("head")
    return head.text.strip().lower() if head else ""

def extract_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

def has_citation(text):
    return bool(re.search(r"\(\s*[A-Z][a-z]+ et al\.,? \d{4}\s*\)|\[\d+\]", text))

# --- Define patterns ---
patterns = {
    "methods": r"\b(we propose|our method|our model|the proposed|architecture|framework|approach)\b",
    "datasets": r"\b(dataset|corpus|benchmark|data set)\b",
    "baselines": r"\b(baseline|compare|against|previous method|competitor)\b",
    "metrics": r"\b(F1|accuracy|precision|recall|AUC|BLEU|ROUGE|MAE|RMSE)\b",
    "domains": r"\b(sentiment|medical|vision|dialogue|text classification|NER|QA|recommendation|speech)\b",
    "explanation_types": r"\b(SHAP|LIME|attention|saliency|gradient|feature importance|counterfactual)\b"
}

# --- Process each TEI file ---
for filename in os.listdir(tei_folder):
    if filename.lower().endswith("_output.tei.xml"):
        file_path = os.path.join(tei_folder, filename)
        print(f"🔍 Processing: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            tei = BeautifulSoup(f, "xml")

        results = defaultdict(set)

        for div in tei.find_all("div"):
            title = section_title(div)
            if any(x in title for x in ["related", "background", "literature", "previous work"]):
                continue

            text = " ".join(p.text for p in div.find_all("p"))
            text = clean_text(text)
            sentences = extract_sentences(text)

            for sent in sentences:
                if has_citation(sent):
                    continue
                for key, pattern in patterns.items():
                    if re.search(pattern, sent, re.I):
                        results[key].add(clean_text(sent))

        summary = {key: list(values) for key, values in results.items()}

        # --- Create output filename ---
        base_name = os.path.splitext(filename)[0].replace("_output.tei", "").lower()
        json_name = f"{base_name}_summary.json"
        output_path = os.path.join(r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\summary_jsons", json_name)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {json_name}")

print("🎉 All TEI files processed successfully!")
