import os
import re
import json
import glob
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# CONFIGURATION
# ------------------------------
TABLE_DIR = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\tables_output"  # parent folder with Paper_X subfolders
KEYWORDS_PATH = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\all_papers_keywords.json"
OUTPUT_PATH = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\papers_metric_results.json"

FUZZY_THRESHOLD = 70


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def clean_num(val):
    """Extract and normalize numeric values (0–1 scale)."""
    try:
        num = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", str(val))[0])
        if "%" in str(val) and num > 1:
            num /= 100
        if num > 1:
            num /= 100
        return round(num, 4)
    except Exception:
        return None


def find_metric_in_table(df, metric):
    """Find the metric value by fuzzy matching headers/rows."""
    best_col, best_row = None, None
    best_col_score, best_row_score = 0, 0
    metric_lower = metric.lower()

    for col in df.columns:
        score = fuzz.partial_ratio(metric_lower, str(col).lower())
        if score > best_col_score:
            best_col_score, best_col = score, col

    first_col = df.columns[0]
    for idx, val in enumerate(df[first_col].astype(str)):
        score = fuzz.partial_ratio(metric_lower, val.lower())
        if score > best_row_score:
            best_row_score, best_row = score, idx

    col_val = None
    if best_col_score >= FUZZY_THRESHOLD:
        try:
            df_numeric = pd.to_numeric(df[best_col], errors="coerce")
            if df_numeric.notna().any():
                col_val = df_numeric.max()
        except Exception:
            pass

    row_val = None
    if best_row_score >= FUZZY_THRESHOLD:
        try:
            row_values = pd.to_numeric(df.iloc[best_row, 1:], errors="coerce")
            if row_values.notna().any():
                row_val = row_values.max()
        except Exception:
            pass

    return max(v for v in [col_val, row_val] if v is not None) if any([col_val, row_val]) else None


def find_baseline_metric_value(df, baseline, metric):
    """Find metric value corresponding to a baseline (row/col)."""
    baseline_l, metric_l = baseline.lower(), metric.lower()
    for r_idx, row in df.iterrows():
        for c_idx, val in enumerate(row):
            text = str(val).lower()
            if fuzz.partial_ratio(baseline_l, text) >= FUZZY_THRESHOLD:
                # try same row
                for col in df.columns:
                    if fuzz.partial_ratio(metric_l, str(col).lower()) >= FUZZY_THRESHOLD:
                        v = clean_num(row[col])
                        if v is not None:
                            return v
    return None


def semantic_fallback(df, query, model):
    """Semantic similarity fallback for missing values."""
    query_emb = model.encode(query, convert_to_tensor=True)
    text_cells = [f"{col}={val}" for _, row in df.iterrows() for col, val in row.items()]
    embeddings = model.encode(text_cells, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_text = text_cells[best_idx]
    num_match = re.search(r"[-+]?[0-9]*\.?[0-9]+", best_text)
    if num_match:
        val = float(num_match.group())
        if "%" in best_text and val > 1:
            val /= 100
        if val > 1:
            val /= 100
        return round(val, 4)
    return None


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load keywords for all papers
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        all_keywords = json.load(f)

    paper_results = {}

    # Loop through all paper folders (Paper_1, Paper_2, ...)
    paper_folders = [d for d in os.listdir(TABLE_DIR) if os.path.isdir(os.path.join(TABLE_DIR, d))]

    for paper_folder in sorted(paper_folders):
        paper_key = paper_folder.lower()  # e.g., "paper_1"
        paper_path = os.path.join(TABLE_DIR, paper_folder)

        if paper_key not in all_keywords:
            print(f"⚠️ Skipping {paper_folder} — not found in all_papers_keywords.json")
            continue

        data = all_keywords[paper_key]
        metrics = data.get("metrics", [])
        baselines = data.get("baselines", [])
        if not metrics or not baselines:
            print(f"⚠️ {paper_folder}: missing metrics or baselines.")
            continue

        print(f"\n📘 Processing {paper_folder}...")
        csv_files = glob.glob(os.path.join(paper_path, "*.csv"))
        if not csv_files:
            print(f"⚠️ No tables found for {paper_folder}.")
            continue

        paper_best = None

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue

            for metric in metrics:
                metric_val = find_metric_in_table(df, metric)
                if metric_val is None:
                    metric_val = semantic_fallback(df, metric, model)

                for baseline in baselines:
                    base_val = find_baseline_metric_value(df, baseline, metric)
                    if base_val is None:
                        base_val = semantic_fallback(df, f"{baseline} {metric}", model)

                    if metric_val and base_val:
                        if (paper_best is None) or (metric_val > paper_best["best_value"]):
                            paper_best = {
                                "metric": metric,
                                "baseline": baseline,
                                "best_value": float(metric_val)
                            }

        if paper_best:
            paper_results[paper_key] = paper_best
            print(f"✅ Found {paper_best}")
        else:
            print(f"⚠️ No matching metrics found for {paper_folder}")

    # Save all results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(paper_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 Saved final metric summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
