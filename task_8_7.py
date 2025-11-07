import json
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
BASE_DIR = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project"
KEYWORDS_PATH = os.path.join(BASE_DIR, "all_papers_keywords.json")
METRICS_PATH = os.path.join(BASE_DIR, "papers_metric_results.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "papers_relationships.json")

# ------------------------------
# LOAD FILES
# ------------------------------
with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
    all_keywords = json.load(f)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metric_results = json.load(f)

# ------------------------------
# BUILD RELATIONSHIPS
# ------------------------------
relationships = []

for paper_id, data in all_keywords.items():
    # --- Methods ---
    if "methods" in data:
        for method in data["methods"]:
            relationships.append(f"{paper_id} -> usesMethod -> ({method})")

    # --- Domains ---
    if "domains" in data:
        for domain in data["domains"]:
            relationships.append(f"{paper_id} -> applicationDomain -> ({domain})")

    # --- Datasets ---
    if "datasets" in data:
        ds_list = ", ".join(data["datasets"])
        relationships.append(f"{paper_id} -> evaluatesOn -> ({ds_list})")

    # --- Explanation types ---
    if "explanation_types" in data:
        for exp_type in data["explanation_types"]:
            relationships.append(f"{paper_id} -> explanationType -> ({exp_type})")

    # --- Metrics ---
    if "metrics" in data:
        metrics_list = ", ".join(data["metrics"])
        relationships.append(f"{paper_id} -> evaluationMetric -> ({metrics_list})")

    # --- Metric Results (performance) ---
    if paper_id in metric_results:
        res = metric_results[paper_id]
        metric = res.get("metric")
        baseline = res.get("baseline")
        best_val = res.get("best_value")

        # If you later calculate improvement, add it here (currently only value)
        relationships.append(
            f"{paper_id} -> achievesImprovementOver -> ({baseline}, {metric.upper()}={best_val})"
        )

# ------------------------------
# SAVE OUTPUT
# ------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(relationships, f, indent=2, ensure_ascii=False)

print(f"✅ Relationships saved to: {OUTPUT_PATH}")
