import json
from collections import Counter

# ===== Load data =====
with open("index.json", encoding="utf-8") as f:
    index = json.load(f)

with open("all_papers_keywords.json", encoding="utf-8") as f:
    keywords = json.load(f)

# ===== Map paper IDs to publication years =====
paper_year = {p["id"]: int(p["year"]) for p in index if p.get("year")}

# ===== Collect methods & metrics by period =====
early_methods, recent_methods = [], []
early_metrics, recent_metrics = [], []

for pid, info in keywords.items():
    if pid not in paper_year:
        continue
    year = paper_year[pid]
    if year <= 2018:
        early_methods.extend(info.get("methods", []))
        early_metrics.extend(info.get("metrics", []))
    elif 2019 <= year <= 2025:
        recent_methods.extend(info.get("methods", []))
        recent_metrics.extend(info.get("metrics", []))

# ===== Count frequency =====
from collections import Counter
early_m = Counter(early_methods)
recent_m = Counter(recent_methods)
early_eval = Counter(early_metrics)
recent_eval = Counter(recent_metrics)

# ===== Identify Shifts =====
def compare_trends(early, recent):
    emerging = [k for k in recent if k not in early]
    declining = [k for k in early if k not in recent]
    stable = [k for k in early if k in recent]
    return emerging, declining, stable

emerging_methods, declining_methods, stable_methods = compare_trends(early_m, recent_m)
emerging_metrics, declining_metrics, stable_metrics = compare_trends(early_eval, recent_eval)

# ===== Display Results =====
print("\nMethod Trends")
print(f"Emerging (New in Recent): {', '.join(emerging_methods) or 'None'}")
print(f"Declining (Old, Not Seen Recently): {', '.join(declining_methods) or 'None'}")
print(f"Stable (Used in Both Eras): {', '.join(stable_methods) or 'None'}")

print("\nEvaluation Metric Trends")
print(f"Emerging: {', '.join(emerging_metrics) or 'None'}")
print(f"Declining: {', '.join(declining_metrics) or 'None'}")
print(f"Stable: {', '.join(stable_metrics) or 'None'}")
