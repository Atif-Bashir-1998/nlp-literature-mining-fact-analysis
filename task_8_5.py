import os
import json
import re
from collections import defaultdict

# --- Define folder containing paper summaries ---
summary_folder = r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project\summary_jsons"  # ✅ change to your folder

# --- Define patterns for known research keywords ---
CATEGORY_KEYWORDS = {
    "methods": [
        "bert", "gpt", "t5", "transformer", "vae", "cnn", "rnn", "gnn", "llm",
        "attention", "autoencoder", "davinci", "flan", "prompt", "finetune"
    ],
    "domains": [
        "movie", "music", "news", "book", "business", "medical", "social", "education", "health",
        "travel", "game", "food", "transport"
    ],
    "explanation_types": [
        "counterfactual", "attention", "feature", "rationale", "rule", "saliency",
        "knowledge-grounded", "kg"
    ],
    "metrics": [
        "f1", "precision", "recall", "accuracy", "bleu", "rouge", "auc",
        "mae", "mse", "perplexity"
    ],
    "baselines": [
        "bert", "t5", "gpt", "tf-idf", "svm", "logistic", "random", "neural"
    ],
    "datasets": [
        "imdb", "yelp", "amazon", "movielens", "redial", "e-redial", "coco",
        "squad", "flickr", "twitter"
    ]
}

# --- Prepare container for all results ---
all_papers_keywords = {}

# --- Loop through all paper_X_summary.json files ---
for filename in sorted(os.listdir(summary_folder)):
    if filename.lower().startswith("paper_") and filename.lower().endswith("_summary.json"):
        file_path = os.path.join(summary_folder, filename)
        print(f"🔍 Processing: {filename}")

        # Load the summary JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract keywords per category
        keywords_summary = {}
        for category, sentences in data.items():
            if category not in CATEGORY_KEYWORDS:
                continue

            extracted = set()
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for kw in CATEGORY_KEYWORDS[category]:
                    if re.search(rf"\b{re.escape(kw)}\b", sentence_lower):
                        extracted.add(kw)

            if extracted:
                keywords_summary[category] = sorted(extracted)

        # Add this paper’s results to the combined output
        paper_key = filename.replace("_summary.json", "")
        all_papers_keywords[paper_key] = keywords_summary

# --- Save merged results ---
output_path = os.path.join(r"C:\Users\hp830\Desktop\OULU\University\Projects\NLP_Project", "all_papers_keywords.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_papers_keywords, f, indent=2, ensure_ascii=False)

print(f"✅ Saved all keyword summaries to: {output_path}")
