import os
import re
import json
from bs4 import BeautifulSoup
from collections import defaultdict

# --- Folders ---
base_dir = os.path.dirname(__file__)
tei_folder = os.path.join(base_dir, "paper_tei_files")
summary_js = os.path.join(base_dir, "summary_jsons")

os.makedirs(summary_js, exist_ok=True)

# --- Helper functions ---
def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def section_title(sec):
    head = sec.find("head")
    return head.text.strip().lower() if head else ""

def extract_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

def has_citation(text):
    return bool(re.search(r"\(\s*[A-Z][a-z]+ et al\.,? \d{4}\s*\)|\[\d+\]", text))

# --- Patterns for initial extraction ---
patterns = {
    "methods": r"\b(we propose|our method|our model|the proposed|proposed method|proposed approach|architecture|framework|approach|pipeline|algorithm|technique|system|strategy|procedure|mechanism|scheme|Transformer|BERT|GPT|RoBERTa|T5|LSTM|CNN|RNN|ResNet|ViT|Graph Neural Network|GNN|autoencoder|diffusion model|GAN|VAE)\b",
    "datasets": r"\b(dataset|datasets|corpus|corpora|benchmark|benchmarks|data set|data sets|collection|data source|data repository|training data|test data|evaluation data)\b",
    "baselines": r"\b(baseline|baselines|compare|compared to|against|previous method|previous work|competitor|existing approach|prior method|state[- ]of[- ]the[- ]art|SOTA|reference model|outperform|improve|achieve|gain|increase|boost|better performance|significant improvement|competitive results|superior results)\b",
    "metrics": r"\b(F1|F1[- ]score|accuracy|precision|recall|AUC|ROC[- ]AUC|BLEU|ROUGE|CIDEr|METEOR|MAE|RMSE|MSE|MAP|NDCG|log[- ]loss|perplexity|WER|CER)\b",
    "domains": r"\b(sentiment|medical|biomedical|vision|computer vision|dialogue|conversation|text classification|NER|named entity recognition|QA|question answering|recommendation|speech|audio|multimodal|finance|legal|social media|climate|translation|code)\b",
    "explanation_types": r"\b(SHAP|LIME|attention|self[- ]attention|saliency|saliency map|gradient|gradient[- ]based|feature importance|feature attribution|counterfactual|influence function|perturbation|occlusion|heatmap|explainability|interpretability)\b"
}

# =====================
# Step 1: Extract raw summaries per TEI file
# =====================
for filename in os.listdir(tei_folder):
    if filename.lower().endswith("_output.tei.xml"):
        file_path = os.path.join(tei_folder, filename)
        print(f" Processing: {filename}")

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

        # --- Save per-paper summary ---
        base_name = os.path.splitext(filename)[0].replace("_output.tei", "").lower()
        json_name = f"{base_name}_summary.json"
        output_path = os.path.join(summary_js, json_name)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f" Saved: {json_name}")

print(" All TEI files processed successfully!")

# =====================
# Step 2: Extract validated keywords and merge all papers
# =====================
CATEGORY_KEYWORDS = {
    "methods": [
        "bert","roberta","albert","deberta","xlnet","electra","gpt","gpt2","gpt3","gpt4",
        "t5","flan","llama","mistral","transformer","encoder-decoder","seq2seq","autoencoder","vae",
        "gan","diffusion","cnn","rnn","lstm","gru","gnn","graph neural network",
        "prompt","prompting","finetune","fine-tune","pretrain","pre-training",
        "instruction-tuning","few-shot","zero-shot","multitask","reinforcement learning",
        "rlhf","self-supervised","unsupervised","semi-supervised","supervised",
        "attention","self-attention","cross-attention","memory network",
        "adapter","mixture of experts","ensemble","meta-learning",
        "contrastive learning","representation learning",
        "davinci","curie","babbage","ada",
        "retrieval-augmented","rag","knowledge distillation"
    ],
    "domains": [
        "movie","film","music","news","book","review","business","finance",
        "medical","biomedical","clinical","health","hospital","pharma",
        "social","social media","twitter","reddit","facebook","youtube",
        "education","academic","science","math","history","law","legal",
        "policy","government","politics","environment","climate",
        "travel","tourism","hotel","transport","vehicle","traffic",
        "game","gaming","sports","ecommerce","shopping",
        "food","recipe","restaurant",
        "dialogue","conversation","chatbot","recommendation","vision",
        "speech","audio","multimodal","code","programming"
    ],
    "explanation_types": [
        "counterfactual","attention","feature","rationale","rule","saliency",
        "knowledge-grounded","kg","interpretability","explainability",
        "shap","lime","gradcam","influence function","feature attribution",
        "occlusion","attention weight","visualization","explanation","faithfulness",
        "transparency","heatmap","rationale extraction","post-hoc","intrinsic"
    ],
    "metrics": [
        "f1","f1-score","precision","recall","accuracy","auc","roc-auc",
        "bleu","rouge","meteor","cider","spice","wer","cer",
        "mae","mse","rmse","perplexity","logloss","ndcg","map",
        "hits@k","top-k accuracy","macro","micro","weighted","pearson","spearman"
    ],
    "baselines": [
        "bert","roberta","t5","gpt","gpt2","gpt3","xlnet","deberta",
        "tf-idf","svm","logistic","naive bayes","random forest",
        "xgboost","knn","cnn","rnn","lstm","gru","transformer",
        "seq2seq","bag-of-words","neural","linear","baseline","sota"
    ],
    "datasets": [
        "imdb","yelp","amazon","movielens","redial","e-redial",
        "coco","mscoco","flickr","flickr8k","flickr30k","squad","squad2.0",
        "twitter","wiki","wikipedia","bookcorpus","commoncrawl",
        "newsqa","triviaqa","natural questions","nq","hotpotqa",
        "cnn-dailymail","xsum","gigaword","multi30k","opus","wmt",
        "sst","sst2","snli","mnli","qnli","cola","rte","mrpc","sts-b",
        "glue","superglue","imagenet","cifar10","cifar100","mnist",
        "medqa","mimic","chexpert","pubmed","biosses",
        "reddit","stackexchange","arxiv","quora"
    ]
}

all_papers_keywords = {}

for filename in sorted(os.listdir(summary_js)):
    if filename.lower().startswith("paper_") and filename.lower().endswith("_summary.json"):
        file_path = os.path.join(summary_js, filename)
        print(f"🔍 Processing: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

        paper_key = filename.replace("_summary.json", "")
        all_papers_keywords[paper_key] = keywords_summary

output_path = os.path.join(base_dir, "all_papers_keywords.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_papers_keywords, f, indent=2, ensure_ascii=False)

print(f" Saved all keyword summaries to: {output_path}")
