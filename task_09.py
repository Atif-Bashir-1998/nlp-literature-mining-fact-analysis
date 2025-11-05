import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re

# -------------------------------------------------------------
# 1. Load the metadata
# -------------------------------------------------------------
with open("literature_analysis/papers_metadata.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

df = pd.json_normalize(papers)
print("Total papers:", len(df))
df = df.fillna("")

# -------------------------------------------------------------
# 2. Extract basic metadata fields
# -------------------------------------------------------------
core = df[['paperId', 'title', 'year', 'venue', 'citationCount', 'referenceCount', 'abstract', 'authors']]
core['n_authors'] = core['authors'].apply(lambda a: len(a) if isinstance(a, list) else 0)

# Get author-country approximation (if available later)
print(core.head(2))

# -------------------------------------------------------------
# 3. Publication trend
# -------------------------------------------------------------
plt.figure(figsize=(10,5))
sns.countplot(x="year", data=core, palette="crest")
plt.title("Publications per Year (Explainable Recommender Systems)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 4. Top Venues, Citations, Authors
# -------------------------------------------------------------
top_venues = core['venue'].value_counts().head(10)
print("\nTop Venues:\n", top_venues)

# Most cited
top_cited = core.sort_values('citationCount', ascending=False).head(10)
print("\nMost Cited Papers:\n", top_cited[['title','citationCount','year','venue']])

# Extract author names
authors_list = []
for idx, row in core.iterrows():
    for a in row['authors']:
        authors_list.append({'paperId': row['paperId'], 'author': a['name']})
auth_df = pd.DataFrame(authors_list)
top_authors = auth_df['author'].value_counts().head(10)
print("\nTop Authors:\n", top_authors)

plt.figure(figsize=(8,4))
sns.barplot(x=top_authors.values, y=top_authors.index)
plt.title("Most Frequent Authors")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 5. Method Extraction from Abstracts
# -------------------------------------------------------------
method_keywords = {
    "Matrix Factorization": r"matrix factorization|mf-?based",
    "Graph Neural Network": r"\bGNN|graph neural",
    "Transformer": r"transformer|bert|gpt",
    "Reinforcement Learning": r"reinforcement learning|rl-based",
    "LLM-based": r"large language model|llm"
}
method_records = []

for i, row in core.iterrows():
    text = row['abstract'].lower()
    for method, pattern in method_keywords.items():
        if re.search(pattern, text):
            method_records.append({'paperId': row['paperId'], 'year': row['year'], 'method': method})

methods_df = pd.DataFrame(method_records)
if not methods_df.empty:
    trend = methods_df.groupby(['year','method']).size().reset_index(name='count')
    plt.figure(figsize=(10,6))
    sns.lineplot(data=trend, x='year', y='count', hue='method', marker='o')
    plt.title("Algorithm Trends Over Time")
    plt.tight_layout()
    plt.show()

print("\nDetected Method Counts:\n", methods_df['method'].value_counts())

# -------------------------------------------------------------
# 6. Application Domains extraction
# -------------------------------------------------------------
domain_keywords = ["health", "movie", "tourism", "education", "social", "agriculture", "e-commerce", "music"]
domain_records = []

for i, row in core.iterrows():
    text = row['abstract'].lower()
    for dom in domain_keywords:
        if dom in text:
            domain_records.append({'paperId': row['paperId'], 'year': row['year'], 'domain': dom})

domain_df = pd.DataFrame(domain_records)
print("\nDomain Count Summary:")
print(domain_df['domain'].value_counts())

plt.figure(figsize=(8,4))
sns.countplot(y="domain", data=domain_df, order=domain_df['domain'].value_counts().index, palette="flare")
plt.title("Application Domains (Extracted from Abstracts)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 7. Collaboration Network
# -------------------------------------------------------------
edges = auth_df.merge(auth_df, on='paperId')
edges = edges[edges['author_x'] != edges['author_y']]
G = nx.from_pandas_edgelist(edges, 'author_x', 'author_y')

print("\nCo-authorship network nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
centrality = nx.degree_centrality(G)
top_collaborators = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop Collaboration Hubs:\n", top_collaborators)

plt.figure(figsize=(8,6))
nx.draw(G, node_size=50, node_color="lightblue", edge_color="gray")
plt.title("Co-authorship Collaboration Network")
plt.show()

# -------------------------------------------------------------
# 8. Save Summary Files
# -------------------------------------------------------------
summary_dir = "analysis_output"
import os
os.makedirs(summary_dir, exist_ok=True)

core.to_csv(f"{summary_dir}/papers_overview.csv", index=False)
methods_df.to_csv(f"{summary_dir}/method_trends.csv", index=False)
domain_df.to_csv(f"{summary_dir}/domains.csv", index=False)
auth_df.to_csv(f"{summary_dir}/authors.csv", index=False)
print("\nAnalysis exported to:", summary_dir)
