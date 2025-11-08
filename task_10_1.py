import json
import os
import networkx as nx
import matplotlib.pyplot as plt

# =======================
# Step 1: Build KG & triples
# =======================

# Load files
with open("index.json", "r", encoding="utf-8") as f:
    papers_index = json.load(f)

with open("all_papers_keywords.json", "r", encoding="utf-8") as f:
    papers_keywords = json.load(f)

# Initialize the merged KG
knowledge_graph = {}
triples = []

for paper in papers_index:
    paper_id = paper["id"]
    title = paper["title"]
    authors = [a.strip() for a in paper["authors"].split(",")]

    # Get keywords if available
    keywords = papers_keywords.get(paper_id, {})
    methods = keywords.get("methods", [])
    domains = keywords.get("domains", [])
    explanation_types = keywords.get("explanation_types", [])
    datasets = keywords.get("datasets", [])
    metrics = keywords.get("metrics", [])

    # Build KG node
    knowledge_graph[paper_id] = {
        "title": title,
        "authors": authors,
        "methods": methods,
        "domains": domains,
        "explanation_types": explanation_types,
        "datasets": datasets,
        "metrics": metrics
    }

    # Build triples
    for author in authors:
        triples.append((paper_id, "AUTHORED_BY", author))
    for method in methods:
        triples.append((paper_id, "USES_METHOD", method))
    for domain in domains:
        triples.append((paper_id, "BELONGS_TO_DOMAIN", domain))
    for explanation_type in explanation_types:
        triples.append((paper_id, "GENERATES_EXPLANATION_TYPE", explanation_type))
    for dataset in datasets:
        triples.append((paper_id, "USES_DATASET", dataset))
    for metric in metrics:
        triples.append((paper_id, "EVALUATED_BY", metric))

# Save merged KG
with open("knowledge_graph.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)

# Save triples for Neo4j or CSV
with open("knowledge_graph_triples.csv", "w", encoding="utf-8") as f:
    f.write("subject,relation,object\n")
    for s, r, o in triples:
        f.write(f"{s},{r},{o}\n")

print("Knowledge Graph JSON and triples CSV generated successfully!")

# =======================
# Step 2: Visualize the KG (first 5 papers)
# =======================

# Load the merged KG JSON
with open("knowledge_graph.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

# Take the first 5 papers for the demo
demo_papers = list(kg.keys())[:5]

# Initialize graph
G = nx.DiGraph()

# Add nodes and edges for the demo papers
for paper_id in demo_papers:
    paper = kg[paper_id]
    G.add_node(paper_id, label=paper["title"], type="paper")

    # Authors
    for author in paper["authors"]:
        G.add_node(author, type="author")
        G.add_edge(paper_id, author, relation="AUTHORED_BY")

    # Methods
    for method in paper.get("methods", []):
        G.add_node(method, type="method")
        G.add_edge(paper_id, method, relation="USES_METHOD")

    # Domains
    for domain in paper.get("domains", []):
        G.add_node(domain, type="domain")
        G.add_edge(paper_id, domain, relation="BELONGS_TO_DOMAIN")

    # Explanation Types
    for expl_type in paper.get("explanation_types", []):
        G.add_node(expl_type, type="explanation_type")
        G.add_edge(paper_id, expl_type, relation="GENERATES_EXPLANATION_TYPE")

# =======================
# Step 3: Plot and save
# =======================

# Create output folder if it doesn't exist
os.makedirs("Plot_KG", exist_ok=True)

# Layout and colors
pos = nx.spring_layout(G, k=0.5, seed=42)
plt.figure(figsize=(15,10))
node_colors = []
for n, attr in G.nodes(data=True):
    if attr.get("type") == "paper":
        node_colors.append("skyblue")
    elif attr.get("type") == "author":
        node_colors.append("lightgreen")
    elif attr.get("type") == "method":
        node_colors.append("orange")
    elif attr.get("type") == "domain":
        node_colors.append("violet")
    elif attr.get("type") == "explanation_type":
        node_colors.append("pink")
    else:
        node_colors.append("grey")

nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=5, font_weight='bold', arrows=True)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4)

plt.title("Knowledge Graph Demo: Papers, Authors, Methods, Domains, Explanation Types")
plt.axis('off')

# Save plot
plot_path = os.path.join("Plot_KG", "knowledge_graph_demo.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f" Knowledge Graph plot saved: {plot_path}")
