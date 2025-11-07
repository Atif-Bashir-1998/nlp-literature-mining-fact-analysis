import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the merged KG JSON
with open("knowledge_graph.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

# Take the first 5 papers for the demo
demo_papers = list(kg.keys())[:20]

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

# Draw the graph
pos = nx.spring_layout(G, k=0.5, seed=42)
plt.figure(figsize=(15,10))

# Draw nodes with colors based on type
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
plt.show()

