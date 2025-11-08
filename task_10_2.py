import json
import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =======================
# Step 1: Generate Neo4j CSVs
# =======================

# Load files
with open("knowledge_graph.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

with open("index.json", "r", encoding="utf-8") as f:
    index = json.load(f)

# Create a mapping from paper_id to year
paper_year_map = {paper["id"]: int(paper["year"]) for paper in index}

# Prepare CSV files
node_file = "neo4j_nodes.csv"
edge_file = "neo4j_edges.csv"

# Nodes set to avoid duplicates
nodes = set()
edges = []

# Add nodes and edges
for paper_id, data in kg.items():
    # Paper node with year property
    year = paper_year_map.get(paper_id, None)
    nodes.add((paper_id, "Paper", data.get("title", ""), year))

    # Authors
    for author in data.get("authors", []):
        nodes.add((author, "Author", "", None))
        edges.append((paper_id, "AUTHORED_BY", author))

    # Methods
    for method in data.get("methods", []):
        nodes.add((method, "Method", "", None))
        edges.append((paper_id, "USES_METHOD", method))

    # Domains
    for domain in data.get("domains", []):
        nodes.add((domain, "Domain", "", None))
        edges.append((paper_id, "BELONGS_TO_DOMAIN", domain))

    # Explanation Types
    for expl_type in data.get("explanation_types", []):
        nodes.add((expl_type, "ExplanationType", "", None))
        edges.append((paper_id, "GENERATES_EXPLANATION_TYPE", expl_type))

    # Datasets
    for dataset in data.get("datasets", []):
        nodes.add((dataset, "Dataset", "", None))
        edges.append((paper_id, "USES_DATASET", dataset))

    # Metrics
    for metric in data.get("metrics", []):
        nodes.add((metric, "Metric", "", None))
        edges.append((paper_id, "EVALUATED_BY", metric))

# Write nodes CSV
with open(node_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label", "title", "year"])
    for n in nodes:
        writer.writerow(n)

# Write edges CSV
with open(edge_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "relation", "target"])
    for e in edges:
        writer.writerow(e)

print(f"Nodes saved to {node_file}")
print(f"Edges saved to {edge_file}")
print("Neo4j-ready CSV files generated successfully!")

# =======================
# Step 2: Build Graph
# =======================

# Load Data
nodes_df = pd.read_csv("neo4j_nodes.csv")
edges_df = pd.read_csv("neo4j_edges.csv")

# Create Graph
G = nx.DiGraph()

# Add nodes with attributes
for _, row in nodes_df.iterrows():
    node_id = row["id"]
    label = row.get("label", "")
    title = row.get("title", "")
    year = row.get("year", None)
    G.add_node(node_id, label=label, title=title, year=year)

# Add edges (relationships)
for _, row in edges_df.iterrows():
    source = str(row["source"]).strip()
    target = str(row["target"]).strip()
    relation = str(row["relation"]).strip()
    G.add_edge(source, target, relation=relation)

print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# =======================
# Step 3: Query Function
# =======================

def find_papers(domain, method, start_year, end_year):
    results = []
    for node, data in G.nodes(data=True):
        if data.get("label") == "Paper":
            year = data.get("year")
            if pd.notna(year):
                try:
                    year = int(year)
                except ValueError:
                    continue
                if start_year <= year <= end_year:
                    # Check domain and method connections
                    if G.has_edge(node, domain) and G.has_edge(node, method):
                        authors = [
                            n for n in G.successors(node)
                            if G.edges[node, n]["relation"] == "AUTHORED_BY"
                        ]
                        results.append({
                            "id": node,
                            "title": data.get("title"),
                            "year": year,
                            "authors": authors
                        })
    return results

# =======================
# Step 4: Example Query
# =======================

domain_query = "medical"
method_query = "transformer"
start_year = 2018
end_year = 2025

results = find_papers(domain_query, method_query, start_year, end_year)

print(f"\nPapers in domain '{domain_query}' using method '{method_query}' ({start_year}–{end_year}):\n")
if results:
    for r in results:
        print(f"- {r['title']} ({r['year']}) | Authors: {', '.join(r['authors'])}")
else:
    print("No matching papers found.")

# =======================
# Step 5: Optional Graph Visualization
# =======================
# Uncomment to visualize
# show_graph = True
# if show_graph:
#     plt.figure(figsize=(12, 8))
#     pos = nx.spring_layout(G, k=0.45)
#     node_colors = [
#         "skyblue" if G.nodes[n].get("label") == "Paper"
#         else "lightgreen" if G.nodes[n].get("label") == "Author"
#         else "orange" if G.nodes[n].get("label") == "Method"
#         else "violet" if G.nodes[n].get("label") == "Domain"
#         else "pink" if G.nodes[n].get("label") == "Dataset"
#         else "grey"
#         for n in G.nodes()
#     ]
#     nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=4)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "relation"), font_size=4)
#     plt.title("Local Knowledge Graph from neo4j_nodes.csv + neo4j_edges.csv")
#     plt.show()
