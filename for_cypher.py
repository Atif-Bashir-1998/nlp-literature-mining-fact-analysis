
import json
import csv

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
