import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# === Load Data ===
nodes_df = pd.read_csv("neo4j_nodes.csv")
edges_df = pd.read_csv("neo4j_edges.csv")

# === Create Graph ===
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

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# === Query Function ===
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


# === Example Query ===
domain_query = "movie"
method_query = "autoencoder"
start_year = 2018
end_year = 2023

results = find_papers(domain_query, method_query, start_year, end_year)

print(f"\n🎯 Papers in domain '{domain_query}' using method '{method_query}' ({start_year}–{end_year}):\n")
if results:
    for r in results:
        print(f"- {r['title']} ({r['year']}) | Authors: {', '.join(r['authors'])}")
else:
    print("No matching papers found.")

# === Optional Graph Visualization ===
#show_graph = True
#if show_graph:
#    plt.figure(figsize=(12, 8))
#    pos = nx.spring_layout(G, k=0.45)
#    node_colors = [
#        "skyblue" if G.nodes[n].get("label") == "Paper"
#        else "lightgreen" if G.nodes[n].get("label") == "Author"
#        else "orange" if G.nodes[n].get("label") == "Method"
#        else "violet" if G.nodes[n].get("label") == "Domain"
#        else "pink" if G.nodes[n].get("label") == "Dataset"
#        else "grey"
#        for n in G.nodes()
#    ]
#    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=4)
#    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "relation"), font_size=4)
#    plt.title("Local Knowledge Graph from neo4j_nodes.csv + neo4j_edges.csv")
#    plt.show()
#
