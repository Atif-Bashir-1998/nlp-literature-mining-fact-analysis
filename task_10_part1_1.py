import json

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
