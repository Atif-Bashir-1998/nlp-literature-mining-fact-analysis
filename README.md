# nlp-literature-mining-fact-analysis
Design and implement a pipeline that automatically searches, collects, and analyzes research papers for a specific topic. The project should include automatic download of papers, metadata extraction, exploratory analysis (publication year, countries, universities, …), and fact-based analysis of trends and methods.

---

## Objectives
1. Extract metadata and scientific entities (Authors, Methods, Datasets, Domains, Metrics) from PDF research papers.  
2. Convert PDFs into structured XML format using **GROBID**.  
3. Parse TEI XML files to extract research information automatically.  
4. Build and query a **Knowledge Graph** in **Neo4j** using Cypher.  
5. Visualize entity relationships with **NetworkX** and **Matplotlib**.
6. Methods trend from CF to Transformers; Amazon/MovieLens are top datasets.
7. Focus is on visual/text explanations; metric extraction.

---

## System Workflow

### Steps:
1. **PDF Processing:** Papers are converted into TEI XML files via GROBID.  
2. **XML Parsing:** Extract metadata and relationships using Python.  
3. **Graph Creation:** Entities are stored in Neo4j to build a knowledge network.  
4. **Visualization:** Graphs visualized via NetworkX for analysis.

---

## Tools and Technologies

| Component | Tool / Library | Version / Notes |
|------------|----------------|----------------|
| **Language** | Python | 3.9+ |
| **PDF to XML** | [GROBID](https://github.com/kermitt2/grobid) | via Docker container |
| **Graph Database** | [Neo4j](https://neo4j.com/) | 5.x |
| **IDE** | Visual Studio Code | Windows 10/11 |
| **Data Handling** | `json`, `os`, `glob`, `pandas` | Built-in / Latest |
| **Requests & Parsing** | `requests`, `xml.etree.ElementTree`, `BeautifulSoup` | |
| **Visualization** | `networkx`, `matplotlib` | |
| **Others** | `re`, `itertools`, `collections` | |

---

##  Running GROBID in Docker

1. **Install Docker Desktop:**  
    [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

2. **Pull and run GROBID:**
   ```bash
   docker pull lfoppiano/grobid:0.8.0
   docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.0

---
## 📑 Specifications and Implementation

| Specification | Python Scripts | Output / Description |
|---------------|----------------|--------------------|
| 8 | `task_8_1.py`, `task_8_2.py`, `task_8_3.py`, `task_8_4.py`, `task_8_5.py` | `task_8_1.py` : tei.xml files created by Grobid and save to `paper_tei_files/`, `task_8_2.py` : Table parsing by Grobid and save to `tables_output/`, `task_8_3.py`: key words extracted and create `all_papers_keywords.json` , `task_8_4.py`: all metric values extracted and create `papers_metric_results.json` , `task_8_5.py`: create `facts.jsonl` by combining `all_papers_keywords.json` and `papers_metric_results.json`|
| 2. Convert PDFs to TEI XML | `process_pdfs_grobid.py` | TEI XML files saved to `paper_tei_files/` |
| 3. Extract Paper Metadata | `parse_tei_metadata.py` | JSON files with title, authors, abstract, year |
| 4. Keyword Extraction from TEI | `extract_keywords_tei.py` | Per-paper summary JSON files in `summary_jsons/` |
| 5. Merge Keywords | `merge_keywords.py` | Combined `all_papers_keywords.json` |
| 6. Knowledge Graph Creation | `build_knowledge_graph.py` | `knowledge_graph.json` and `knowledge_graph_triples.csv` |
| 7. Knowledge Graph Visualization | `visualize_kg.py` | Graph plots saved to `Plot_KG/` |
| 8. Neo4j Node & Edge CSVs | `generate_neo4j_csv.py` | `neo4j_nodes.csv` and `neo4j_edges.csv` |
| 9. Load Graph & Query | `graph_query.py` | Search results for papers by domain, method, year |
| 10. Metrics Extraction | `extract_metrics.py` | JSON/CSV of metrics per paper |
| 11. Final JSONL Creation | `create_jsonl.py` | `papers_combined.jsonl` for downstream use |






