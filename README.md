# nlp-literature-mining-fact-analysis
Design and implement a pipeline that automatically searches, collects, and analyzes research papers for a specific topic. The project should include automatic download of papers, metadata extraction, exploratory analysis (publication year, countries, universities, …), and fact-based analysis of trends and methods.


## Overview
This project automates the extraction of structured information from research papers (PDFs) and connects extracted entities into a **Knowledge Graph (KG)** for analysis and visualization.  
It focuses on research related to **Explainable and Interpretable Recommender Systems**, combining **Natural Language Processing**, **Information Extraction**, and **Graph Databases**.

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






