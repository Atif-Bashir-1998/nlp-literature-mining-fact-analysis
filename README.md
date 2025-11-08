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
| 9 | `task_09.py`| `task_09.py`: In this phase, we implemented an analytical layer to synthesize the thousands of extracted fact triples from the literature. We aggregated this data to identify high-level patterns, such as the most frequently used datasets and the most common research domains. A key part of this implementation was a comparative study that grouped papers by their publication date, allowing us to clearly distinguish trends. This analysis revealed a significant methodological shift, showing a decline in older methods like matrix factorization and a clear rise in modern deep learning approaches, including Transformers, Graph Neural Networks, and Large Language Models. We also confirmed that a few key datasets, like Amazon, MovieLens, and Yelp, serve as the standard benchmarks for the field |
| 10 | `task_10_1.py` , `task_10_1.py` | `task_10_1.py`: build, vusualize KG and save it to `Plot_KG/`  , `task_10_1.py`: test Cypher queries by giving keywords. "Papers in domain 'medical' using method 'transformer' (2018–2025):- Graph Learning (2025) | Authors: Feng Xia, Ciyuan Peng, Jing Ren, F. Febrinanto, Renqiang Luo, Vidya Saikrishna, Shuo Yu, Xiangjie Kong" |
| 11 | `task_11.py` | `task_11.py`: Compare early vs. recent research, Highlight shifts in methods and evaluation."Declining (Old, Not Seen Recently): knowledge distillation, memory network, Stable (Used in Both Eras): autoencoder, cnn, attention, lstm, reinforcement learning, rnn, unsupervised, Evaluation Metric Trends: Emerging: f1, f1-score, cer, meteor, weighted, mse, Declining: None, Stable: accuracy, auc, bleu, mae, ndcg, precision, recall, rmse, rouge, map" |







