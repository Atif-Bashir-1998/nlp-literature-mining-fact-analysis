import os
import json
import requests
import pandas as pd
from typing import List, Dict, Any
import time
import re
from urllib.parse import urlencode

class LiteratureMiner:
    def __init__(self, topic: str = "Explainable Recommender Systems"):
        self.topic = topic
        self.base_dir = "literature_analysis"
        self.papers_dir = os.path.join(self.base_dir, "papers")
        self.metadata_file = os.path.join(self.base_dir, "papers_metadata.json")
        self.enriched_metadata_file = os.path.join(self.base_dir, "papers_metadata_enriched.json")
        
        # Create directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.papers_dir, exist_ok=True)
        
        # Semantic Scholar API configuration
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1/paper/search"
        # self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        
        
    def search_papers(self, limit: int = 150) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar API
        """
        papers = []
        keywords = [
            "Explainable Recommender Systems",
            "XAI Recommender",
            "Interpretable Recommendation",
            "Explainable AI Recommendation",
        ]
        
        for keyword in keywords:
            print(f"Searching for: {keyword}")
            params = {
                'query': keyword,
                'limit': min(50, limit),
                'fields': 'paperId,title,abstract,authors,venue,year,referenceCount,citationCount,publicationTypes,publicationDate,externalIds,url,openAccessPdf'
            }
            
            try:
                response = requests.get(self.semantic_scholar_base, params=params)
                response.raise_for_status()
                data = response.json()
                
                for paper in data.get('data', []):
                    # Check if paper is not already in our list
                    if not any(p.get('paperId') == paper.get('paperId') for p in papers):
                        papers.append(paper)
                        
                print(f"Found {len(data.get('data', []))} papers for '{keyword}'")
                time.sleep(1)  # Be nice to the API
                
            except Exception as e:
                print(f"Error searching for '{keyword}': {e}")
                
        print(f"Total unique papers found: {len(papers)}")
        return papers
    
    def save_metadata(self, papers: List[Dict[str, Any]]):
        """
        Save paper metadata to JSON file
        """
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to {self.metadata_file}")
    
    def download_pdf(self, paper: Dict[str, Any]) -> str:
        """
        Download PDF for a paper if available
        """
        pdf_url = paper.get('openAccessPdf', {}).get('url')
        if not pdf_url:
            return None
            
        try:
            # Create filename: <year>_<firstAuthor>_<title>.pdf
            year = str(paper.get('year', 'unknown'))
            first_author = "unknown"
            if paper.get('authors') and len(paper['authors']) > 0:
                first_author = paper['authors'][0]['name'].split()[-1]  # Get last name
            
            title = paper.get('title', 'unknown')
            # Clean title for filename
            title_clean = re.sub(r'[^\w\s-]', '', title)[:50].replace(' ', '_')
            
            filename = f"{year}_{first_author}_{title_clean}.pdf"
            filepath = os.path.join(self.papers_dir, filename)
            
            # Download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            print(f"Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            print(f"Error downloading PDF for {paper.get('title')}: {e}")
            return None
    
    def download_all_pdfs(self, papers: List[Dict[str, Any]]):
        """
        Download PDFs for all papers that have open access
        """
        downloaded_count = 0
        for paper in papers:
            result = self.download_pdf(paper)
            if result:
                downloaded_count += 1
            time.sleep(0.5)  # Be nice to servers
            
        print(f"Downloaded {downloaded_count} PDFs")
    
    def enrich_metadata(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich metadata with additional information
        """
        enriched_papers = []
        
        for paper in papers:
            enriched_paper = paper.copy()
            
            # Extract affiliations from authors if available
            authors = paper.get('authors', [])
            affiliations = []
            countries = []
            
            for author in authors:
                if 'affiliations' in author and author['affiliations']:
                    for aff in author['affiliations']:
                        if aff not in affiliations:
                            affiliations.append(aff)
                            # Simple country extraction (this is basic - you might need more sophisticated approach)
                            country = self.extract_country(aff)
                            if country and country not in countries:
                                countries.append(country)
            
            enriched_paper['affiliations'] = affiliations
            enriched_paper['countries'] = countries
            enriched_paper['num_authors'] = len(authors)
            
            enriched_papers.append(enriched_paper)
        
        return enriched_papers
    
    def extract_country(self, affiliation: str) -> str:
        """
        Basic country extraction from affiliation string
        """
        # This is a simple approach - you might want to use a more sophisticated method
        country_keywords = {
            'usa': 'United States', 'united states': 'United States', 'us': 'United States',
            'china': 'China', 'chinese': 'China',
            'germany': 'Germany', 'german': 'Germany',
            'uk': 'United Kingdom', 'united kingdom': 'United Kingdom', 'britain': 'United Kingdom',
            'canada': 'Canada', 'australia': 'Australia', 'france': 'France',
            'japan': 'Japan', 'korea': 'South Korea', 'india': 'India',
            'italy': 'Italy', 'spain': 'Spain', 'netherlands': 'Netherlands',
            'brazil': 'Brazil', 'russia': 'Russia', 'switzerland': 'Switzerland'
        }
        
        affiliation_lower = affiliation.lower()
        for keyword, country in country_keywords.items():
            if keyword in affiliation_lower:
                return country
        
        return None

# Let's test our data collection
if __name__ == "__main__":
    miner = LiteratureMiner()
    
    print("Step 1: Searching for papers...")
    papers = miner.search_papers(limit=150)
    
    print("Step 2: Saving initial metadata...")
    miner.save_metadata(papers)
    
    print("Step 3: Enriching metadata...")
    enriched_papers = miner.enrich_metadata(papers)
    
    with open(miner.enriched_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_papers, f, indent=2, ensure_ascii=False)
    print(f"Enriched metadata saved to {miner.enriched_metadata_file}")
    
    print("Step 4: Downloading PDFs...")
    miner.download_all_pdfs(papers)
    
    print("Data collection phase completed!")