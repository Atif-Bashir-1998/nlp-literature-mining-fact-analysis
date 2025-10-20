import json
import os

def verify_data():
    base_dir = "literature_analysis"
    metadata_file = os.path.join(base_dir, "papers_metadata.json")
    enriched_file = os.path.join(base_dir, "papers_metadata_enriched.json")
    papers_dir = os.path.join(base_dir, "papers")
    
    print("Data Verification Report")
    print("=" * 40)
    
    # Check metadata files
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"✅ papers_metadata.json: {len(metadata)} papers")
    else:
        print("❌ papers_metadata.json: Not found")
    
    if os.path.exists(enriched_file):
        with open(enriched_file, 'r') as f:
            enriched = json.load(f)
        print(f"✅ papers_metadata_enriched.json: {len(enriched)} papers")
        
        # Show sample of data
        if enriched:
            print("\nSample paper:")
            sample = enriched[0]
            print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"  Year: {sample.get('year', 'N/A')}")
            print(f"  Authors: {len(sample.get('authors', []))}")
            print(f"  Venue: {sample.get('venue', 'N/A')}")
            
    else:
        print("❌ papers_metadata_enriched.json: Not found")
    
    # Check PDFs
    if os.path.exists(papers_dir):
        pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
        print(f"✅ PDFs downloaded: {len(pdf_files)} files")
    else:
        print("❌ PDFs directory: Not found")

if __name__ == "__main__":
    verify_data()