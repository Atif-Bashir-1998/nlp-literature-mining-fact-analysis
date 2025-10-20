from literature_miner import LiteratureMiner

def main():
    print("Starting Literature Mining Project...")
    print("=" * 50)
    
    # Initialize the miner
    miner = LiteratureMiner("Explainable Recommender Systems")
    
    # Step 1: Search for papers
    print("🔍 Searching for papers...")
    papers = miner.search_papers(limit=150)
    
    if len(papers) < 100:
        print(f"⚠️  Warning: Only found {len(papers)} papers. Trying alternative approach...")
        # You might need to implement additional API calls here
    else:
        print(f"✅ Found {len(papers)} papers")
    
    # Step 2: Save metadata
    print("💾 Saving metadata...")
    miner.save_metadata(papers)
    
    # Step 3: Enrich metadata
    print("✨ Enriching metadata...")
    enriched_papers = miner.enrich_metadata(papers)
    
    with open(miner.enriched_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_papers, f, indent=2, ensure_ascii=False)
    print(f"✅ Enriched metadata saved to {miner.enriched_metadata_file}")
    
    # Step 4: Download PDFs
    print("📥 Downloading PDFs...")
    miner.download_all_pdfs(papers)
    
    print("\n" + "=" * 50)
    print("🎉 Phase 1 (Data Collection) Completed!")
    print(f"📊 Total papers: {len(papers)}")
    print(f"💾 Metadata files created")
    print(f"📁 PDFs downloaded to: {miner.papers_dir}")

if __name__ == "__main__":
    import json
    main()