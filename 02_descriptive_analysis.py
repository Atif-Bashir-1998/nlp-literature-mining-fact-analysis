import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any
import os
from datetime import datetime

class DescriptiveAnalyzer:
    def __init__(self, metadata_file: str = "literature_analysis/papers_metadata_enriched.json"):
        self.metadata_file = metadata_file
        self.output_dir = "literature_analysis/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
        
        # Load data
        self.df = self.load_data()
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the data"""
        print("📂 Loading data...")
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} papers")
        return df
    
    def publication_trends_analysis(self):
        """Analyze publication trends over years"""
        print("\n📈 Analyzing publication trends...")
        
        # Filter out papers without year
        df_year = self.df[self.df['year'].notna()]
        df_year = df_year[df_year['year'] >= 2000]  # Focus on recent decades
        
        # Count publications per year
        yearly_counts = df_year['year'].value_counts().sort_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Line plot
        ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8, color=self.colors[0])
        ax1.set_title('Publication Trends: Explainable Recommender Systems', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Publications')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot
        ax2.bar(yearly_counts.index, yearly_counts.values, color=self.colors[1], alpha=0.7)
        ax2.set_title('Publications per Year', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Publications')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(yearly_counts.values):
            ax2.text(yearly_counts.index[i], v + 0.1, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/publication_trends.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/publication_trends.pdf', bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"📊 Publication Trends Summary:")
        print(f"   - Time span: {int(yearly_counts.index.min())} - {int(yearly_counts.index.max())}")
        print(f"   - Total publications: {yearly_counts.sum()}")
        print(f"   - Peak year: {yearly_counts.idxmax()} ({yearly_counts.max()} publications)")
        print(f"   - Average per year: {yearly_counts.mean():.1f}")
        
        return yearly_counts
    
    def venue_analysis(self):
        """Analyze publication venues"""
        print("\n🏛️  Analyzing publication venues...")
        
        # Count publications per venue
        venue_counts = self.df['venue'].value_counts()
        
        # Focus on top venues
        top_venues = venue_counts.head(15)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_venues)), top_venues.values, color=self.colors[2])
        plt.yticks(range(len(top_venues)), top_venues.index)
        plt.xlabel('Number of Publications')
        plt.title('Top Publication Venues for Explainable Recommender Systems', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(top_venues.values):
            plt.text(v + 0.1, i, str(v), va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_venues.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/top_venues.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"📊 Venue Analysis Summary:")
        print(f"   - Total unique venues: {len(venue_counts)}")
        print(f"   - Top 5 venues:")
        for i, (venue, count) in enumerate(top_venues.head().items()):
            print(f"     {i+1}. {venue}: {count} papers")
        
        return venue_counts
    
    def author_analysis(self):
        """Analyze authors and their productivity"""
        print("\n👥 Analyzing authors...")
        
        # Extract all authors
        all_authors = []
        for authors_list in self.df['authors']:
            if authors_list:
                for author in authors_list:
                    all_authors.append(author['name'])
        
        # Count publications per author
        author_counts = Counter(all_authors)
        top_authors = author_counts.most_common(20)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        authors_names = [author[0] for author in top_authors]
        authors_counts = [author[1] for author in top_authors]
        
        bars = plt.barh(range(len(authors_names)), authors_counts, color=self.colors[3])
        plt.yticks(range(len(authors_names)), authors_names)
        plt.xlabel('Number of Publications')
        plt.title('Most Prolific Authors in Explainable Recommender Systems', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(authors_counts):
            plt.text(v + 0.1, i, str(v), va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_authors.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/top_authors.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"📊 Author Analysis Summary:")
        print(f"   - Total unique authors: {len(author_counts)}")
        print(f"   - Top 5 authors:")
        for i, (author, count) in enumerate(top_authors[:5]):
            print(f"     {i+1}. {author}: {count} papers")
        
        return author_counts
    
    def geographic_analysis(self):
        """Analyze geographic distribution"""
        print("\n🌍 Analyzing geographic distribution...")
        
        # Extract countries from enriched metadata
        all_countries = []
        for countries_list in self.df['countries']:
            if countries_list:
                all_countries.extend(countries_list)
        
        # Count publications per country
        country_counts = Counter(all_countries)
        
        # Remove 'Unknown' if present
        if 'Unknown' in country_counts:
            del country_counts['Unknown']
        
        top_countries = country_counts.most_common(15)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        countries_names = [country[0] for country in top_countries]
        countries_counts = [country[1] for country in top_countries]
        
        bars = plt.barh(range(len(countries_names)), countries_counts, color=self.colors[4])
        plt.yticks(range(len(countries_names)), countries_names)
        plt.xlabel('Number of Publications')
        plt.title('Geographic Distribution of Research in Explainable Recommender Systems', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(countries_counts):
            plt.text(v + 0.1, i, str(v), va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/geographic_distribution.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"📊 Geographic Analysis Summary:")
        print(f"   - Total countries represented: {len(country_counts)}")
        print(f"   - Top 5 countries:")
        for i, (country, count) in enumerate(top_countries[:5]):
            print(f"     {i+1}. {country}: {count} papers")
        
        return country_counts
    
    def collaboration_network_analysis(self):
        """Build and analyze co-authorship network"""
        print("\n🔗 Analyzing collaboration network...")
        
        # Create collaboration graph
        G = nx.Graph()
        
        # Add authors and collaborations
        for _, row in self.df.iterrows():
            if row['authors']:
                authors = [author['name'] for author in row['authors']]
                
                # Add authors as nodes
                for author in authors:
                    G.add_node(author)
                
                # Add collaborations as edges
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        if G.has_edge(authors[i], authors[j]):
                            # Increase weight for multiple collaborations
                            G[authors[i]][authors[j]]['weight'] += 1
                        else:
                            G.add_edge(authors[i], authors[j], weight=1)
        
        print(f"   - Network nodes: {G.number_of_nodes()}")
        print(f"   - Network edges: {G.number_of_edges()}")
        
        # Calculate network statistics
        if G.number_of_nodes() > 0:
            print(f"   - Network density: {nx.density(G):.4f}")
            print(f"   - Connected components: {nx.number_connected_components(G)}")
            
            # Get largest connected component for visualization
            if nx.number_connected_components(G) > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                G_largest = G.subgraph(largest_cc)
                
                print(f"   - Largest component size: {len(largest_cc)} nodes")
                
                # Visualize the largest connected component
                self._visualize_network(G_largest)
            else:
                print("   ⚠️  No connected components found for visualization")
        
        return G
    
    def _visualize_network(self, G):
        """Visualize the collaboration network"""
        plt.figure(figsize=(15, 12))
        
        # Use spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Calculate node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 100 + 100 for node in G.nodes()]
        
        # Calculate edge widths based on weight
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=self.colors[5], alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                              alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Collaboration Network in Explainable Recommender Systems Research', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/collaboration_network.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/collaboration_network.pdf', bbox_inches='tight')
        plt.show()
    
    def citation_analysis(self):
        """Analyze citation patterns"""
        print("\n📚 Analyzing citation patterns...")
        
        # Filter papers with citation data
        df_citations = self.df[self.df['citationCount'].notna()]
        
        if len(df_citations) == 0:
            print("   ⚠️  No citation data available")
            return None
        
        # Basic citation statistics
        citation_stats = {
            'total_citations': df_citations['citationCount'].sum(),
            'average_citations': df_citations['citationCount'].mean(),
            'median_citations': df_citations['citationCount'].median(),
            'max_citations': df_citations['citationCount'].max(),
            'min_citations': df_citations['citationCount'].min()
        }
        
        # Top cited papers
        top_cited = df_citations.nlargest(10, 'citationCount')[['title', 'citationCount', 'year', 'venue']]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Citation distribution
        ax1.hist(df_citations['citationCount'], bins=20, color=self.colors[6], alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Citation Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Citation Count')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(True, alpha=0.3)
        
        # Citations vs Year
        ax2.scatter(df_citations['year'], df_citations['citationCount'], alpha=0.6, color=self.colors[7])
        ax2.set_title('Citations vs Publication Year', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Publication Year')
        ax2.set_ylabel('Citation Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/citation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/citation_analysis.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"📊 Citation Analysis Summary:")
        print(f"   - Total citations: {citation_stats['total_citations']}")
        print(f"   - Average citations per paper: {citation_stats['average_citations']:.1f}")
        print(f"   - Median citations: {citation_stats['median_citations']:.1f}")
        print(f"   - Most cited paper: {citation_stats['max_citations']} citations")
        
        print(f"\n🏆 Top 5 Most Cited Papers:")
        for i, (_, paper) in enumerate(top_cited.head().iterrows()):
            title_short = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
            print(f"   {i+1}. {title_short}")
            print(f"      Citations: {paper['citationCount']}, Year: {paper['year']}, Venue: {paper['venue']}")
        
        return citation_stats, top_cited
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("=" * 60)
        print("📊 COMPREHENSIVE DESCRIPTIVE ANALYSIS REPORT")
        print("=" * 60)
        
        # Run all analyses
        yearly_counts = self.publication_trends_analysis()
        venue_counts = self.venue_analysis()
        author_counts = self.author_analysis()
        country_counts = self.geographic_analysis()
        collaboration_network = self.collaboration_network_analysis()
        citation_results = self.citation_analysis()
        
        print("\n" + "=" * 60)
        print("✅ DESCRIPTIVE ANALYSIS COMPLETED!")
        print("=" * 60)
        
        # Save summary statistics
        summary = {
            'total_papers': len(self.df),
            'time_span': f"{int(yearly_counts.index.min())}-{int(yearly_counts.index.max())}",
            'unique_venues': len(venue_counts),
            'unique_authors': len(author_counts),
            'countries_represented': len(country_counts),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f'{self.output_dir}/analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Analysis results saved to: {self.output_dir}/")
        print(f"📈 Visualizations generated:")
        print(f"   - publication_trends.png")
        print(f"   - top_venues.png")
        print(f"   - top_authors.png")
        print(f"   - geographic_distribution.png")
        print(f"   - collaboration_network.png")
        print(f"   - citation_analysis.png")

def main():
    """Main function to run the descriptive analysis"""
    print("Starting Phase 2: Descriptive Analysis")
    print("=" * 50)
    
    analyzer = DescriptiveAnalyzer()
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()