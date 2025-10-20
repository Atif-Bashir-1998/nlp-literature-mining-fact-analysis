import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
import re
import pdfplumber

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("NLTK data download failed - using fallback methods")

class TextMiningAnalyzer:
    def __init__(self, metadata_file: str = "literature_analysis/papers_metadata_enriched.json"):
        self.metadata_file = metadata_file
        self.output_dir = "literature_analysis/text_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 10)
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add academic stop words
        self.academic_stopwords = {
            'paper', 'study', 'research', 'method', 'approach', 'result', 'experiment',
            'analysis', 'propose', 'show', 'demonstrate', 'investigate', 'examine',
            'discuss', 'present', 'introduce', 'develop', 'implement', 'evaluate',
            'system', 'model', 'framework', 'algorithm', 'technique', 'process',
            'based', 'using', 'used', 'use', 'also', 'however', 'therefore',
            'additionally', 'furthermore', 'moreover', 'conclusion', 'abstract',
            'introduction', 'background', 'related', 'work', 'literature'
        }
        self.stop_words.update(self.academic_stopwords)
        
        # Load data
        self.df = self.load_data()
        self.corpus = None
        self.processed_texts = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the data"""
        print("📂 Loading data for text analysis...")
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} papers for text analysis")
        return df
    
    def extract_text_from_pdfs(self):
        """Extract text from downloaded PDFs"""
        print("\n📄 Extracting text from PDFs...")
        
        pdf_dir = "literature_analysis/papers"
        if not os.path.exists(pdf_dir):
            print("⚠️  PDF directory not found. Using abstracts only.")
            return
        
        pdf_texts = []
        successful_extractions = 0
        
        for idx, row in self.df.iterrows():
            paper_id = row.get('paperId', f"paper_{idx}")
            title = row.get('title', '')
            
            # Try to find matching PDF
            pdf_path = self._find_pdf_for_paper(title, pdf_dir)
            full_text = ""
            
            if pdf_path and os.path.exists(pdf_path):
                try:
                    # Extract text from PDF
                    doc = pdfplumber.open(pdf_path)
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    successful_extractions += 1
                except Exception as e:
                    print(f"❌ Error extracting text from {pdf_path}: {e}")
                    full_text = ""
            else:
                full_text = ""
            
            pdf_texts.append(full_text)
        
        self.df['full_text'] = pdf_texts
        print(f"✅ Successfully extracted text from {successful_extractions} PDFs")
    
    def _find_pdf_for_paper(self, title: str, pdf_dir: str) -> str:
        """Find PDF file for a given paper title"""
        if not os.path.exists(pdf_dir):
            return None
        
        # Clean title for matching
        title_clean = re.sub(r'[^\w\s]', '', title.lower())[:30]
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith('.pdf'):
                # Check if title substring is in filename
                if title_clean in filename.lower():
                    return os.path.join(pdf_dir, filename)
        
        return None
    
    def text_preprocessing(self, use_abstracts: bool = True):
        """Preprocess text data (abstracts or full text)"""
        print("\n🔧 Preprocessing text data...")
        
        if use_abstracts:
            texts = self.df['abstract'].fillna('').tolist()
            text_type = "abstracts"
        else:
            texts = self.df['full_text'].fillna('').tolist()
            text_type = "full texts"
        
        print(f"   Processing {len(texts)} {text_type}...")
        
        processed_texts = []
        
        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                processed_texts.append("")
                continue
                
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            processed_texts.append(cleaned_text)
            
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(texts)} {text_type}")
        
        self.processed_texts = processed_texts
        self.corpus = [text for text in processed_texts if text.strip() != ""]
        
        print(f"✅ Text preprocessing completed. {len(self.corpus)} non-empty documents.")
        
        # Create word cloud
        self._create_word_cloud()
        
        return processed_texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess a single text document"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _create_word_cloud(self):
        """Create word cloud from processed texts"""
        if not self.corpus:
            print("⚠️  No corpus available for word cloud")
            return
        
        all_text = ' '.join(self.corpus)
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: Most Frequent Terms in Explainable Recommender Systems Literature', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/word_cloud.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/word_cloud.pdf', bbox_inches='tight')
        plt.show()
    
    def explore_keyword_evolution(self):
        """Explore how key terms evolve over time"""
        print("\n📈 Analyzing keyword evolution over time...")
        
        # Define key terms related to explainable recommender systems
        key_terms = {
            'neural': ['neural', 'deep learning', 'transformer', 'embedding'],
            'explainability': ['explainability', 'interpretability', 'transparency'],
            'user_study': ['user study', 'user experiment', 'user evaluation'],
            'evaluation': ['evaluation metric', 'performance', 'accuracy'],
            'fairness': ['fairness', 'bias', 'ethical'],
            'visualization': ['visualization', 'visual', 'interface'],
            'knowledge_graph': ['knowledge graph', 'knowledge base', 'kg'],
            'causal': ['causal', 'causality', 'counterfactual']
        }
        
        # Prepare data for analysis
        df_analysis = self.df.copy()
        df_analysis['processed_abstract'] = self.processed_texts
        df_analysis = df_analysis[df_analysis['year'].notna()]
        df_analysis = df_analysis[df_analysis['year'] >= 2010]  # Focus on recent years
        
        # Calculate term frequencies per year
        yearly_term_freq = {}
        
        for term_group, terms in key_terms.items():
            yearly_freq = {}
            for year in sorted(df_analysis['year'].unique()):
                year_texts = df_analysis[df_analysis['year'] == year]['processed_abstract']
                total_docs = len(year_texts)
                if total_docs == 0:
                    continue
                
                term_count = 0
                for text in year_texts:
                    if any(term in text for term in terms):
                        term_count += 1
                
                yearly_freq[year] = term_count / total_docs * 100  # Percentage of papers
            
            yearly_term_freq[term_group] = yearly_freq
        
        # Create evolution plot
        plt.figure(figsize=(14, 8))
        
        for i, (term_group, frequencies) in enumerate(yearly_term_freq.items()):
            if frequencies:
                years = list(frequencies.keys())
                values = list(frequencies.values())
                plt.plot(years, values, marker='o', linewidth=2, 
                        label=term_group.replace('_', ' ').title(), 
                        color=self.colors[i % len(self.colors)])
        
        plt.xlabel('Year')
        plt.ylabel('Percentage of Papers (%)')
        plt.title('Evolution of Research Themes in Explainable Recommender Systems', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/theme_evolution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/theme_evolution.pdf', bbox_inches='tight')
        plt.show()
        
        return yearly_term_freq
    
    def perform_topic_modeling_lda(self, num_topics: int = 8):
        """Perform LDA topic modeling"""
        print(f"\n🧠 Performing LDA Topic Modeling with {num_topics} topics...")
        
        if not self.corpus:
            print("⚠️  No corpus available for topic modeling")
            return
        
        # Prepare documents for LDA - only use non-empty documents
        documents = [doc.split() for doc in self.corpus]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        print(f"✅ LDA Model trained. Coherence Score: {coherence_score:.4f}")
        
        # Visualize topics
        self._visualize_lda_topics(lda_model, corpus, dictionary, coherence_score)
        
        # Assign topics to documents - align with original DataFrame
        topic_assignments = []
        non_empty_indices = []  # Track which documents had content
        
        for i, doc in enumerate(corpus):
            topic_probs = lda_model.get_document_topics(doc)
            if topic_probs:
                main_topic = max(topic_probs, key=lambda x: x[1])[0]
                topic_assignments.append(main_topic)
                non_empty_indices.append(i)
            else:
                topic_assignments.append(-1)
                non_empty_indices.append(i)
        
        # Create full topic assignments for all rows (including empty ones)
        full_topic_assignments = [-1] * len(self.df)
        for idx, topic in zip(non_empty_indices, topic_assignments):
            full_topic_assignments[idx] = topic
        
        self.df['lda_topic'] = full_topic_assignments
        
        return lda_model, coherence_score
    
    def _visualize_lda_topics(self, lda_model, corpus, dictionary, coherence_score):
        """Visualize LDA topics"""
        # Get topic keywords
        topics = lda_model.print_topics(num_words=10)
        
        # Create topic visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, topic in enumerate(topics):
            if idx < len(axes):
                # Better way to extract words and weights
                topic_str = topic[1]
                # Split by '*' and clean
                components = topic_str.split(' + ')
                words = []
                weights = []
                
                for comp in components:
                    parts = comp.split('*')
                    if len(parts) == 2:
                        try:
                            weight = float(parts[0].strip())
                            word = parts[1].replace('"', '').strip()
                            words.append(word)
                            weights.append(weight)
                        except ValueError:
                            continue
                
                # Create horizontal bar chart for topic
                ax = axes[idx]
                y_pos = np.arange(len(words))
                ax.barh(y_pos, weights, color=self.colors[idx])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words)
                ax.invert_yaxis()
                ax.set_title(f'Topic {idx}\nKeywords', fontsize=10)
                ax.set_xlabel('Weight')
        
        plt.suptitle(f'LDA Topics\nCoherence Score: {coherence_score:.4f}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lda_topics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print topics in console
        print("\n📊 LDA Topics Discovered:")
        for idx, topic in enumerate(topics):
            print(f"Topic {idx}: {topic[1]}")
    
    def perform_bertopic_analysis(self):
        """Perform topic modeling using BERTopic (if available)"""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            print("\n🤖 Attempting BERTopic analysis...")
            
            if not self.corpus:
                print("⚠️  No corpus available for BERTopic")
                return
            
            # Initialize BERTopic
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(embedding_model=sentence_model, verbose=True)
            
            # Fit model
            topics, probabilities = topic_model.fit_transform(self.corpus)
            
            # Visualize topics
            fig = topic_model.visualize_topics()
            fig.write_html(f"{self.output_dir}/bertopic_topics.html")
            
            fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
            fig_barchart.write_html(f"{self.output_dir}/bertopic_barchart.html")
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            print("\n📊 BERTopic Topics:")
            print(topic_info.head(10))
            
            self.df['bertopic_topic'] = topics
            
            print("✅ BERTopic analysis completed")
            return topic_model
            
        except ImportError:
            print("❌ BERTopic not available. Install with: pip install bertopic sentence-transformers")
            return None
        except Exception as e:
            print(f"❌ BERTopic analysis failed: {e}")
            return None
    
    def analyze_topic_evolution(self):
        """Analyze how topics evolve over time"""
        print("\n📅 Analyzing topic evolution over time...")
        
        if 'lda_topic' not in self.df.columns:
            print("⚠️  Please run LDA topic modeling first")
            return
        
        # Prepare data
        df_evolution = self.df[self.df['year'].notna()].copy()
        df_evolution = df_evolution[df_evolution['year'] >= 2010]
        
        if len(df_evolution) == 0:
            print("⚠️  No data available for topic evolution analysis")
            return
        
        # Calculate topic distribution per year
        topic_evolution = {}
        years = sorted(df_evolution['year'].unique())
        
        for topic in range(8):  # Assuming 8 topics from LDA
            yearly_percentages = []
            for year in years:
                year_data = df_evolution[df_evolution['year'] == year]
                total_papers = len(year_data)
                if total_papers > 0:
                    topic_papers = len(year_data[year_data['lda_topic'] == topic])
                    percentage = (topic_papers / total_papers) * 100
                else:
                    percentage = 0
                yearly_percentages.append(percentage)
            topic_evolution[topic] = yearly_percentages
        
        # Create evolution plot
        plt.figure(figsize=(14, 8))
        
        for topic, percentages in topic_evolution.items():
            plt.plot(years, percentages, marker='o', linewidth=2, 
                    label=f'Topic {topic}', color=self.colors[topic % len(self.colors)])
        
        plt.xlabel('Year')
        plt.ylabel('Percentage of Papers (%)')
        plt.title('Topic Evolution in Explainable Recommender Systems Research', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/topic_evolution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/topic_evolution.pdf', bbox_inches='tight')
        plt.show()
        
        return topic_evolution
    
    def perform_text_clustering(self):
        """Perform K-means clustering on text data"""
        print("\n🔍 Performing text clustering with K-means...")
        
        if not self.corpus:
            print("⚠️  No corpus available for clustering")
            return
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.corpus)
        
        # Find optimal number of clusters using elbow method
        wcss = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            wcss.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss, marker='o', linewidth=2, color=self.colors[0])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.title('Elbow Method for Optimal Number of Clusters', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/clustering_elbow.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Choose optimal k
        optimal_k = 6
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_assignments = final_kmeans.fit_predict(tfidf_matrix)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_assignments)
        print(f"✅ Clustering completed. Silhouette Score: {silhouette_avg:.4f}")
        
        # Create full cluster assignments for all rows
        full_cluster_assignments = [-1] * len(self.df)
        corpus_index = 0
        for i in range(len(self.df)):
            if i < len(self.processed_texts) and self.processed_texts[i].strip():
                full_cluster_assignments[i] = cluster_assignments[corpus_index]
                corpus_index += 1
        
        self.df['text_cluster'] = full_cluster_assignments
        
        # Analyze cluster characteristics
        self._analyze_clusters(cluster_assignments, vectorizer, tfidf_matrix, optimal_k)
        
        return cluster_assignments, silhouette_avg
    
    def _analyze_clusters(self, clusters, vectorizer, tfidf_matrix, n_clusters):
        """Analyze and visualize clustering results"""
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Analyze top terms per cluster
        print(f"\n📊 Cluster Analysis ({n_clusters} clusters):")
        
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = (clusters == cluster_id)
            cluster_docs = tfidf_matrix[cluster_mask]
            
            if cluster_docs.shape[0] > 0:
                # Get average TF-IDF scores for this cluster
                avg_tfidf = np.asarray(cluster_docs.mean(axis=0)).flatten()
                
                # Get top terms
                top_indices = avg_tfidf.argsort()[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                
                cluster_size = cluster_docs.shape[0]  # Use shape[0] for sparse matrices
                cluster_analysis[cluster_id] = {
                    'size': cluster_size,
                    'top_terms': top_terms,
                    'avg_tfidf_scores': avg_tfidf[top_indices]
                }
                
                print(f"Cluster {cluster_id} (Size: {cluster_size} papers):")
                print(f"  Top terms: {', '.join(top_terms[:5])}")
        
        # Visualize clusters (using PCA for 2D projection)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2, random_state=42)
        tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=clusters, 
                            cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Text Clustering Visualization (PCA Projection)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/text_clustering.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/text_clustering.pdf', bbox_inches='tight')
        plt.show()
        
        return cluster_analysis
    
    def generate_comprehensive_report(self):
        """Generate comprehensive text mining report"""
        print("=" * 60)
        print("📊 COMPREHENSIVE TEXT MINING ANALYSIS REPORT")
        print("=" * 60)
        
        # Step 1: Extract text from PDFs (if available)
        self.extract_text_from_pdfs()
        
        # Step 2: Preprocess text (using abstracts as primary source)
        self.text_preprocessing(use_abstracts=True)
        
        # Step 3: Analyze keyword evolution
        keyword_evolution = self.explore_keyword_evolution()
        
        # Step 4: Perform topic modeling
        lda_model, coherence_score = self.perform_topic_modeling_lda(num_topics=8)
        
        # Step 5: Analyze topic evolution
        topic_evolution = self.analyze_topic_evolution()
        
        # Step 6: Perform text clustering
        clustering_results, silhouette_score = self.perform_text_clustering()
        
        # Step 7: Try BERTopic (optional)
        bertopic_model = self.perform_bertopic_analysis()
        
        print("\n" + "=" * 60)
        print("✅ TEXT MINING ANALYSIS COMPLETED!")
        print("=" * 60)
        
        # Save results
        results_summary = {
            'total_documents': len(self.corpus),
            'lda_coherence_score': coherence_score,
            'clustering_silhouette_score': silhouette_score,
            'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f'{self.output_dir}/text_analysis_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save processed data with topics
        self.df.to_csv(f'{self.output_dir}/papers_with_topics.csv', index=False)
        
        print(f"\n💾 Analysis results saved to: {self.output_dir}/")
        print(f"📈 Generated visualizations:")
        print(f"   - word_cloud.png")
        print(f"   - theme_evolution.png")
        print(f"   - lda_topics.png")
        print(f"   - topic_evolution.png")
        print(f"   - clustering_elbow.png")
        print(f"   - text_clustering.png")
        print(f"   - bertopic_topics.html (if BERTopic available)")

def main():
    """Main function to run text mining analysis"""
    print("Starting Phase 3: Text Mining and Topic Analysis")
    print("=" * 50)
    
    analyzer = TextMiningAnalyzer()
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()