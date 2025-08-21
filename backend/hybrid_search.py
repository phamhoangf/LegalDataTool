#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Search Engine cho Legal Data Tool
Kết hợp BM25 (lexical) và Semantic Search (vector embeddings) để tăng hiệu quả tìm kiếm
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HybridSearchEngine:
    """
    Hybrid search engine kết hợp BM25 và semantic search
    
    Features:
    - BM25: Lexical/keyword matching 
    - Semantic: Vector embeddings với multilingual model
    - Smart weighting: Adaptive combination based on query type
    - Vietnamese optimization: Preprocessing cho tiếng Việt
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 bm25_weight: float = 0.3,
                 semantic_weight: float = 0.7,
                 device: str = None):
        """
        Initialize hybrid search engine
        
        Args:
            model_name: Sentence transformer model name (multilingual cho tiếng Việt)
            bm25_weight: Weight cho BM25 score (0-1)
            semantic_weight: Weight cho semantic score (0-1) 
            device: Device để run model ('cpu', 'cuda', None=auto)
        """
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Initialize models
        print(f"🔍 Initializing Hybrid Search Engine...")
        
        # Detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        try:
            # Load sentence transformer model
            print(f"📥 Loading semantic model: {model_name} on {device}")
            self.semantic_model = SentenceTransformer(model_name, device=device)
            print("✅ Semantic model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading semantic model: {e}")
            print("📥 Falling back to distiluse-base-multilingual-cased")
            self.semantic_model = SentenceTransformer('distiluse-base-multilingual-cased', device=device)
        
        # Initialize components
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.document_embeddings = None
        self.documents = []
        self.preprocessed_docs = []
        
        print(f"🎯 Hybrid weights: BM25={bm25_weight}, Semantic={semantic_weight}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text cho tiếng Việt
        
        Args:
            text: Raw text
            
        Returns:
            List of tokens
        """
        import re
        
        if not isinstance(text, str):
            return []
        
        # Lowercase và remove extra whitespace
        text = text.lower().strip()
        
        # Remove special characters nhưng giữ dấu câu quan trọng
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        
        # Tokenize đơn giản
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if len(token.strip()) > 0]
        
        return tokens
    
    def index_documents(self, documents: List[str], doc_ids: Optional[List[Any]] = None):
        """
        Index documents cho hybrid search
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
        """
        if not documents:
            print("⚠️ No documents to index")
            return
        
        print(f"📚 Indexing {len(documents)} documents...")
        
        # Store documents
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else list(range(len(documents)))
        
        # Preprocess documents
        print("🔤 Preprocessing documents...")
        self.preprocessed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Initialize BM25
        print("🔍 Building BM25 index...")
        self.bm25 = BM25Okapi(self.preprocessed_docs)
        
        # Initialize TF-IDF (fallback)
        print("📊 Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Keep Vietnamese stop words
            max_features=10000,
            ngram_range=(1, 2)
        )
        try:
            self.tfidf_vectorizer.fit(documents)
        except Exception as e:
            print(f"⚠️ TF-IDF error: {e}")
        
        # Create semantic embeddings
        print("🧠 Creating semantic embeddings...")
        try:
            self.document_embeddings = self.semantic_model.encode(
                documents, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"✅ Created embeddings: {self.document_embeddings.shape}")
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            self.document_embeddings = None
        
        print("🎯 Hybrid search index ready!")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               return_scores: bool = False,
               adaptive_weighting: bool = True) -> List[Dict[str, Any]]:
        """
        Hybrid search với BM25 + semantic
        
        Args:
            query: Search query
            top_k: Number of results to return
            return_scores: Whether to return detailed scores
            adaptive_weighting: Whether to adapt weights based on query
            
        Returns:
            List of search results with scores
        """
        if not self.documents:
            return []
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        if not query_tokens:
            return []
        
        results = []
        
        # 1. BM25 Search
        bm25_scores = np.zeros(len(self.documents))
        if self.bm25:
            try:
                bm25_scores = self.bm25.get_scores(query_tokens)
                # Normalize BM25 scores
                if np.max(bm25_scores) > 0:
                    bm25_scores = bm25_scores / np.max(bm25_scores)
            except Exception as e:
                print(f"⚠️ BM25 error: {e}")
        
        # 2. Semantic Search  
        semantic_scores = np.zeros(len(self.documents))
        if self.document_embeddings is not None:
            try:
                query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)
                semantic_scores = cosine_similarity(query_embedding, self.document_embeddings)[0]
                # Semantic scores are already 0-1
            except Exception as e:
                print(f"⚠️ Semantic search error: {e}")
        
        # 3. TF-IDF Fallback
        tfidf_scores = np.zeros(len(self.documents))
        if self.tfidf_vectorizer and np.sum(semantic_scores) == 0:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                docs_tfidf = self.tfidf_vectorizer.transform(self.documents)
                tfidf_scores = cosine_similarity(query_tfidf, docs_tfidf)[0]
            except Exception as e:
                print(f"⚠️ TF-IDF error: {e}")
        
        # 4. Adaptive Weighting
        bm25_weight = self.bm25_weight
        semantic_weight = self.semantic_weight
        
        if adaptive_weighting:
            # Adapt weights based on query characteristics
            query_length = len(query_tokens)
            
            if query_length <= 3:
                # Short queries: favor BM25 (exact matching)
                bm25_weight = 0.6
                semantic_weight = 0.4
            elif query_length > 10:
                # Long queries: favor semantic (context understanding)
                bm25_weight = 0.2
                semantic_weight = 0.8
            
            # If semantic search failed, use TF-IDF + BM25
            if np.sum(semantic_scores) == 0 and np.sum(tfidf_scores) > 0:
                semantic_scores = tfidf_scores
                semantic_weight = 0.5
                bm25_weight = 0.5
        
        # 5. Combine Scores
        combined_scores = (bm25_weight * bm25_scores + 
                          semantic_weight * semantic_scores)
        
        # 6. Rank and Return Results
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        for i, idx in enumerate(top_indices):
            result = {
                'doc_id': self.doc_ids[idx],
                'document': self.documents[idx],
                'rank': i + 1,
                'combined_score': float(combined_scores[idx])
            }
            
            if return_scores:
                result.update({
                    'bm25_score': float(bm25_scores[idx]),
                    'semantic_score': float(semantic_scores[idx]),
                    'tfidf_score': float(tfidf_scores[idx]) if np.sum(tfidf_scores) > 0 else 0.0,
                    'weights_used': {
                        'bm25_weight': bm25_weight,
                        'semantic_weight': semantic_weight
                    }
                })
            
            results.append(result)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute hybrid similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dict with different similarity scores
        """
        # Temporary index for comparison
        temp_docs = [text1, text2]
        temp_preprocessed = [self.preprocess_text(doc) for doc in temp_docs]
        
        # BM25 similarity
        bm25_score = 0.0
        try:
            temp_bm25 = BM25Okapi([temp_preprocessed[0]])
            bm25_scores = temp_bm25.get_scores(temp_preprocessed[1])
            bm25_score = bm25_scores[0] if len(bm25_scores) > 0 else 0.0
            # Normalize (simple approach)
            if bm25_score > 10:
                bm25_score = 1.0
            else:
                bm25_score = bm25_score / 10.0
        except:
            bm25_score = 0.0
        
        # Semantic similarity
        semantic_score = 0.0
        if self.semantic_model:
            try:
                embeddings = self.semantic_model.encode([text1, text2], convert_to_numpy=True)
                semantic_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            except:
                semantic_score = 0.0
        
        # TF-IDF similarity (fallback)
        tfidf_score = 0.0
        try:
            temp_tfidf = TfidfVectorizer()
            temp_matrix = temp_tfidf.fit_transform([text1, text2])
            tfidf_score = cosine_similarity(temp_matrix[0:1], temp_matrix[1:2])[0][0]
        except:
            tfidf_score = 0.0
        
        # Combined score
        combined_score = (self.bm25_weight * bm25_score + 
                         self.semantic_weight * semantic_score)
        
        return {
            'bm25_score': float(bm25_score),
            'semantic_score': float(semantic_score),
            'tfidf_score': float(tfidf_score),
            'combined_score': float(combined_score)
        }

# Factory function để tạo hybrid search engine
def create_hybrid_search_engine(config: Optional[Dict[str, Any]] = None) -> HybridSearchEngine:
    """
    Factory function để tạo hybrid search engine với config
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Configured HybridSearchEngine instance
    """
    default_config = {
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'bm25_weight': 0.3,
        'semantic_weight': 0.7,
        'device': None
    }
    
    if config:
        default_config.update(config)
    
    return HybridSearchEngine(**default_config)
