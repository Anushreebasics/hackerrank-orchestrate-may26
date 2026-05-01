import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(self, kb_path="knowledge_base.pkl"):
        with open(kb_path, "rb") as f:
            data = pickle.load(f)
            
        self.documents = data["documents"]
        self.embeddings_matrix = data["embeddings_matrix"]
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def search(self, query, top_k=5, company_filter=None):
        # Stage 1: Fast Bi-Encoder retrieval (get broad pool of candidates)
        fetch_k = max(20, top_k * 3)
        query_vec = self.bi_encoder.encode([query], convert_to_tensor=False)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, self.embeddings_matrix).flatten()
        
        # Sort indices by similarity
        indices = similarities.argsort()[::-1]
        
        candidates = []
        for idx in indices:
            sim = similarities[idx]
            if sim == 0:
                continue # no match at all
                
            doc = self.documents[idx].copy()
            
            # Optional company filter
            if company_filter and company_filter.lower() != "none":
                if doc["company"].lower() != company_filter.lower():
                    continue
                    
            candidates.append(doc)
            if len(candidates) >= fetch_k:
                break
                
        # If filtering gave too few results, backfill with non-filtered
        if len(candidates) < fetch_k:
            for idx in indices:
                sim = similarities[idx]
                if sim == 0:
                    continue
                doc = self.documents[idx].copy()
                # Simple check since dictionaries don't equate nicely if mutated, but here text is unique enough
                if not any(c["text"] == doc["text"] for c in candidates):
                    candidates.append(doc)
                if len(candidates) >= fetch_k:
                    break
                    
        if not candidates:
            return []
            
        # Stage 2: Cross-Encoder Re-ranking
        cross_inp = [[query, doc["text"]] for doc in candidates]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        for i in range(len(cross_scores)):
            candidates[i]["cross_score"] = cross_scores[i]
            
        # Sort by highly accurate cross_score
        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]
