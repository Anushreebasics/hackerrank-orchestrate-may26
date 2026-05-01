import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    def __init__(self, cache_path="semantic_cache.pkl", model=None):
        self.cache_path = cache_path
        if model is None:
            # fallback model if one isn't provided
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = model

        # embeddings is a 2D numpy array (N x D)
        self.embeddings = None
        self.responses = []
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.responses = data.get('responses', [])
                self.embeddings = data.get('embeddings')
                if self.embeddings is None:
                    self.embeddings = np.zeros((0, self.model.get_embedding_dimension()))
                else:
                    self.embeddings = np.asarray(self.embeddings)
            except Exception:
                # If loading fails, start clean
                self.embeddings = np.zeros((0, self.model.get_embedding_dimension()))
                self.responses = []
        else:
            self.embeddings = np.zeros((0, self.model.get_embedding_dimension()))
            self.responses = []

    def _save(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'embeddings': self.embeddings, 'responses': self.responses}, f)
        except Exception:
            # best-effort save; don't block pipeline on failure
            pass

    def get_cached_response(self, query, threshold=0.98):
        qvec = self.model.encode([query], convert_to_tensor=False)
        qvec = np.atleast_2d(np.asarray(qvec).squeeze())
        if self.embeddings.shape[0] == 0:
            return None

        sims = cosine_similarity(qvec, self.embeddings).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return self.responses[best_idx]
        return None

    def add(self, query, response_dict):
        qvec = self.model.encode([query], convert_to_tensor=False)
        qvec = np.atleast_2d(np.asarray(qvec).squeeze())

        if self.embeddings.shape[0] == 0:
            self.embeddings = qvec
        else:
            self.embeddings = np.vstack([self.embeddings, qvec])

        self.responses.append(response_dict)
        self._save()
