# memory.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle

class VisualMemory:
    def __init__(self, similarity_threshold=0.85):
        """
        Visual memory system for object recognition
        
        Args:
            similarity_threshold: Cosine similarity threshold for same object
        """
        self.embeddings = []
        self.metadata = []  # Store {image_path, bbox, timestamp, etc}
        self.clusters = None
        self.threshold = similarity_threshold
    
    def add_observation(self, embedding, metadata=None):
        """Add new object observation to memory"""
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def find_similar(self, query_embedding, top_k=5):
        """Find most similar objects in memory"""
        if len(self.embeddings) == 0:
            return []
        
        embeddings_matrix = np.array(self.embeddings)
        query = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': idx,
                'similarity': similarities[idx],
                'metadata': self.metadata[idx],
                'embedding': self.embeddings[idx]
            })
        
        return results
    
    def cluster_memory(self, method='dbscan', eps=0.3, min_samples=2):
        """Cluster embeddings to discover object types"""
        if len(self.embeddings) < 2:
            print("Not enough observations to cluster")
            return
        
        embeddings_matrix = np.array(self.embeddings)
        
        if method == 'dbscan':
            # DBSCAN in cosine space
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            self.clusters = clustering.fit_predict(embeddings_matrix)
        
        return self.clusters
    
    def get_cluster_summary(self):
        """Get summary of discovered object types"""
        if self.clusters is None:
            self.cluster_memory()
        
        unique_clusters = set(self.clusters)
        unique_clusters.discard(-1)  # Remove noise cluster
        
        summary = {
            'num_object_types': len(unique_clusters),
            'num_observations': len(self.embeddings),
            'cluster_sizes': defaultdict(int)
        }
        
        for cluster_id in self.clusters:
            if cluster_id != -1:
                summary['cluster_sizes'][cluster_id] += 1
        
        return summary
    
    def save(self, filepath):
        """Save memory to disk"""
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'clusters': self.clusters,
            'threshold': self.threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Memory saved to {filepath}")
    
    def load(self, filepath):
        """Load memory from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        self.clusters = data['clusters']
        self.threshold = data['threshold']
        print(f"Memory loaded from {filepath}")