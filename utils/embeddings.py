# utils/embeddings.py
"""
Utilitaire pour générer des embeddings textuels

Les embeddings sont des représentations vectorielles du texte qui
permettent de mesurer la similarité sémantique entre contenus.

Modèle utilisé : all-MiniLM-L6-v2 (léger et efficace)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingGenerator:
    """
    Générateur d'embeddings pour textes
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise le générateur d'embeddings
        
        Args:
            model_name: Nom du modèle Sentence-Transformer
                       (all-MiniLM-L6-v2 = 80MB, rapide et précis)
        """
        print(f"🔄 Chargement du modèle d'embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Modèle chargé avec succès")
        
        # Dimension des vecteurs produits par ce modèle
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Dimension des embeddings: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Génère des embeddings pour un ou plusieurs textes
        
        Args:
            texts: Texte unique (str) ou liste de textes (List[str])
            batch_size: Nombre de textes à traiter en parallèle
        
        Returns:
            Array numpy de shape (n_texts, embedding_dim)
            Si texte unique : shape (embedding_dim,)
        """
        # Convertir texte unique en liste
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Générer les embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,  # Pas de barre de progression
            convert_to_numpy=True
        )
        
        # Si c'était un texte unique, retourner un vecteur 1D
        if is_single:
            return embeddings[0]
        
        return embeddings
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Args:
            emb1: Premier embedding
            emb2: Deuxième embedding
        
        Returns:
            Score de similarité entre -1 et 1 (1 = identique)
        """
        # Normaliser les vecteurs
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Produit scalaire = similarité cosinus
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 3) -> List[tuple]:
        """
        Trouve les textes les plus similaires à une requête
        
        Args:
            query: Texte de requête
            candidates: Liste de textes candidats
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste de tuples (index, texte, score) triée par similarité décroissante
        """
        # Générer embeddings
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates)
        
        # Calculer similarités
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            score = self.cosine_similarity(query_emb, cand_emb)
            similarities.append((i, candidates[i], score))
        
        # Trier par score décroissant
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]


# Instance globale (singleton) pour éviter de recharger le modèle
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """
    Obtient l'instance globale du générateur d'embeddings
    (Pattern Singleton pour économiser la mémoire)
    
    Returns:
        Instance d'EmbeddingGenerator
    """
    global _embedding_generator
    
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    
    return _embedding_generator