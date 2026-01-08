# agents/profiling_agent.py
"""
Profiling Agent - Analyse et crée des profils d'apprentissage

Rôle :
- Analyser l'historique des interactions utilisateur
- Identifier le style d'apprentissage via clustering
- Évaluer le niveau (débutant, intermédiaire, avancé)
- Détecter les forces et faiblesses
- Stocker le profil dans le Blackboard

Technologies : Embeddings, Clustering (K-Means), LLM
"""

import ollama
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from memory.blackboard import Blackboard
from utils.embeddings import get_embedding_generator


class ProfilingAgent:
    """
    Agent responsable du profilage des utilisateurs avec clustering
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b", n_clusters: int = 4):
        """
        Initialise le Profiling Agent
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Nom du modèle LLM local (Ollama)
            n_clusters: Nombre de clusters pour le style d'apprentissage
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        self.embedding_gen = get_embedding_generator()
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        # Mapping des clusters vers les styles d'apprentissage
        self.cluster_to_style = {
            0: "visual",
            1: "reading",
            2: "kinesthetic",
            3: "auditory"
        }
        
        print(f"✓ Profiling Agent initialisé avec modèle {llm_model}")
        print(f"✓ Clustering activé avec {n_clusters} clusters")
    
    def analyze_user(self, user_id: str) -> Dict:
        """
        Analyse complète d'un utilisateur et création de son profil
        
        Args:
            user_id: Identifiant de l'utilisateur
        
        Returns:
            Profil utilisateur complet (Dict)
        """
        print(f"\n{'='*60}")
        print(f"🔍 ANALYSE DU PROFIL UTILISATEUR: {user_id}")
        print(f"{'='*60}")
        
        # Étape 1 : Récupérer l'historique
        history = self.blackboard.get_user_history(user_id)
        
        if not history:
            print(f"⚠️  Aucun historique trouvé pour {user_id}")
            return self._create_default_profile(user_id)
        
        print(f"✓ Historique récupéré: {len(history)} interactions")
        
        # Étape 2 : Créer l'embedding du comportement utilisateur
        behavior_embedding = self._create_behavior_embedding(history)
        print(f"✓ Embedding comportemental créé (dim={len(behavior_embedding)})")
        
        # Étape 3 : Analyser le style d'apprentissage via clustering
        learning_style, cluster_id, confidence = self._infer_learning_style_with_clustering(
            behavior_embedding, history
        )
        print(f"✓ Style d'apprentissage détecté: {learning_style} (cluster {cluster_id}, conf: {confidence:.2f})")
        
        # Étape 4 : Évaluer le niveau
        level = self._infer_level(history)
        print(f"✓ Niveau estimé: {level}")
        
        # Étape 5 : Identifier les intérêts via embeddings sémantiques
        interests = self._extract_interests_with_embeddings(history)
        print(f"✓ Intérêts identifiés: {interests}")
        
        # Étape 6 : Détecter forces et faiblesses
        strengths, weaknesses = self._analyze_performance(history)
        print(f"✓ Forces: {strengths}")
        print(f"✓ Faiblesses: {weaknesses}")
        
        # Étape 7 : Trouver des utilisateurs similaires
        similar_users = self._find_similar_users(user_id, behavior_embedding)
        print(f"✓ Utilisateurs similaires: {similar_users}")
        
        # Étape 8 : Générer un résumé avec le LLM
        summary = self._generate_summary(
            user_id, learning_style, level, interests, confidence
        )
        
        # Créer le profil complet
        profile = {
            "user_id": user_id,
            "learning_style": learning_style,
            "learning_style_confidence": confidence,
            "cluster_id": cluster_id,
            "level": level,
            "interests": interests,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "behavior_embedding": behavior_embedding.tolist(),  # Pour réutilisation
            "similar_users": similar_users,
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_interactions": len(history)
        }
        
        # Sauvegarder dans le Blackboard
        self.blackboard.write("profiles", user_id, profile)
        print(f"\n✅ Profil sauvegardé dans le Blackboard")
        
        return profile
    
    def _create_behavior_embedding(self, history: List[Dict]) -> np.ndarray:
        """
        Crée un vecteur d'embedding représentant le comportement de l'utilisateur
        
        Features extraites :
        - Distribution des types de ressources (video, text, quiz, exercise)
        - Temps moyen par ressource
        - Scores moyens
        - Fréquence d'interaction
        - Diversité des sujets
        
        Args:
            history: Liste des interactions
        
        Returns:
            Vecteur numpy (embedding comportemental)
        """
        # Initialiser les compteurs
        resource_type_counts = {"video": 0, "text": 0, "quiz": 0, "exercise": 0, "audio": 0}
        total_time = 0
        scores = []
        subjects = set()
        
        for interaction in history:
            resource_id = interaction.get("resource_id", "").lower()
            
            # Compter les types de ressources
            if "video" in resource_id or "visual" in resource_id:
                resource_type_counts["video"] += 1
            elif "quiz" in resource_id or "test" in resource_id:
                resource_type_counts["quiz"] += 1
            elif "exercise" in resource_id or "lab" in resource_id:
                resource_type_counts["exercise"] += 1
            elif "audio" in resource_id or "podcast" in resource_id:
                resource_type_counts["audio"] += 1
            else:
                resource_type_counts["text"] += 1
            
            # Temps passé
            total_time += interaction.get("time_spent", 0)
            
            # Scores
            if interaction.get("type") == "quiz":
                scores.append(interaction.get("score", 0))
            
            # Sujets
            parts = resource_id.split("_")
            if len(parts) > 1:
                subjects.add(parts[1])
        
        # Calculer les features
        total_interactions = len(history)
        
        features = [
            # Proportions des types de ressources (normalisées)
            resource_type_counts["video"] / total_interactions,
            resource_type_counts["text"] / total_interactions,
            resource_type_counts["quiz"] / total_interactions,
            resource_type_counts["exercise"] / total_interactions,
            resource_type_counts["audio"] / total_interactions,
            
            # Temps moyen par interaction (normalisé)
            (total_time / total_interactions) / 60 if total_interactions > 0 else 0,
            
            # Performance moyenne
            np.mean(scores) / 100 if scores else 0,
            
            # Diversité des sujets (normalisée)
            len(subjects) / 10,  # Supposons max 10 sujets
            
            # Nombre d'interactions (log scale pour éviter les valeurs extrêmes)
            np.log1p(total_interactions) / 5
        ]
        
        return np.array(features)
    
    def _infer_learning_style_with_clustering(
        self, 
        behavior_embedding: np.ndarray, 
        history: List[Dict]
    ) -> Tuple[str, int, float]:
        """
        Infère le style d'apprentissage via clustering K-Means
        
        Args:
            behavior_embedding: Vecteur d'embedding comportemental
            history: Historique pour validation
        
        Returns:
            Tuple (style, cluster_id, confidence)
        """
        # Récupérer tous les profils existants pour le clustering
        all_profiles = self.blackboard.read_section("profiles")
        
        if not all_profiles or len(all_profiles) < self.n_clusters:
            # Pas assez de données pour clustering, utiliser heuristique
            return self._infer_learning_style_heuristic(history), -1, 0.5
        
        # Extraire les embeddings de tous les utilisateurs
        embeddings = []
        user_ids = []
        
        for uid, profile in all_profiles.items():
            if "behavior_embedding" in profile:
                embeddings.append(profile["behavior_embedding"])
                user_ids.append(uid)
        
        # Ajouter l'embedding actuel
        embeddings.append(behavior_embedding.tolist())
        embeddings_array = np.array(embeddings)
        
        # Normaliser
        embeddings_scaled = self.scaler.fit_transform(embeddings_array)
        
        # Clustering K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Cluster de l'utilisateur actuel (dernier élément)
        user_cluster = cluster_labels[-1]
        
        # Calculer la confidence (distance au centroïde du cluster)
        distances = kmeans.transform(embeddings_scaled)
        user_distance = distances[-1, user_cluster]
        
        # Confidence inversement proportionnelle à la distance
        confidence = 1 / (1 + user_distance)
        
        # Mapper le cluster au style d'apprentissage
        learning_style = self.cluster_to_style.get(user_cluster, "visual")
        
        return learning_style, int(user_cluster), float(confidence)
    
    def _infer_learning_style_heuristic(self, history: List[Dict]) -> str:
        """
        Version heuristique (fallback si pas assez de données pour clustering)
        """
        resource_types = []
        
        for interaction in history:
            resource_id = interaction.get("resource_id", "").lower()
            
            if any(keyword in resource_id for keyword in ["video", "visual", "diagram"]):
                resource_types.append("visual")
            elif any(keyword in resource_id for keyword in ["exercise", "lab", "project", "quiz"]):
                resource_types.append("kinesthetic")
            elif any(keyword in resource_id for keyword in ["text", "article", "doc", "course"]):
                resource_types.append("reading")
            elif any(keyword in resource_id for keyword in ["audio", "podcast"]):
                resource_types.append("auditory")
            else:
                resource_types.append("reading")
        
        if not resource_types:
            return "visual"
        
        style_counts = Counter(resource_types)
        return style_counts.most_common(1)[0][0]
    
    def _extract_interests_with_embeddings(self, history: List[Dict]) -> List[str]:
        """
        Extrait les intérêts via analyse sémantique des embeddings
        
        Args:
            history: Liste des interactions
        
        Returns:
            Liste des sujets d'intérêt
        """
        # Extraire tous les resource_ids et créer leurs embeddings
        resources = [i.get("resource_id", "") for i in history if i.get("resource_id")]
        
        if not resources:
            return ["programming"]
        
        # Méthode simple : extraction de mots-clés
        keywords = []
        for resource in resources:
            parts = resource.split("_")
            if len(parts) > 1:
                keywords.append(parts[1])
        
        # Compter les occurrences
        keyword_counts = Counter(keywords)
        top_interests = [k for k, v in keyword_counts.most_common(3)]
        
        return top_interests if top_interests else ["programming"]
    
    def _find_similar_users(self, user_id: str, embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """
        Trouve les utilisateurs les plus similaires via similarité cosine
        
        Args:
            user_id: ID de l'utilisateur actuel
            embedding: Embedding comportemental
            top_k: Nombre d'utilisateurs similaires à retourner
        
        Returns:
            Liste des IDs des utilisateurs similaires
        """
        all_profiles = self.blackboard.read_section("profiles")
        
        if not all_profiles or len(all_profiles) < 2:
            return []
        
        similarities = []
        
        for uid, profile in all_profiles.items():
            if uid == user_id:
                continue
            
            if "behavior_embedding" in profile:
                other_embedding = np.array(profile["behavior_embedding"])
                
                # Similarité cosine
                similarity = np.dot(embedding, other_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_embedding) + 1e-10
                )
                
                similarities.append((uid, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [uid for uid, sim in similarities[:top_k]]
    
    def _infer_level(self, history: List[Dict]) -> str:
        """Estime le niveau de compétence"""
        quiz_scores = [i.get("score", 0) for i in history if i.get("type") == "quiz"]
        
        if not quiz_scores:
            if len(history) < 5:
                return "beginner"
            elif len(history) < 15:
                return "intermediate"
            else:
                return "advanced"
        
        avg_score = np.mean(quiz_scores)
        
        # Ajustement des seuils pour mieux distinguer intermédiaire / avancé
        if avg_score < 60:
            return "beginner"
        elif avg_score < 85:
            return "intermediate"
        else:
            return "advanced"
    
    def _analyze_performance(self, history: List[Dict]) -> Tuple[List[str], List[str]]:
        """Analyse les forces et faiblesses"""
        performance_by_topic = {}
        
        for interaction in history:
            if interaction.get("type") == "quiz":
                resource = interaction.get("resource_id", "")
                score = interaction.get("score", 0)
                
                topic = resource.split("_")[1] if "_" in resource else "general"
                
                if topic not in performance_by_topic:
                    performance_by_topic[topic] = []
                performance_by_topic[topic].append(score)
        
        strengths = []
        weaknesses = []
        
        for topic, scores in performance_by_topic.items():
            avg_score = np.mean(scores)
            if avg_score >= 80:
                strengths.append(topic)
            elif avg_score < 60:
                weaknesses.append(topic)
        
        return strengths, weaknesses
    
    def _generate_summary(
        self, 
        user_id: str, 
        style: str, 
        level: str, 
        interests: List[str],
        confidence: float
    ) -> str:
        """Génère un résumé avec le LLM"""
        prompt = f"""You are an educational profiling expert. Create a brief learner profile summary (2-3 sentences) based on:

User ID: {user_id}
Learning Style: {style} (confidence: {confidence:.0%})
Level: {level}
Interests: {', '.join(interests)}

Write a professional, concise summary highlighting the learner's preferences and potential."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return f"Learner with {style} style at {level} level, interested in {', '.join(interests)}."
    
    def _create_default_profile(self, user_id: str) -> Dict:
        """Crée un profil par défaut"""
        profile = {
            "user_id": user_id,
            "learning_style": "visual",
            "learning_style_confidence": 0.5,
            "cluster_id": -1,
            "level": "beginner",
            "interests": ["general"],
            "strengths": [],
            "weaknesses": [],
            "behavior_embedding": np.zeros(9).tolist(),
            "similar_users": [],
            "summary": "New learner with no interaction history yet.",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_interactions": 0
        }
        
        self.blackboard.write("profiles", user_id, profile)
        return profile
    
    def update_profile(self, user_id: str) -> Dict:
        """Met à jour un profil existant"""
        print(f"🔄 Mise à jour du profil de {user_id}...")
        return self.analyze_user(user_id)