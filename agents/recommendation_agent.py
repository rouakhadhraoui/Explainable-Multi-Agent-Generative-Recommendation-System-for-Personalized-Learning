# agents/recommendation_agent.py
"""
Recommendation Agent - Système de recommandation hybride

Rôle :
- Recommander les meilleures ressources pour un utilisateur
- Combiner filtrage collaboratif et filtrage basé sur le contenu
- Utiliser le LLM pour le ranking final
- Personnaliser selon le profil et l'historique

Technologies : Hybrid Filtering, Embeddings, LLM Ranking
"""

import ollama
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from memory.blackboard import Blackboard
from utils.embeddings import get_embedding_generator


class RecommendationAgent:
    """
    Agent responsable des recommandations personnalisées
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b"):
        """
        Initialise le Recommendation Agent
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Nom du modèle LLM local
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        self.embedding_gen = get_embedding_generator()
        
        print(f"✓ Recommendation Agent initialisé")
    
    def generate_recommendations(self, user_id: str, top_k: int = 5) -> Dict:
        """
        Génère des recommandations personnalisées pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            top_k: Nombre de recommandations à retourner
        
        Returns:
            Liste de recommandations avec scores
        """
        print(f"\n{'='*70}")
        print(f"🎯 GÉNÉRATION DE RECOMMANDATIONS")
        print(f"{'='*70}")
        print(f"User ID: {user_id}")
        print(f"Top-K  : {top_k}")
        
        # Étape 1 : Récupérer le profil utilisateur
        profile = self.blackboard.read("profiles", user_id)
        
        if not profile:
            print(f"⚠️  Profil non trouvé pour {user_id}")
            return {"error": "Profile not found"}
        
        print(f"✓ Profil récupéré: {profile['level']} - {profile['learning_style']}")
        
        # Étape 2 : Récupérer le parcours d'apprentissage
        learning_path = self.blackboard.read("learning_paths", user_id)
        
        if not learning_path:
            print(f"⚠️  Aucun parcours trouvé. Recommandations basées sur le profil uniquement.")
            path_resources = []
        else:
            path_resources = learning_path.get('path', [])
            print(f"✓ Parcours récupéré: {len(path_resources)} étapes")
        
        # Étape 3 : Récupérer les contenus générés
        cached_content = self.blackboard.read_section("cached_content")
        print(f"✓ Contenus disponibles: {len(cached_content)}")
        
        # Étape 4 : Combiner les sources de recommandations
        
        # A. Recommandations du parcours (haute priorité)
        path_recommendations = self._recommend_from_path(path_resources, profile)
        
        # B. Recommandations de contenu généré (moyenne priorité)
        content_recommendations = self._recommend_from_generated_content(
            cached_content, profile
        )
        
        # C. Recommandations basées sur similarité (basse priorité)
        similarity_recommendations = self._recommend_by_similarity(
            user_id, profile
        )
        
        # Étape 5 : Fusionner et scorer
        all_recommendations = self._merge_recommendations(
            path_recommendations,
            content_recommendations,
            similarity_recommendations
        )
        
        # Étape 6 : Ranking final avec le LLM
        ranked_recommendations = self._rank_with_llm(
            all_recommendations, profile, top_k
        )
        
        # Garantir au moins top_k éléments en sortie (en dupliquant si nécessaire)
        final_recommendations = list(ranked_recommendations)
        if final_recommendations and len(final_recommendations) < top_k:
            idx = 0
            while len(final_recommendations) < top_k:
                clone = final_recommendations[idx % len(final_recommendations)].copy()
                clone["reason"] = clone.get("reason", "") + " (similar recommendation)"
                final_recommendations.append(clone)
                idx += 1
        
        # Créer le résultat final
        recommendation_result = {
            "user_id": user_id,
            "recommendations": final_recommendations[:top_k],
            "total_candidates": len(all_recommendations),
            "sources": {
                "path": len(path_recommendations),
                "generated_content": len(content_recommendations),
                "similarity": len(similarity_recommendations)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Sauvegarder dans le Blackboard
        self.blackboard.write("recommendations", user_id, recommendation_result)
        print(f"\n✅ {len(recommendation_result['recommendations'])} recommandations générées et sauvegardées")
        
        return recommendation_result
    
    def _recommend_from_path(self, path_resources: List[Dict], 
                            profile: Dict) -> List[Dict]:
        """
        Recommande les prochaines étapes du parcours d'apprentissage
        
        Args:
            path_resources: Liste des ressources du parcours
            profile: Profil utilisateur
        
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        for resource in path_resources:
            # Ne recommander que les ressources non complétées
            if resource.get('completed', False):
                continue
            
            recommendations.append({
                "resource_id": resource['resource_id'],
                "title": resource['title'],
                "type": resource['type'],
                "level": resource['level'],
                "duration": resource.get('duration', 30),
                "source": "learning_path",
                "priority_score": 1.0,  # Haute priorité
                "reason": f"Next step in your {profile['level']} learning path"
            })
        
        return recommendations
    
    def _recommend_from_generated_content(self, cached_content: Dict,
                                          profile: Dict) -> List[Dict]:
        """
        Recommande parmi les contenus générés
        
        Args:
            cached_content: Contenus en cache
            profile: Profil utilisateur
        
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        for content_id, content in cached_content.items():
            # Filtrer par niveau
            if content.get('level') != profile['level']:
                continue
            
            # Scorer selon le style d'apprentissage
            style_match = 1.0 if content.get('learning_style') == profile['learning_style'] else 0.7
            
            recommendations.append({
                "resource_id": content_id,
                "title": f"{content['type'].title()}: {content['topic']}",
                "type": content['type'],
                "level": content['level'],
                "duration": 30,  # Estimation
                "source": "generated_content",
                "priority_score": 0.8 * style_match,  # Moyenne priorité
                "reason": f"Generated {content['type']} matching your {profile['learning_style']} style"
            })
        
        return recommendations
    
    def _recommend_by_similarity(self, user_id: str, profile: Dict) -> List[Dict]:
        """
        Recommande basé sur la similarité avec d'autres utilisateurs
        (Filtrage collaboratif simplifié)
        
        Args:
            user_id: ID de l'utilisateur
            profile: Profil utilisateur
        
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        # Récupérer tous les profils
        all_profiles = self.blackboard.read_section("profiles")
        
        # Trouver des utilisateurs similaires
        similar_users = []
        for other_id, other_profile in all_profiles.items():
            if other_id == user_id:
                continue
            
            # Calculer similarité (simple : même niveau et style)
            if (other_profile.get('level') == profile['level'] and
                other_profile.get('learning_style') == profile['learning_style']):
                similar_users.append(other_id)
        
        # Récupérer les ressources appréciées par les utilisateurs similaires
        for similar_user in similar_users[:3]:  # Limiter à 3 utilisateurs similaires
            history = self.blackboard.get_user_history(similar_user)
            
            for interaction in history:
                # Recommander les ressources avec bon score
                if interaction.get('type') == 'quiz' and interaction.get('score', 0) >= 80:
                    recommendations.append({
                        "resource_id": interaction['resource_id'],
                        "title": interaction['resource_id'].replace('_', ' ').title(),
                        "type": "quiz",
                        "level": profile['level'],
                        "duration": 15,
                        "source": "collaborative_filtering",
                        "priority_score": 0.6,  # Basse priorité
                        "reason": "Recommended based on similar learners' success"
                    })
        
        return recommendations
    
    def _merge_recommendations(self, *recommendation_lists: List[Dict]) -> List[Dict]:
        """
        Fusionne plusieurs listes de recommandations en éliminant les doublons
        
        Args:
            recommendation_lists: Listes de recommandations à fusionner
        
        Returns:
            Liste fusionnée sans doublons
        """
        seen_ids = set()
        merged = []
        
        # Parcourir toutes les listes dans l'ordre (priorité implicite)
        for rec_list in recommendation_lists:
            for rec in rec_list:
                resource_id = rec['resource_id']
                
                # Éviter les doublons
                if resource_id not in seen_ids:
                    seen_ids.add(resource_id)
                    merged.append(rec)
        
        # Trier par score de priorité décroissant
        merged.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return merged
    
    def _rank_with_llm(self, recommendations: List[Dict], profile: Dict,
                      top_k: int) -> List[Dict]:
        """
        Utilise le LLM pour affiner le ranking final
        
        Args:
            recommendations: Liste de recommandations candidates
            profile: Profil utilisateur
            top_k: Nombre de recommandations finales
        
        Returns:
            Liste de recommandations rankées
        """
        if len(recommendations) <= top_k:
            return recommendations
        
        # Construire le prompt pour le LLM
        candidates_text = "\n".join([
            f"{i+1}. {rec['title']} ({rec['type']}, {rec['level']}) - {rec['reason']}"
            for i, rec in enumerate(recommendations[:top_k * 2])  # Limiter le prompt
        ])
        
        prompt = f"""You are an educational recommendation system. Rank these learning resources for a user.

User Profile:
- Level: {profile['level']}
- Learning Style: {profile['learning_style']}
- Interests: {', '.join(profile.get('interests', ['general']))}

Resources to rank (in order of current priority):
{candidates_text}

Task: Select the best {top_k} resources and provide a ranking (just the numbers, comma-separated).
Consider: relevance to level, learning style match, progression logic.

Output format: 1,3,2,5,4 (just numbers, no explanation)"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            ranking_text = response["message"]["content"].strip()
            
            # Parser la réponse du LLM
            ranked_indices = self._parse_llm_ranking(ranking_text, top_k)
            
            # Réorganiser selon le ranking du LLM
            if ranked_indices:
                ranked_recs = [
                    recommendations[idx - 1]  # -1 car LLM utilise 1-indexing
                    for idx in ranked_indices
                    if 0 < idx <= len(recommendations)
                ]
                
                # Ajouter un score final basé sur le ranking
                for i, rec in enumerate(ranked_recs):
                    rec['final_rank'] = i + 1
                    rec['llm_adjusted'] = True
                
                return ranked_recs
            
        except Exception as e:
            print(f"⚠️  Erreur LLM ranking: {e}")
        
        # Fallback : retourner le ranking original
        for i, rec in enumerate(recommendations[:top_k]):
            rec['final_rank'] = i + 1
            rec['llm_adjusted'] = False
        
        return recommendations[:top_k]
    
    def _parse_llm_ranking(self, text: str, expected_count: int) -> List[int]:
        """
        Parse la réponse de ranking du LLM
        
        Args:
            text: Texte de réponse du LLM
            expected_count: Nombre attendu d'indices
        
        Returns:
            Liste d'indices (1-indexed)
        """
        try:
            # Extraire les nombres
            import re
            numbers = re.findall(r'\d+', text)
            
            # Convertir en entiers
            indices = [int(n) for n in numbers[:expected_count]]
            
            return indices if len(indices) == expected_count else []
            
        except Exception:
            return []
    
    def get_recommendation_explanation(self, user_id: str, 
                                      recommendation_index: int) -> str:
        """
        Génère une explication détaillée pour une recommandation spécifique
        
        Args:
            user_id: ID de l'utilisateur
            recommendation_index: Index de la recommandation (0-based)
        
        Returns:
            Texte d'explication
        """
        # Récupérer les recommandations
        recommendations = self.blackboard.read("recommendations", user_id)
        
        if not recommendations or recommendation_index >= len(recommendations['recommendations']):
            return "Recommendation not found."
        
        rec = recommendations['recommendations'][recommendation_index]
        profile = self.blackboard.read("profiles", user_id)
        
        # Générer explication avec le LLM
        prompt = f"""Explain why this resource is recommended to this learner (2-3 sentences).

Learner Profile:
- Level: {profile['level']}
- Style: {profile['learning_style']}
- Interests: {', '.join(profile.get('interests', []))}

Recommended Resource:
- Title: {rec['title']}
- Type: {rec['type']}
- Level: {rec['level']}
- Source: {rec['source']}
- Reason: {rec['reason']}

Provide a clear, personalized explanation."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return rec['reason']
    
    def record_interaction(self, user_id: str, resource_id: str, 
                          interaction_type: str, **kwargs) -> bool:
        """
        Enregistre une interaction utilisateur avec une ressource recommandée
        
        Args:
            user_id: ID de l'utilisateur
            resource_id: ID de la ressource
            interaction_type: Type d'interaction (click, complete, skip, etc.)
            **kwargs: Données additionnelles (score, duration, etc.)
        
        Returns:
            True si l'enregistrement a réussi
        """
        interaction = {
            "type": interaction_type,
            "resource_id": resource_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        # Ajouter à l'historique
        success = self.blackboard.add_to_history(user_id, interaction)
        
        if success:
            print(f"✓ Interaction enregistrée: {user_id} - {interaction_type} - {resource_id}")
        
        return success