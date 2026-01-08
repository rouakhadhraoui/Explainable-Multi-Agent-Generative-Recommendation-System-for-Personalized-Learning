# evaluation/recommendation_metrics.py
"""
Métriques d'évaluation pour les systèmes de recommandation

Implémente les métriques standard :
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Recall@K
- Precision@K
- MAP (Mean Average Precision)

Ces métriques sont utilisées pour évaluer la qualité des recommandations
par rapport à un ground truth (vérité terrain).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class RecommendationMetrics:
    """
    Calcule les métriques d'évaluation pour les recommandations
    """
    
    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcule le NDCG@K (Normalized Discounted Cumulative Gain)
        
        NDCG mesure la qualité du ranking en tenant compte de la position.
        Les éléments pertinents en haut de la liste ont plus de poids.
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
            k: Nombre de recommandations à considérer
        
        Returns:
            Score NDCG entre 0 et 1 (1 = parfait)
        """
        # Limiter aux k premières recommandations
        recommended = recommended[:k]
        
        if not recommended or not relevant:
            return 0.0
        
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(recommended):
            if item in relevant:
                # Position i+1 (1-indexed), discount = log2(i+2)
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG (Ideal DCG) - si tous les items pertinents étaient en tête
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mrr(recommended: List[str], relevant: List[str]) -> float:
        """
        Calcule le MRR (Mean Reciprocal Rank)
        
        MRR mesure la position du premier item pertinent.
        Plus il apparaît tôt, meilleur est le score.
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
        
        Returns:
            Score MRR entre 0 et 1 (1 = premier item est pertinent)
        """
        if not recommended or not relevant:
            return 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                # Reciprocal du rang (1-indexed)
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def recall_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcule le Recall@K
        
        Recall@K = nombre d'items pertinents dans top-K / nombre total d'items pertinents
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
            k: Nombre de recommandations à considérer
        
        Returns:
            Score Recall entre 0 et 1
        """
        if not relevant:
            return 0.0
        
        # Limiter aux k premières recommandations
        recommended = recommended[:k]
        
        # Compter les items pertinents dans les recommandations
        hits = len(set(recommended) & set(relevant))
        
        return hits / len(relevant)
    
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcule la Precision@K
        
        Precision@K = nombre d'items pertinents dans top-K / K
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
            k: Nombre de recommandations à considérer
        
        Returns:
            Score Precision entre 0 et 1
        """
        if not recommended:
            return 0.0
        
        # Limiter aux k premières recommandations
        recommended = recommended[:k]
        
        # Compter les items pertinents dans les recommandations
        hits = len(set(recommended) & set(relevant))
        
        return hits / k
    
    @staticmethod
    def average_precision(recommended: List[str], relevant: List[str]) -> float:
        """
        Calcule l'Average Precision (AP)
        
        AP prend en compte la précision à chaque position où un item pertinent apparaît.
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
        
        Returns:
            Score AP entre 0 et 1
        """
        if not relevant:
            return 0.0
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        if hits == 0:
            return 0.0
        
        return sum_precisions / len(relevant)
    
    @staticmethod
    def hit_rate_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcule le Hit Rate@K
        
        Hit Rate@K = 1 si au moins un item pertinent est dans top-K, sinon 0
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
            k: Nombre de recommandations à considérer
        
        Returns:
            1.0 ou 0.0
        """
        recommended = recommended[:k]
        
        # Vérifier s'il y a au moins un hit
        if set(recommended) & set(relevant):
            return 1.0
        
        return 0.0
    
    @classmethod
    def evaluate_all(cls, recommended: List[str], relevant: List[str], 
                    k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Évalue toutes les métriques pour différentes valeurs de K
        
        Args:
            recommended: Liste des items recommandés (dans l'ordre)
            relevant: Liste des items pertinents (ground truth)
            k_values: Liste des valeurs de K à évaluer
        
        Returns:
            Dictionnaire contenant toutes les métriques
        """
        results = {}
        
        # MRR et AP (pas de K)
        results['MRR'] = cls.mrr(recommended, relevant)
        results['MAP'] = cls.average_precision(recommended, relevant)
        
        # Métriques pour chaque K
        for k in k_values:
            results[f'NDCG@{k}'] = cls.ndcg_at_k(recommended, relevant, k)
            results[f'Recall@{k}'] = cls.recall_at_k(recommended, relevant, k)
            results[f'Precision@{k}'] = cls.precision_at_k(recommended, relevant, k)
            results[f'HitRate@{k}'] = cls.hit_rate_at_k(recommended, relevant, k)
        
        return results
    
    @classmethod
    def evaluate_batch(cls, batch_recommendations: List[Tuple[List[str], List[str]]],
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Évalue un batch de recommandations et calcule les moyennes
        
        Args:
            batch_recommendations: Liste de tuples (recommended, relevant)
            k_values: Liste des valeurs de K à évaluer
        
        Returns:
            Dictionnaire des métriques moyennes
        """
        all_results = []
        
        for recommended, relevant in batch_recommendations:
            results = cls.evaluate_all(recommended, relevant, k_values)
            all_results.append(results)
        
        # Calculer les moyennes
        avg_results = {}
        
        if all_results:
            for metric in all_results[0].keys():
                values = [r[metric] for r in all_results]
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results


class GroundTruthGenerator:
    """
    Génère un ground truth pour l'évaluation des recommandations
    
    Dans un contexte e-learning, le ground truth peut être :
    - Les ressources que l'étudiant a effectivement consultées après
    - Les ressources sur lesquelles l'étudiant a eu de bons scores
    - Les ressources recommandées par des experts
    """
    
    @staticmethod
    def from_future_interactions(past_interactions: List[Dict],
                                 future_interactions: List[Dict],
                                 min_score: float = 70.0) -> List[str]:
        """
        Génère un ground truth basé sur les interactions futures
        
        Les items pertinents sont ceux que l'utilisateur a :
        - Consultés dans le futur
        - Réussi avec un bon score (pour les quiz)
        
        Args:
            past_interactions: Interactions passées (pour l'entraînement)
            future_interactions: Interactions futures (ground truth)
            min_score: Score minimum pour considérer un quiz comme réussi
        
        Returns:
            Liste des resource_ids pertinents
        """
        relevant = []
        
        for interaction in future_interactions:
            resource_id = interaction.get('resource_id')
            
            if not resource_id:
                continue
            
            # Si c'est un quiz, vérifier le score
            if interaction.get('type') == 'quiz':
                score = interaction.get('score', 0)
                if score >= min_score:
                    relevant.append(resource_id)
            else:
                # Pour les autres types, considérer comme pertinent si consulté
                relevant.append(resource_id)
        
        return list(set(relevant))  # Éliminer les doublons
    
    @staticmethod
    def from_successful_peers(user_profile: Dict,
                             all_users_data: List[Dict]) -> List[str]:
        """
        Génère un ground truth basé sur les utilisateurs similaires qui ont réussi
        
        Args:
            user_profile: Profil de l'utilisateur cible
            all_users_data: Données de tous les utilisateurs
        
        Returns:
            Liste des resource_ids pertinents
        """
        # Trouver les utilisateurs similaires
        similar_users = []
        
        for other_user in all_users_data:
            if (other_user.get('level') == user_profile.get('level') and
                other_user.get('learning_style') == user_profile.get('learning_style')):
                similar_users.append(other_user)
        
        # Récupérer les ressources sur lesquelles ils ont réussi
        relevant = []
        
        for user in similar_users:
            for interaction in user.get('interactions', []):
                if interaction.get('type') == 'quiz' and interaction.get('score', 0) >= 80:
                    relevant.append(interaction.get('resource_id'))
        
        return list(set(relevant))


def print_metrics_report(metrics: Dict[str, float], title: str = "Métriques d'Évaluation"):
    """
    Affiche un rapport formaté des métriques
    
    Args:
        metrics: Dictionnaire des métriques
        title: Titre du rapport
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Grouper par type de métrique
    mrr_map = {k: v for k, v in metrics.items() if k in ['MRR', 'MAP']}
    ndcg = {k: v for k, v in metrics.items() if 'NDCG' in k and '_std' not in k}
    recall = {k: v for k, v in metrics.items() if 'Recall' in k and '_std' not in k}
    precision = {k: v for k, v in metrics.items() if 'Precision' in k and '_std' not in k}
    hitrate = {k: v for k, v in metrics.items() if 'HitRate' in k and '_std' not in k}
    
    # Afficher par catégorie
    if mrr_map:
        print(f"\n📊 Métriques Globales:")
        for metric, value in mrr_map.items():
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:20s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:20s} : {value:.4f}")
    
    if ndcg:
        print(f"\n🎯 NDCG (Normalized Discounted Cumulative Gain):")
        for metric, value in sorted(ndcg.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:20s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:20s} : {value:.4f}")
    
    if recall:
        print(f"\n📈 Recall (Couverture):")
        for metric, value in sorted(recall.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:20s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:20s} : {value:.4f}")
    
    if precision:
        print(f"\n🎯 Precision (Précision):")
        for metric, value in sorted(precision.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:20s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:20s} : {value:.4f}")
    
    if hitrate:
        print(f"\n✓ Hit Rate (Taux de succès):")
        for metric, value in sorted(hitrate.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:20s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:20s} : {value:.4f}")
    
    print(f"\n{'='*70}\n")