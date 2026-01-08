# evaluation/xai_metrics.py
"""
Métriques d'évaluation pour l'explicabilité (XAI)

Implémente les métriques pour évaluer la qualité des explications :
- Faithfulness (Fidélité) : L'explication reflète-t-elle vraiment le raisonnement ?
- Plausibility (Plausibilité) : L'explication est-elle logique pour un humain ?
- Completeness (Complétude) : L'explication couvre-t-elle tous les aspects ?
- Trust Score (Confiance) : L'utilisateur fait-il confiance à l'explication ?

Ces métriques sont essentielles pour évaluer les systèmes XAI en e-learning.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re


class XAIMetrics:
    """
    Calcule les métriques d'évaluation pour les explications (XAI)
    """
    
    @staticmethod
    def faithfulness_score(explanation: Dict, actual_features: Dict,
                          feature_importance: Dict) -> float:
        """
        Calcule la fidélité (faithfulness) d'une explication
        
        La fidélité mesure si l'explication reflète vraiment les features
        importantes qui ont conduit à la décision.
        
        Args:
            explanation: Dict contenant l'explication structurée
            actual_features: Features réelles de l'utilisateur
            feature_importance: Importance réelle de chaque feature
        
        Returns:
            Score de fidélité entre 0 et 1
        """
        if not explanation or not feature_importance:
            return 0.0
        
        # Extraire les features mentionnées dans l'explication
        explanation_text = str(explanation).lower()
        
        # Vérifier quelles features importantes sont mentionnées
        mentioned_important = 0
        total_important = 0
        
        for feature, importance in feature_importance.items():
            # Considérer seulement les features importantes (importance > 0.1)
            if importance > 0.1:
                total_important += 1
                
                # Vérifier si la feature est mentionnée dans l'explication
                feature_lower = feature.lower().replace('_', ' ')
                if feature_lower in explanation_text:
                    mentioned_important += 1
        
        # Score = ratio des features importantes mentionnées
        if total_important == 0:
            return 1.0  # Si aucune feature importante, score max
        
        return mentioned_important / total_important
    
    @staticmethod
    def plausibility_score(explanation: Dict) -> float:
        """
        Calcule la plausibilité d'une explication
        
        La plausibilité mesure si l'explication est logique et compréhensible
        pour un humain, indépendamment du modèle.
        
        Args:
            explanation: Dict contenant l'explication structurée
        
        Returns:
            Score de plausibilité entre 0 et 1
        """
        score = 0.0
        max_score = 4.0
        
        # Critère 1 : L'explication contient des justifications (0.25)
        explanation_text = str(explanation).lower()
        reasoning_keywords = ['because', 'since', 'due to', 'as', 'therefore', 
                             'thus', 'hence', 'consequently']
        
        if any(keyword in explanation_text for keyword in reasoning_keywords):
            score += 1.0
        
        # Critère 2 : L'explication est structurée (0.25)
        if isinstance(explanation, dict) and len(explanation) > 2:
            score += 1.0
        
        # Critère 3 : L'explication contient des détails concrets (0.25)
        # Vérifier la présence de nombres, scores, ou niveaux
        if re.search(r'\d+|beginner|intermediate|advanced|score', explanation_text):
            score += 1.0
        
        # Critère 4 : L'explication n'est ni trop courte ni trop longue (0.25)
        text_length = len(explanation_text)
        if 100 < text_length < 1000:  # Longueur raisonnable
            score += 1.0
        
        return score / max_score
    
    @staticmethod
    def completeness_score(explanation: Dict, required_components: List[str]) -> float:
        """
        Calcule la complétude d'une explication
        
        La complétude mesure si l'explication couvre tous les aspects requis.
        
        Args:
            explanation: Dict contenant l'explication structurée
            required_components: Liste des composants requis
                                (ex: ['level_reasoning', 'style_reasoning'])
        
        Returns:
            Score de complétude entre 0 et 1
        """
        if not required_components:
            return 1.0
        
        components_present = 0
        
        for component in required_components:
            # Vérifier si le composant existe et n'est pas vide
            if component in explanation:
                value = explanation[component]
                # Vérifier que la valeur n'est pas vide ou "N/A"
                if value and str(value).lower() not in ['n/a', 'none', '']:
                    components_present += 1
        
        return components_present / len(required_components)
    
    @staticmethod
    def trust_score_heuristic(explanation: Dict, confidence_indicators: Dict) -> float:
        """
        Calcule un score de confiance heuristique
        
        Ce score estime la confiance qu'un utilisateur pourrait avoir
        basé sur des indicateurs de qualité.
        
        Args:
            explanation: Dict contenant l'explication
            confidence_indicators: Dict avec des indicateurs
                                  (ex: {'data_quality': 0.9, 'model_confidence': 0.85})
        
        Returns:
            Score de confiance entre 0 et 1
        """
        # Score de base selon la qualité de l'explication
        base_score = 0.0
        
        # Facteur 1 : Présence de données concrètes
        explanation_text = str(explanation).lower()
        has_numbers = bool(re.search(r'\d+', explanation_text))
        has_specifics = any(word in explanation_text for word in 
                          ['score', 'level', 'interactions', 'based on'])
        
        if has_numbers and has_specifics:
            base_score += 0.4
        elif has_numbers or has_specifics:
            base_score += 0.2
        
        # Facteur 2 : Structure claire
        if isinstance(explanation, dict) and len(explanation) >= 3:
            base_score += 0.3
        
        # Facteur 3 : Indicateurs de confiance externes
        if confidence_indicators:
            avg_confidence = np.mean(list(confidence_indicators.values()))
            base_score += 0.3 * avg_confidence
        
        return min(base_score, 1.0)
    
    @staticmethod
    def consistency_score(explanations: List[Dict]) -> float:
        """
        Calcule la cohérence entre plusieurs explications
        
        Vérifie si les explications pour des cas similaires sont cohérentes.
        
        Args:
            explanations: Liste d'explications pour des cas similaires
        
        Returns:
            Score de cohérence entre 0 et 1
        """
        if len(explanations) < 2:
            return 1.0
        
        # Extraire les textes
        texts = [str(exp).lower() for exp in explanations]
        
        # Calculer la similarité moyenne entre paires
        similarities = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Similarité simple basée sur les mots communs
                words_i = set(texts[i].split())
                words_j = set(texts[j].split())
                
                if words_i and words_j:
                    similarity = len(words_i & words_j) / len(words_i | words_j)
                    similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        return np.mean(similarities)
    
    @staticmethod
    def contrastive_quality_score(counterfactuals: Dict) -> float:
        """
        Évalue la qualité des explications contrefactuelles
        
        Vérifie si les explications "et si..." sont pertinentes.
        
        Args:
            counterfactuals: Dict contenant les explications contrefactuelles
        
        Returns:
            Score de qualité entre 0 et 1
        """
        score = 0.0
        max_score = 3.0
        
        # Critère 1 : Présence de plusieurs scénarios
        if isinstance(counterfactuals, dict) and len(counterfactuals) >= 2:
            score += 1.0
        
        # Critère 2 : Les scénarios contiennent des conditions
        text = str(counterfactuals).lower()
        if_keywords = ['if', 'would', 'could', 'should', 'change', 'different']
        
        if sum(1 for keyword in if_keywords if keyword in text) >= 3:
            score += 1.0
        
        # Critère 3 : Les scénarios sont spécifiques
        has_specifics = any(word in text for word in 
                          ['level', 'score', 'style', 'interactions', 'higher', 'lower'])
        
        if has_specifics:
            score += 1.0
        
        return score / max_score
    
    @classmethod
    def evaluate_all(cls, explanation: Dict, actual_features: Dict = None,
                    feature_importance: Dict = None,
                    confidence_indicators: Dict = None) -> Dict[str, float]:
        """
        Évalue toutes les métriques XAI
        
        Args:
            explanation: Dict contenant l'explication complète
            actual_features: Features réelles (optionnel)
            feature_importance: Importance des features (optionnel)
            confidence_indicators: Indicateurs de confiance (optionnel)
        
        Returns:
            Dict contenant toutes les métriques XAI
        """
        results = {}
        
        # Faithfulness
        if actual_features and feature_importance:
            results['faithfulness'] = cls.faithfulness_score(
                explanation, actual_features, feature_importance
            )
        
        # Plausibility
        results['plausibility'] = cls.plausibility_score(explanation)
        
        # Completeness (profil)
        profile_components = ['level_reasoning', 'style_reasoning', 'interests_reasoning']
        if 'profile_explanation' in explanation:
            results['profile_completeness'] = cls.completeness_score(
                explanation['profile_explanation'], profile_components
            )
        
        # Completeness (parcours)
        path_components = ['path_logic', 'personalization', 'expected_outcomes']
        if 'path_explanation' in explanation:
            results['path_completeness'] = cls.completeness_score(
                explanation['path_explanation'], path_components
            )
        
        # Completeness (recommandations)
        rec_components = ['selection_criteria', 'ranking_logic', 'personalization_factors']
        if 'recommendations_explanation' in explanation:
            results['recommendations_completeness'] = cls.completeness_score(
                explanation['recommendations_explanation'], rec_components
            )
        
        # Trust score
        results['trust_score'] = cls.trust_score_heuristic(
            explanation, confidence_indicators or {}
        )
        
        # Contrastive quality
        if 'counterfactuals' in explanation:
            results['contrastive_quality'] = cls.contrastive_quality_score(
                explanation['counterfactuals']
            )
        
        return results
    
    @classmethod
    def evaluate_batch(cls, batch_explanations: List[Dict],
                      batch_features: List[Dict] = None,
                      batch_importance: List[Dict] = None) -> Dict[str, float]:
        """
        Évalue un batch d'explications
        
        Args:
            batch_explanations: Liste d'explications
            batch_features: Liste de features (optionnel)
            batch_importance: Liste d'importances (optionnel)
        
        Returns:
            Dict des métriques moyennes
        """
        all_results = []
        
        for i, explanation in enumerate(batch_explanations):
            features = batch_features[i] if batch_features else None
            importance = batch_importance[i] if batch_importance else None
            
            results = cls.evaluate_all(explanation, features, importance)
            all_results.append(results)
        
        # Calculer les moyennes
        avg_results = {}
        
        if all_results:
            for metric in all_results[0].keys():
                values = [r.get(metric, 0) for r in all_results]
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
        
        # Ajouter la consistance
        avg_results['consistency'] = cls.consistency_score(batch_explanations)
        
        return avg_results


def print_xai_metrics_report(metrics: Dict[str, float],
                            title: str = "Métriques XAI"):
    """
    Affiche un rapport formaté des métriques XAI
    
    Args:
        metrics: Dictionnaire des métriques
        title: Titre du rapport
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Fidélité
    if 'faithfulness' in metrics:
        print(f"\n🔍 Faithfulness (Fidélité):")
        std_key = 'faithfulness_std'
        if std_key in metrics:
            print(f"  Faithfulness           : {metrics['faithfulness']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  Faithfulness           : {metrics['faithfulness']:.4f}")
    
    # Plausibilité
    if 'plausibility' in metrics:
        print(f"\n💡 Plausibility (Plausibilité):")
        std_key = 'plausibility_std'
        if std_key in metrics:
            print(f"  Plausibility           : {metrics['plausibility']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  Plausibility           : {metrics['plausibility']:.4f}")
    
    # Complétude
    completeness_metrics = {k: v for k, v in metrics.items() 
                           if 'completeness' in k and '_std' not in k}
    if completeness_metrics:
        print(f"\n✓ Completeness (Complétude):")
        for metric, value in sorted(completeness_metrics.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:30s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:30s} : {value:.4f}")
    
    # Confiance
    if 'trust_score' in metrics:
        print(f"\n🤝 Trust Score (Confiance):")
        std_key = 'trust_score_std'
        if std_key in metrics:
            print(f"  Trust Score            : {metrics['trust_score']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  Trust Score            : {metrics['trust_score']:.4f}")
    
    # Qualité contrastive
    if 'contrastive_quality' in metrics:
        print(f"\n🔄 Contrastive Quality (Contrefactuels):")
        std_key = 'contrastive_quality_std'
        if std_key in metrics:
            print(f"  Contrastive Quality    : {metrics['contrastive_quality']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  Contrastive Quality    : {metrics['contrastive_quality']:.4f}")
    
    # Cohérence
    if 'consistency' in metrics:
        print(f"\n🔗 Consistency (Cohérence):")
        print(f"  Consistency            : {metrics['consistency']:.4f}")
    
    print(f"\n{'='*70}\n")