# evaluation/system_evaluation.py
"""
Évaluation Complète du Système Multi-Agents

Ce module évalue le système complet sur le dataset OULAD avec toutes les métriques :
- Métriques de recommandation (NDCG, MRR, Recall@K)
- Métriques de génération (ROUGE, BERTScore)
- Métriques XAI (Faithfulness, Plausibility, Trust Score)

Les résultats sont sauvegardés pour publication scientifique.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from memory.blackboard import Blackboard
from orchestrator.orchestrator import Orchestrator
from utils.oulad_integration import OULADIntegration
from evaluation.recommendation_metrics import RecommendationMetrics, print_metrics_report
from evaluation.generation_metrics import GenerationMetrics, print_generation_metrics_report
from evaluation.xai_metrics import XAIMetrics, print_xai_metrics_report


class SystemEvaluation:
    """
    Évalue le système complet avec toutes les métriques
    """
    
    def __init__(self, output_dir: str = "evaluation/results"):
        """
        Initialise l'évaluation
        
        Args:
            output_dir: Dossier pour sauvegarder les résultats
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialiser les composants
        self.blackboard = Blackboard()
        self.orchestrator = Orchestrator(self.blackboard)
        self.oulad = OULADIntegration(self.blackboard)
        
        # Métriques
        self.rec_metrics = RecommendationMetrics()
        self.gen_metrics = GenerationMetrics()
        self.xai_metrics = XAIMetrics()
        
        print(f"✓ System Evaluation initialisé")
        print(f"  Dossier de sortie: {output_dir}")
    
    def evaluate_recommendations(self, n_users: int = 50) -> Dict:
        """
        Évalue les recommandations sur N utilisateurs OULAD
        
        Args:
            n_users: Nombre d'utilisateurs à évaluer
        
        Returns:
            Dict avec les résultats des métriques
        """
        print(f"\n{'='*70}")
        print(f"📊 ÉVALUATION DES RECOMMANDATIONS ({n_users} utilisateurs)")
        print(f"{'='*70}")
        
        # Charger les utilisateurs
        students = self.oulad.load_multiple_students(n=n_users)
        
        batch_recommendations = []
        
        for i, student_id in enumerate(students, 1):
            print(f"\n[{i}/{len(students)}] Évaluation de {student_id}...")
            
            try:
                # Lancer l'analyse complète
                result = self.orchestrator.process_user_request(
                    student_id, 
                    request_type="full_analysis"
                )
                
                if result['overall_status'] != 'completed':
                    print(f"  ⚠️  Analyse échouée pour {student_id}")
                    continue
                
                # Récupérer les recommandations
                recommendations = self.blackboard.read("recommendations", student_id)
                
                if not recommendations:
                    print(f"  ⚠️  Pas de recommandations pour {student_id}")
                    continue
                
                # Extraire les IDs recommandés
                recommended_ids = [
                    rec['resource_id'] 
                    for rec in recommendations['recommendations']
                ]
                
                # Générer un ground truth simple
                # (Dans un vrai système, utiliser les interactions futures)
                relevant_ids = self._generate_ground_truth(student_id)
                
                batch_recommendations.append((recommended_ids, relevant_ids))
                
                print(f"  ✓ {len(recommended_ids)} recommandations, {len(relevant_ids)} pertinents")
                
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
                continue
        
        # Calculer les métriques
        print(f"\n📊 Calcul des métriques sur {len(batch_recommendations)} utilisateurs...")
        
        if not batch_recommendations:
            print("⚠️  Aucune donnée à évaluer")
            return {}
        
        results = self.rec_metrics.evaluate_batch(
            batch_recommendations,
            k_values=[3, 5, 10]
        )
        
        print_metrics_report(results, title="Résultats Recommandations")
        
        return results
    
    def evaluate_generation(self, n_samples: int = 20) -> Dict:
        """
        Évalue la qualité du contenu généré
        
        Args:
            n_samples: Nombre d'échantillons à évaluer
        
        Returns:
            Dict avec les résultats des métriques
        """
        print(f"\n{'='*70}")
        print(f"📝 ÉVALUATION DE LA GÉNÉRATION DE CONTENU ({n_samples} échantillons)")
        print(f"{'='*70}")
        
        # Récupérer les contenus générés du cache
        cached_content = self.blackboard.read_section("cached_content")
        
        if not cached_content:
            print("⚠️  Aucun contenu généré dans le cache")
            return {}
        
        # Prendre un échantillon
        sample_ids = list(cached_content.keys())[:n_samples]
        
        batch_generation = []
        
        for i, content_id in enumerate(sample_ids, 1):
            content = cached_content[content_id]
            
            print(f"\n[{i}/{len(sample_ids)}] Évaluation de {content_id}...")
            
            # Extraire le texte généré
            generated_text = self._extract_generated_text(content)
            
            # Créer une référence (dans un vrai système, utiliser des références humaines)
            reference_text = self._create_reference_text(content)
            
            if generated_text and reference_text:
                batch_generation.append((generated_text, reference_text))
                print(f"  ✓ Texte: {len(generated_text)} caractères")
        
        # Calculer les métriques
        print(f"\n📊 Calcul des métriques sur {len(batch_generation)} textes...")
        
        if not batch_generation:
            print("⚠️  Aucune donnée à évaluer")
            return {}
        
        results = self.gen_metrics.evaluate_batch(batch_generation)
        
        print_generation_metrics_report(results, title="Résultats Génération")
        
        return results
    
    def evaluate_xai(self, n_users: int = 30) -> Dict:
        """
        Évalue la qualité des explications XAI
        
        Args:
            n_users: Nombre d'utilisateurs à évaluer
        
        Returns:
            Dict avec les résultats des métriques
        """
        print(f"\n{'='*70}")
        print(f"🔍 ÉVALUATION DES EXPLICATIONS XAI ({n_users} utilisateurs)")
        print(f"{'='*70}")
        
        # Récupérer les explications du Blackboard
        explanations_section = self.blackboard.read_section("explanations")
        
        if not explanations_section:
            print("⚠️  Aucune explication dans le Blackboard")
            return {}
        
        # Prendre un échantillon
        sample_ids = list(explanations_section.keys())[:n_users]
        
        batch_explanations = []
        batch_features = []
        batch_importance = []
        
        for i, user_id in enumerate(sample_ids, 1):
            explanation = explanations_section[user_id]
            
            print(f"\n[{i}/{len(sample_ids)}] Évaluation XAI de {user_id}...")
            
            # Récupérer le profil pour les features
            profile = self.blackboard.read("profiles", user_id)
            
            if profile:
                actual_features = {
                    "level": profile['level'],
                    "learning_style": profile['learning_style']
                }
                
                feature_importance = {
                    "level": 0.35,
                    "learning_style": 0.25,
                    "interests": 0.20
                }
                
                batch_explanations.append(explanation)
                batch_features.append(actual_features)
                batch_importance.append(feature_importance)
                
                print(f"  ✓ Explication avec {len(explanation)} sections")
        
        # Calculer les métriques
        print(f"\n📊 Calcul des métriques XAI sur {len(batch_explanations)} explications...")
        
        if not batch_explanations:
            print("⚠️  Aucune donnée à évaluer")
            return {}
        
        results = self.xai_metrics.evaluate_batch(
            batch_explanations,
            batch_features,
            batch_importance
        )
        
        print_xai_metrics_report(results, title="Résultats XAI")
        
        return results
    
    def run_complete_evaluation(self, n_users: int = 30) -> Dict:
        """
        Lance une évaluation complète du système
        
        Args:
            n_users: Nombre d'utilisateurs à évaluer
        
        Returns:
            Dict avec tous les résultats
        """
        print(f"\n" + "#"*70)
        print(f"# ÉVALUATION COMPLÈTE DU SYSTÈME MULTI-AGENTS")
        print(f"# Dataset: OULAD | Users: {n_users}")
        print(f"#"*70)
        
        start_time = datetime.now()
        
        results = {
            "metadata": {
                "evaluation_date": start_time.isoformat(),
                "n_users": n_users,
                "dataset": "OULAD",
                "system_version": "1.0"
            },
            "recommendations": {},
            "generation": {},
            "xai": {}
        }
        
        # 1. Évaluer les recommandations
        try:
            results["recommendations"] = self.evaluate_recommendations(n_users)
        except Exception as e:
            print(f"\n❌ Erreur évaluation recommandations: {e}")
        
        # 2. Évaluer la génération
        try:
            results["generation"] = self.evaluate_generation(n_samples=min(20, n_users))
        except Exception as e:
            print(f"\n❌ Erreur évaluation génération: {e}")
        
        # 3. Évaluer XAI
        try:
            results["xai"] = self.evaluate_xai(n_users)
        except Exception as e:
            print(f"\n❌ Erreur évaluation XAI: {e}")
        
        # Temps d'exécution
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results["metadata"]["duration_seconds"] = duration
        
        # Sauvegarder les résultats
        self._save_results(results)
        
        # Afficher le résumé final
        self._print_final_summary(results)
        
        return results
    
    def _generate_ground_truth(self, student_id: str) -> List[str]:
        """
        Génère un ground truth simple pour les recommandations
        
        Dans un vrai système, utiliser les interactions futures de l'étudiant
        
        Args:
            student_id: ID de l'étudiant
        
        Returns:
            Liste des ressources pertinentes
        """
        # Simulé : utiliser le parcours planifié comme référence
        learning_path = self.blackboard.read("learning_paths", student_id)
        
        if learning_path and 'path' in learning_path:
            return [step['resource_id'] for step in learning_path['path'][:5]]
        
        return []
    
    def _extract_generated_text(self, content: Dict) -> str:
        """
        Extrait le texte généré d'un contenu
        
        Args:
            content: Contenu généré
        
        Returns:
            Texte généré
        """
        if 'content' in content:
            content_data = content['content']
            
            if isinstance(content_data, dict):
                return content_data.get('full_text', '')
        
        return ""
    
    def _create_reference_text(self, content: Dict) -> str:
        """
        Crée un texte de référence simple
        
        Dans un vrai système, utiliser des références créées par des experts
        
        Args:
            content: Contenu généré
        
        Returns:
            Texte de référence
        """
        # Référence simple basée sur le sujet
        topic = content.get('topic', 'programming')
        level = content.get('level', 'beginner')
        content_type = content.get('type', 'course')
        
        reference = f"This {content_type} covers {topic} concepts at {level} level. "
        reference += f"It provides clear explanations and practical examples for learners."
        
        return reference
    
    def _save_results(self, results: Dict):
        """
        Sauvegarde les résultats en JSON
        
        Args:
            results: Dict des résultats
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {filepath}")
    
    def _print_final_summary(self, results: Dict):
        """
        Affiche un résumé final des résultats
        
        Args:
            results: Dict des résultats
        """
        print(f"\n" + "="*70)
        print(f"📊 RÉSUMÉ FINAL DE L'ÉVALUATION")
        print(f"="*70)
        
        print(f"\n⏱️  Durée totale: {results['metadata'].get('duration_seconds', 0):.2f} secondes")
        print(f"👥 Utilisateurs évalués: {results['metadata']['n_users']}")
        
        # Recommandations
        if results['recommendations']:
            print(f"\n🎯 Recommandations:")
            for metric in ['NDCG@5', 'MRR', 'Recall@10']:
                if metric in results['recommendations']:
                    print(f"  • {metric:15s} : {results['recommendations'][metric]:.4f}")
        
        # Génération
        if results['generation']:
            print(f"\n📝 Génération de Contenu:")
            for metric in ['ROUGE-1_f1', 'BERTScore']:
                if metric in results['generation']:
                    print(f"  • {metric:15s} : {results['generation'][metric]:.4f}")
        
        # XAI
        if results['xai']:
            print(f"\n🔍 Explicabilité (XAI):")
            for metric in ['faithfulness', 'plausibility', 'trust_score']:
                if metric in results['xai']:
                    print(f"  • {metric:15s} : {results['xai'][metric]:.4f}")
        
        print(f"\n{'='*70}")
        print(f"✅ ÉVALUATION COMPLÈTE TERMINÉE !")
        print(f"{'='*70}\n")


def run_evaluation(n_users: int = 30):
    """
    Point d'entrée pour lancer l'évaluation
    
    Args:
        n_users: Nombre d'utilisateurs à évaluer
    """
    evaluator = SystemEvaluation()
    results = evaluator.run_complete_evaluation(n_users=n_users)
    return results


if __name__ == "__main__":
    # Lancer l'évaluation avec 30 utilisateurs
    run_evaluation(n_users=30)