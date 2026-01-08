# tests/test_xai_metrics.py
"""
Tests pour les métriques XAI (Explicabilité)

Vérifie que les métriques d'explicabilité sont correctement implémentées
"""

import sys
sys.path.append('..')

from evaluation.xai_metrics import XAIMetrics, print_xai_metrics_report


def test_faithfulness():
    """Test de la fidélité"""
    print("\n" + "="*70)
    print("TEST 1 : Faithfulness (Fidélité)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Cas 1 : Explication fidèle (mentionne les features importantes)
    explanation = {
        "level_reasoning": "The level beginner was assigned based on low quiz scores",
        "style_reasoning": "Visual learning style detected from video preferences"
    }
    
    actual_features = {
        "level": "beginner",
        "learning_style": "visual",
        "avg_score": 55
    }
    
    feature_importance = {
        "level": 0.35,
        "learning_style": 0.25,
        "avg_score": 0.20,
        "total_interactions": 0.10
    }
    
    faithfulness = metrics.faithfulness_score(explanation, actual_features, feature_importance)
    
    print(f"\n📊 Cas 1 : Explication fidèle")
    print(f"  Features importantes mentionnées")
    print(f"  Faithfulness : {faithfulness:.4f}")
    
    assert faithfulness > 0.5, "❌ Faithfulness devrait être > 0.5"
    
    # Cas 2 : Explication non fidèle
    explanation2 = {
        "reasoning": "The user likes Python"
    }
    
    faithfulness2 = metrics.faithfulness_score(explanation2, actual_features, feature_importance)
    
    print(f"\n📊 Cas 2 : Explication non fidèle")
    print(f"  Features importantes non mentionnées")
    print(f"  Faithfulness : {faithfulness2:.4f}")
    
    assert faithfulness2 < faithfulness, "❌ Faithfulness devrait être plus basse"
    
    print("\n✅ TEST 1 RÉUSSI - Faithfulness correctement implémenté")


def test_plausibility():
    """Test de la plausibilité"""
    print("\n" + "="*70)
    print("TEST 2 : Plausibility (Plausibilité)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Cas 1 : Explication plausible
    explanation1 = {
        "level_reasoning": "The beginner level was assigned because the average quiz score is 55%, which is below the intermediate threshold of 60%.",
        "style_reasoning": "Visual learning style was detected since the user primarily watched 5 video tutorials."
    }
    
    plausibility1 = metrics.plausibility_score(explanation1)
    
    print(f"\n📊 Cas 1 : Explication plausible et structurée")
    print(f"  Plausibility : {plausibility1:.4f}")
    
    # Cas 2 : Explication peu plausible
    explanation2 = "User is beginner"
    
    plausibility2 = metrics.plausibility_score(explanation2)
    
    print(f"\n📊 Cas 2 : Explication courte et vague")
    print(f"  Plausibility : {plausibility2:.4f}")
    
    assert plausibility1 > plausibility2, "❌ Explication 1 devrait être plus plausible"
    
    print("\n✅ TEST 2 RÉUSSI - Plausibility correctement implémenté")


def test_completeness():
    """Test de la complétude"""
    print("\n" + "="*70)
    print("TEST 3 : Completeness (Complétude)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Cas 1 : Explication complète
    explanation1 = {
        "level_reasoning": "Beginner level based on scores",
        "style_reasoning": "Visual style based on videos",
        "interests_reasoning": "Python interest from interactions"
    }
    
    required = ['level_reasoning', 'style_reasoning', 'interests_reasoning']
    
    completeness1 = metrics.completeness_score(explanation1, required)
    
    print(f"\n📊 Cas 1 : Explication complète (3/3 composants)")
    print(f"  Completeness : {completeness1:.4f}")
    
    assert completeness1 == 1.0, "❌ Completeness devrait être 1.0"
    
    # Cas 2 : Explication incomplète
    explanation2 = {
        "level_reasoning": "Beginner level based on scores"
    }
    
    completeness2 = metrics.completeness_score(explanation2, required)
    
    print(f"\n📊 Cas 2 : Explication incomplète (1/3 composants)")
    print(f"  Completeness : {completeness2:.4f}")
    
    assert completeness2 < 1.0, "❌ Completeness devrait être < 1.0"
    
    print("\n✅ TEST 3 RÉUSSI - Completeness correctement implémenté")


def test_trust_score():
    """Test du score de confiance"""
    print("\n" + "="*70)
    print("TEST 4 : Trust Score (Confiance)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Explication avec données concrètes
    explanation = {
        "level_reasoning": "Beginner level assigned based on average score of 55% across 10 interactions",
        "style_reasoning": "Visual learning style detected from 8 video views and 2 exercises"
    }
    
    confidence_indicators = {
        "data_quality": 0.9,
        "model_confidence": 0.85
    }
    
    trust = metrics.trust_score_heuristic(explanation, confidence_indicators)
    
    print(f"\n📊 Score de confiance")
    print(f"  Trust Score : {trust:.4f}")
    
    assert 0 <= trust <= 1, "❌ Trust score devrait être entre 0 et 1"
    
    print("\n✅ TEST 4 RÉUSSI - Trust Score correctement implémenté")


def test_consistency():
    """Test de la cohérence"""
    print("\n" + "="*70)
    print("TEST 5 : Consistency (Cohérence)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Explications cohérentes (mêmes thèmes)
    explanations1 = [
        {"reasoning": "Beginner level based on low scores"},
        {"reasoning": "Beginner level due to low performance"},
        {"reasoning": "Beginner assigned from poor scores"}
    ]
    
    consistency1 = metrics.consistency_score(explanations1)
    
    print(f"\n📊 Cas 1 : Explications cohérentes")
    print(f"  Consistency : {consistency1:.4f}")
    
    # Explications incohérentes
    explanations2 = [
        {"reasoning": "Beginner level based on scores"},
        {"reasoning": "Visual style from videos"},
        {"reasoning": "Python interest detected"}
    ]
    
    consistency2 = metrics.consistency_score(explanations2)
    
    print(f"\n📊 Cas 2 : Explications moins cohérentes")
    print(f"  Consistency : {consistency2:.4f}")
    
    assert consistency1 > consistency2, "❌ Cas 1 devrait être plus cohérent"
    
    print("\n✅ TEST 5 RÉUSSI - Consistency correctement implémenté")


def test_contrastive_quality():
    """Test de la qualité des contrefactuels"""
    print("\n" + "="*70)
    print("TEST 6 : Contrastive Quality (Qualité des contrefactuels)")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Bons contrefactuels
    counterfactuals1 = {
        "if_higher_level": "If your level was intermediate, you would receive more advanced resources",
        "if_different_style": "If you had a kinesthetic style, more exercises would be recommended",
        "if_more_practice": "If you completed 10 more interactions, your level could improve"
    }
    
    quality1 = metrics.contrastive_quality_score(counterfactuals1)
    
    print(f"\n📊 Cas 1 : Contrefactuels détaillés")
    print(f"  Contrastive Quality : {quality1:.4f}")
    
    # Contrefactuels vagues
    counterfactuals2 = {
        "scenario": "Things would be different"
    }
    
    quality2 = metrics.contrastive_quality_score(counterfactuals2)
    
    print(f"\n📊 Cas 2 : Contrefactuels vagues")
    print(f"  Contrastive Quality : {quality2:.4f}")
    
    assert quality1 > quality2, "❌ Cas 1 devrait avoir meilleure qualité"
    
    print("\n✅ TEST 6 RÉUSSI - Contrastive Quality correctement implémenté")


def test_evaluate_all():
    """Test de l'évaluation complète"""
    print("\n" + "="*70)
    print("TEST 7 : Évaluation XAI Complète")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # Explication complète
    explanation = {
        "profile_explanation": {
            "level_reasoning": "Beginner level due to average score of 58%",
            "style_reasoning": "Visual style from 10 video views",
            "interests_reasoning": "Python interest from interactions"
        },
        "path_explanation": {
            "path_logic": "Path starts with basics",
            "personalization": "Adapted to visual style",
            "expected_outcomes": "Master fundamentals"
        },
        "recommendations_explanation": {
            "selection_criteria": "Based on level and style",
            "ranking_logic": "Priority to foundational topics",
            "personalization_factors": "Visual resources prioritized"
        },
        "counterfactuals": {
            "if_higher_level": "More advanced content",
            "if_different_style": "Different resource types"
        }
    }
    
    actual_features = {
        "level": "beginner",
        "learning_style": "visual"
    }
    
    feature_importance = {
        "level": 0.35,
        "learning_style": 0.25
    }
    
    results = metrics.evaluate_all(explanation, actual_features, feature_importance)
    
    print_xai_metrics_report(results, title="Résultats d'Évaluation XAI")
    
    # Vérifications
    assert 'plausibility' in results, "❌ Plausibility manquante"
    assert 'profile_completeness' in results, "❌ Profile completeness manquante"
    assert 'trust_score' in results, "❌ Trust score manquant"
    
    print("✅ TEST 7 RÉUSSI - Évaluation XAI complète fonctionnelle")


def test_batch_evaluation():
    """Test de l'évaluation en batch"""
    print("\n" + "="*70)
    print("TEST 8 : Évaluation XAI en BATCH")
    print("="*70)
    
    metrics = XAIMetrics()
    
    # 3 explications
    batch = [
        {"profile_explanation": {"level_reasoning": "Beginner due to scores", "style_reasoning": "Visual from videos"}},
        {"profile_explanation": {"level_reasoning": "Intermediate from performance", "style_reasoning": "Kinesthetic from exercises"}},
        {"profile_explanation": {"level_reasoning": "Advanced based on results", "style_reasoning": "Reading from articles"}}
    ]
    
    avg_results = metrics.evaluate_batch(batch)
    
    print_xai_metrics_report(avg_results, title="Résultats Moyens XAI (3 explications)")
    
    # Vérifications
    assert 'plausibility_std' in avg_results, "❌ Écart-type manquant"
    assert 'consistency' in avg_results, "❌ Consistency manquante"
    
    print("✅ TEST 8 RÉUSSI - Évaluation batch XAI fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS - MÉTRIQUES XAI")
    print("#"*70)
    
    try:
        test_faithfulness()
        test_plausibility()
        test_completeness()
        test_trust_score()
        test_consistency()
        test_contrastive_quality()
        test_evaluate_all()
        test_batch_evaluation()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DES MÉTRIQUES XAI RÉUSSIS !")
        print("="*70)
        print("\n✅ Toutes les métriques d'explicabilité implémentées")
        print("✅ Faithfulness, Plausibility, Trust Score validés")
        print("\n🎯 ÉTAPE 10 COMPLÈTE - Toutes les métriques d'évaluation sont prêtes !\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()