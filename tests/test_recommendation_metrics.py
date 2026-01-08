# tests/test_recommendation_metrics.py
"""
Tests pour les métriques de recommandation

Vérifie que les métriques sont correctement implémentées
"""

import sys
sys.path.append('..')

from evaluation.recommendation_metrics import RecommendationMetrics, print_metrics_report


def test_ndcg():
    """Test du NDCG"""
    print("\n" + "="*70)
    print("TEST 1 : NDCG (Normalized Discounted Cumulative Gain)")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Cas 1 : Recommandations parfaites
    recommended = ['A', 'B', 'C', 'D', 'E']
    relevant = ['A', 'B', 'C']
    
    ndcg = metrics.ndcg_at_k(recommended, relevant, k=5)
    print(f"\n📊 Cas 1 : Recommandations parfaites")
    print(f"  Recommandé : {recommended}")
    print(f"  Pertinent  : {relevant}")
    print(f"  NDCG@5     : {ndcg:.4f}")
    
    assert ndcg == 1.0, "❌ NDCG devrait être 1.0 pour des recommandations parfaites"
    
    # Cas 2 : Recommandations moyennes
    recommended2 = ['A', 'X', 'B', 'Y', 'C']
    relevant2 = ['A', 'B', 'C']
    
    ndcg2 = metrics.ndcg_at_k(recommended2, relevant2, k=5)
    print(f"\n📊 Cas 2 : Recommandations moyennes")
    print(f"  Recommandé : {recommended2}")
    print(f"  Pertinent  : {relevant2}")
    print(f"  NDCG@5     : {ndcg2:.4f}")
    
    assert 0 < ndcg2 < 1, "❌ NDCG devrait être entre 0 et 1"
    
    # Cas 3 : Aucune recommandation pertinente
    recommended3 = ['X', 'Y', 'Z']
    relevant3 = ['A', 'B', 'C']
    
    ndcg3 = metrics.ndcg_at_k(recommended3, relevant3, k=5)
    print(f"\n📊 Cas 3 : Aucune recommandation pertinente")
    print(f"  Recommandé : {recommended3}")
    print(f"  Pertinent  : {relevant3}")
    print(f"  NDCG@5     : {ndcg3:.4f}")
    
    assert ndcg3 == 0.0, "❌ NDCG devrait être 0.0 sans recommandations pertinentes"
    
    print("\n✅ TEST 1 RÉUSSI - NDCG correctement implémenté")


def test_mrr():
    """Test du MRR"""
    print("\n" + "="*70)
    print("TEST 2 : MRR (Mean Reciprocal Rank)")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Cas 1 : Premier item pertinent
    recommended = ['A', 'B', 'C']
    relevant = ['A']
    
    mrr = metrics.mrr(recommended, relevant)
    print(f"\n📊 Cas 1 : Premier item pertinent")
    print(f"  Recommandé : {recommended}")
    print(f"  Pertinent  : {relevant}")
    print(f"  MRR        : {mrr:.4f}")
    
    assert mrr == 1.0, "❌ MRR devrait être 1.0 si le premier item est pertinent"
    
    # Cas 2 : Troisième item pertinent
    recommended2 = ['X', 'Y', 'A', 'B']
    relevant2 = ['A']
    
    mrr2 = metrics.mrr(recommended2, relevant2)
    print(f"\n📊 Cas 2 : Troisième item pertinent")
    print(f"  Recommandé : {recommended2}")
    print(f"  Pertinent  : {relevant2}")
    print(f"  MRR        : {mrr2:.4f}")
    
    assert abs(mrr2 - 0.333) < 0.01, "❌ MRR devrait être ~0.33 (1/3)"
    
    # Cas 3 : Aucun item pertinent
    recommended3 = ['X', 'Y', 'Z']
    relevant3 = ['A']
    
    mrr3 = metrics.mrr(recommended3, relevant3)
    print(f"\n📊 Cas 3 : Aucun item pertinent")
    print(f"  Recommandé : {recommended3}")
    print(f"  Pertinent  : {relevant3}")
    print(f"  MRR        : {mrr3:.4f}")
    
    assert mrr3 == 0.0, "❌ MRR devrait être 0.0"
    
    print("\n✅ TEST 2 RÉUSSI - MRR correctement implémenté")


def test_recall():
    """Test du Recall@K"""
    print("\n" + "="*70)
    print("TEST 3 : Recall@K")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Cas : 2 sur 3 items pertinents trouvés
    recommended = ['A', 'B', 'X', 'Y', 'Z']
    relevant = ['A', 'B', 'C']
    
    recall = metrics.recall_at_k(recommended, relevant, k=5)
    print(f"\n📊 Cas : 2 sur 3 items pertinents trouvés")
    print(f"  Recommandé : {recommended}")
    print(f"  Pertinent  : {relevant}")
    print(f"  Recall@5   : {recall:.4f}")
    
    assert abs(recall - 0.667) < 0.01, "❌ Recall devrait être ~0.67 (2/3)"
    
    print("\n✅ TEST 3 RÉUSSI - Recall@K correctement implémenté")


def test_precision():
    """Test de la Precision@K"""
    print("\n" + "="*70)
    print("TEST 4 : Precision@K")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Cas : 2 sur 5 recommandations sont pertinentes
    recommended = ['A', 'B', 'X', 'Y', 'Z']
    relevant = ['A', 'B', 'C']
    
    precision = metrics.precision_at_k(recommended, relevant, k=5)
    print(f"\n📊 Cas : 2 sur 5 recommandations pertinentes")
    print(f"  Recommandé  : {recommended}")
    print(f"  Pertinent   : {relevant}")
    print(f"  Precision@5 : {precision:.4f}")
    
    assert abs(precision - 0.4) < 0.01, "❌ Precision devrait être 0.4 (2/5)"
    
    print("\n✅ TEST 4 RÉUSSI - Precision@K correctement implémenté")


def test_evaluate_all():
    """Test de l'évaluation complète"""
    print("\n" + "="*70)
    print("TEST 5 : Évaluation complète (toutes métriques)")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Recommandations réalistes
    recommended = ['res1', 'res2', 'res3', 'res4', 'res5', 
                   'res6', 'res7', 'res8', 'res9', 'res10']
    relevant = ['res2', 'res5', 'res11', 'res12']
    
    results = metrics.evaluate_all(recommended, relevant, k_values=[5, 10])
    
    print_metrics_report(results, title="Résultats d'Évaluation")
    
    # Vérifications
    assert 'NDCG@5' in results, "❌ NDCG@5 manquant"
    assert 'MRR' in results, "❌ MRR manquant"
    assert 'Recall@10' in results, "❌ Recall@10 manquant"
    
    print("✅ TEST 5 RÉUSSI - Évaluation complète fonctionnelle")


def test_batch_evaluation():
    """Test de l'évaluation en batch"""
    print("\n" + "="*70)
    print("TEST 6 : Évaluation en BATCH")
    print("="*70)
    
    metrics = RecommendationMetrics()
    
    # Simuler 3 utilisateurs
    batch = [
        (['A', 'B', 'C', 'D'], ['A', 'B']),
        (['X', 'A', 'Y', 'C'], ['A', 'C', 'Z']),
        (['M', 'N', 'O', 'P'], ['X', 'Y', 'Z'])
    ]
    
    avg_results = metrics.evaluate_batch(batch, k_values=[3, 5])
    
    print_metrics_report(avg_results, title="Résultats Moyens (3 utilisateurs)")
    
    # Vérifier les écarts-types
    assert 'NDCG@3_std' in avg_results, "❌ Écart-type NDCG@3 manquant"
    assert 'MRR_std' in avg_results, "❌ Écart-type MRR manquant"
    
    print("✅ TEST 6 RÉUSSI - Évaluation batch fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS - MÉTRIQUES DE RECOMMANDATION")
    print("#"*70)
    
    try:
        test_ndcg()
        test_mrr()
        test_recall()
        test_precision()
        test_evaluate_all()
        test_batch_evaluation()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DES MÉTRIQUES SONT RÉUSSIS !")
        print("="*70)
        print("\n✅ Les métriques de recommandation sont correctement implémentées")
        print("✅ Prêt pour l'évaluation du système avec OULAD\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()