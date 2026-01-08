# tests/test_generation_metrics.py
"""
Tests pour les métriques de génération de contenu

Vérifie que les métriques ROUGE, BERTScore, etc. fonctionnent
"""

import sys
sys.path.append('..')

from evaluation.generation_metrics import GenerationMetrics, print_generation_metrics_report


def test_rouge_1():
    """Test du ROUGE-1"""
    print("\n" + "="*70)
    print("TEST 1 : ROUGE-1 (Unigrammes)")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    # Cas 1 : Textes identiques
    generated = "Python is a programming language"
    reference = "Python is a programming language"
    
    rouge1 = metrics.rouge_n(generated, reference, n=1)
    print(f"\n📊 Cas 1 : Textes identiques")
    print(f"  Généré     : {generated}")
    print(f"  Référence  : {reference}")
    print(f"  Precision  : {rouge1['precision']:.4f}")
    print(f"  Recall     : {rouge1['recall']:.4f}")
    print(f"  F1         : {rouge1['f1']:.4f}")
    
    assert rouge1['f1'] == 1.0, "❌ F1 devrait être 1.0 pour textes identiques"
    
    # Cas 2 : Chevauchement partiel
    generated2 = "Python is a great programming language"
    reference2 = "Python is a programming language"
    
    rouge1_2 = metrics.rouge_n(generated2, reference2, n=1)
    print(f"\n📊 Cas 2 : Chevauchement partiel")
    print(f"  Généré     : {generated2}")
    print(f"  Référence  : {reference2}")
    print(f"  Precision  : {rouge1_2['precision']:.4f}")
    print(f"  Recall     : {rouge1_2['recall']:.4f}")
    print(f"  F1         : {rouge1_2['f1']:.4f}")
    
    assert 0 < rouge1_2['f1'] < 1, "❌ F1 devrait être entre 0 et 1"
    
    # Cas 3 : Aucun chevauchement
    generated3 = "Machine learning algorithms"
    reference3 = "Python programming language"
    
    rouge1_3 = metrics.rouge_n(generated3, reference3, n=1)
    print(f"\n📊 Cas 3 : Aucun chevauchement")
    print(f"  Généré     : {generated3}")
    print(f"  Référence  : {reference3}")
    print(f"  F1         : {rouge1_3['f1']:.4f}")
    
    assert rouge1_3['f1'] == 0.0, "❌ F1 devrait être 0.0"
    
    print("\n✅ TEST 1 RÉUSSI - ROUGE-1 correctement implémenté")


def test_rouge_2():
    """Test du ROUGE-2"""
    print("\n" + "="*70)
    print("TEST 2 : ROUGE-2 (Bigrammes)")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    generated = "Python is a great programming language for beginners"
    reference = "Python is a programming language for beginners"
    
    rouge2 = metrics.rouge_n(generated, reference, n=2)
    print(f"\n📊 Test ROUGE-2")
    print(f"  Généré     : {generated}")
    print(f"  Référence  : {reference}")
    print(f"  Precision  : {rouge2['precision']:.4f}")
    print(f"  Recall     : {rouge2['recall']:.4f}")
    print(f"  F1         : {rouge2['f1']:.4f}")
    
    assert 0 <= rouge2['f1'] <= 1, "❌ F1 devrait être entre 0 et 1"
    
    print("\n✅ TEST 2 RÉUSSI - ROUGE-2 correctement implémenté")


def test_rouge_l():
    """Test du ROUGE-L"""
    print("\n" + "="*70)
    print("TEST 3 : ROUGE-L (Longest Common Subsequence)")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    generated = "Python is widely used for data science"
    reference = "Python is used for data analysis and science"
    
    rougeL = metrics.rouge_l(generated, reference)
    print(f"\n📊 Test ROUGE-L")
    print(f"  Généré     : {generated}")
    print(f"  Référence  : {reference}")
    print(f"  Precision  : {rougeL['precision']:.4f}")
    print(f"  Recall     : {rougeL['recall']:.4f}")
    print(f"  F1         : {rougeL['f1']:.4f}")
    
    assert 0 <= rougeL['f1'] <= 1, "❌ F1 devrait être entre 0 et 1"
    
    print("\n✅ TEST 3 RÉUSSI - ROUGE-L correctement implémenté")


def test_bertscore():
    """Test du BERTScore"""
    print("\n" + "="*70)
    print("TEST 4 : BERTScore (Similarité Sémantique)")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    # Phrases sémantiquement similaires
    generated = "Python is a popular programming language"
    reference = "Python is a widely-used coding language"
    
    print(f"\n🔄 Chargement du modèle d'embeddings...")
    bertscore = metrics.bertscore_simple(generated, reference)
    
    print(f"\n📊 Test BERTScore")
    print(f"  Généré     : {generated}")
    print(f"  Référence  : {reference}")
    print(f"  BERTScore  : {bertscore:.4f}")
    
    assert 0 <= bertscore <= 1, "❌ BERTScore devrait être entre 0 et 1"
    assert bertscore > 0.5, "❌ Phrases similaires devraient avoir BERTScore > 0.5"
    
    print("\n✅ TEST 4 RÉUSSI - BERTScore correctement implémenté")


def test_readability():
    """Test des métriques de lisibilité"""
    print("\n" + "="*70)
    print("TEST 5 : Métriques de Lisibilité")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    text = "Python is a high-level programming language. It is easy to learn and widely used."
    
    readability = metrics.readability_score(text)
    
    print(f"\n📖 Texte analysé:")
    print(f"  {text}")
    
    print(f"\n📊 Métriques de lisibilité:")
    print(f"  Longueur moy. mots : {readability['avg_word_length']:.2f}")
    print(f"  Total mots         : {readability['total_words']}")
    print(f"  Mots uniques       : {readability['unique_words']}")
    print(f"  Diversité lexicale : {readability['lexical_diversity']:.4f}")
    
    assert readability['total_words'] > 0, "❌ Devrait avoir des mots"
    assert 0 <= readability['lexical_diversity'] <= 1, "❌ Diversité entre 0 et 1"
    
    print("\n✅ TEST 5 RÉUSSI - Métriques de lisibilité OK")


def test_coherence():
    """Test de la cohérence"""
    print("\n" + "="*70)
    print("TEST 6 : Score de Cohérence")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    # Texte avec connecteurs
    text1 = "Python is popular. However, it can be slow. Therefore, optimization is important."
    coherence1 = metrics.coherence_score(text1)
    
    print(f"\n📊 Texte avec connecteurs:")
    print(f"  {text1}")
    print(f"  Cohérence : {coherence1:.4f}")
    
    # Texte sans connecteurs
    text2 = "Python is popular. It can be slow. Optimization is important."
    coherence2 = metrics.coherence_score(text2)
    
    print(f"\n📊 Texte sans connecteurs:")
    print(f"  {text2}")
    print(f"  Cohérence : {coherence2:.4f}")
    
    assert coherence1 > coherence2, "❌ Texte avec connecteurs devrait avoir meilleure cohérence"
    
    print("\n✅ TEST 6 RÉUSSI - Score de cohérence OK")


def test_evaluate_all():
    """Test de l'évaluation complète"""
    print("\n" + "="*70)
    print("TEST 7 : Évaluation Complète")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    generated = """Python is a high-level programming language. 
    It is widely used for data science and machine learning. 
    Python has a simple syntax that makes it easy to learn."""
    
    reference = """Python is a popular programming language. 
    It is commonly used for data analysis and AI. 
    Python has clear syntax that makes it beginner-friendly."""
    
    print(f"\n📝 Texte généré:")
    print(f"  {generated[:80]}...")
    
    print(f"\n📝 Texte de référence:")
    print(f"  {reference[:80]}...")
    
    results = metrics.evaluate_all(generated, reference)
    
    print_generation_metrics_report(results, title="Résultats d'Évaluation")
    
    # Vérifications
    assert 'ROUGE-1_f1' in results, "❌ ROUGE-1 F1 manquant"
    assert 'BERTScore' in results, "❌ BERTScore manquant"
    assert 'coherence' in results, "❌ Cohérence manquante"
    
    print("✅ TEST 7 RÉUSSI - Évaluation complète fonctionnelle")


def test_batch_evaluation():
    """Test de l'évaluation en batch"""
    print("\n" + "="*70)
    print("TEST 8 : Évaluation en BATCH")
    print("="*70)
    
    metrics = GenerationMetrics()
    
    batch = [
        ("Python is great", "Python is excellent"),
        ("Java is popular", "Java is widely used"),
        ("C++ is fast", "C++ is efficient")
    ]
    
    avg_results = metrics.evaluate_batch(batch)
    
    print_generation_metrics_report(avg_results, title="Résultats Moyens (3 textes)")
    
    # Vérifications
    assert 'ROUGE-1_f1_std' in avg_results, "❌ Écart-type manquant"
    assert 'BERTScore_std' in avg_results, "❌ Écart-type BERTScore manquant"
    
    print("✅ TEST 8 RÉUSSI - Évaluation batch fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS - MÉTRIQUES DE GÉNÉRATION")
    print("#"*70)
    
    try:
        test_rouge_1()
        test_rouge_2()
        test_rouge_l()
        test_bertscore()
        test_readability()
        test_coherence()
        test_evaluate_all()
        test_batch_evaluation()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DES MÉTRIQUES DE GÉNÉRATION RÉUSSIS !")
        print("="*70)
        print("\n✅ ROUGE, BERTScore et autres métriques implémentées")
        print("✅ Prêt pour évaluer la qualité du contenu généré\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()