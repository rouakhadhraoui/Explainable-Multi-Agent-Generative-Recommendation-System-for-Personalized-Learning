# tests/test_xai_agent.py
"""
Tests pour le XAI Agent

Ce fichier teste la génération d'explications pour toutes les décisions
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent
from agents.path_planning_agent import PathPlanningAgent
from agents.content_generator import ContentGenerator
from agents.recommendation_agent import RecommendationAgent
from agents.xai_agent import XAIAgent


def setup_complete_system(bb, user_id):
    """
    Configure un système complet avec tous les agents pour un utilisateur
    
    Args:
        bb: Blackboard
        user_id: ID de l'utilisateur
    """
    # Créer les agents
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    # Ajouter historique
    interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 75},
        {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 300},
        {"type": "quiz", "resource_id": "quiz_python_loops", "score": 82},
    ]
    
    for interaction in interactions:
        bb.add_to_history(user_id, interaction)
    
    # Exécuter le pipeline
    profiling_agent.analyze_user(user_id)
    planning_agent.plan_learning_path(user_id)
    content_gen.generate_content(user_id, "course", "python", "intermediate")
    rec_agent.generate_recommendations(user_id, top_k=5)
    
    print(f"✓ Système complet configuré pour {user_id}")
    
    return profiling_agent, planning_agent, content_gen, rec_agent


def test_full_explanation():
    """Test de génération d'explications complètes"""
    print("\n" + "="*70)
    print("TEST 1 : Génération d'explications COMPLÈTES")
    print("="*70)
    
    bb = Blackboard()
    user_id = "test_xai_user_001"
    
    # Configurer le système
    setup_complete_system(bb, user_id)
    
    # Créer l'agent XAI
    xai_agent = XAIAgent(bb)
    
    # Générer toutes les explications
    explanations = xai_agent.explain_full_system(user_id)
    
    # Vérifications
    assert "error" not in explanations, "❌ Erreur lors de la génération"
    assert "profile_explanation" in explanations, "❌ Explication du profil manquante"
    assert "path_explanation" in explanations, "❌ Explication du parcours manquante"
    assert "recommendations_explanation" in explanations, "❌ Explication des recommandations manquante"
    assert "counterfactuals" in explanations, "❌ Contrefactuels manquants"
    assert "summary" in explanations, "❌ Résumé manquant"
    
    # Afficher les explications
    print(f"\n📊 EXPLICATIONS GÉNÉRÉES:")
    print(f"  User ID: {explanations['user_id']}")
    
    print(f"\n1️⃣ EXPLICATION DU PROFIL:")
    profile_exp = explanations['profile_explanation']
    print(f"   • Level: {profile_exp.get('level_reasoning', 'N/A')[:100]}...")
    print(f"   • Style: {profile_exp.get('style_reasoning', 'N/A')[:100]}...")
    
    print(f"\n2️⃣ EXPLICATION DU PARCOURS:")
    if explanations['path_explanation']:
        path_exp = explanations['path_explanation']
        print(f"   • Logic: {path_exp.get('path_logic', 'N/A')[:100]}...")
    
    print(f"\n3️⃣ EXPLICATION DES RECOMMANDATIONS:")
    if explanations['recommendations_explanation']:
        rec_exp = explanations['recommendations_explanation']
        print(f"   • Criteria: {rec_exp.get('selection_criteria', 'N/A')[:100]}...")
    
    print(f"\n4️⃣ CONTREFACTUELS:")
    cf = explanations['counterfactuals']
    print(f"   • If higher level: {cf.get('if_higher_level', 'N/A')[:80]}...")
    
    print(f"\n5️⃣ RÉSUMÉ GLOBAL:")
    print(f"   {explanations['summary'][:200]}...")
    
    print("\n✅ TEST 1 RÉUSSI - Explications complètes générées")


def test_profile_explanation():
    """Test d'explication du profil uniquement"""
    print("\n" + "="*70)
    print("TEST 2 : Explication du PROFIL")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    xai_agent = XAIAgent(bb)
    
    user_id = "test_profile_exp"
    
    # Créer un profil
    interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 55},
    ]
    
    for i in interactions:
        bb.add_to_history(user_id, i)
    
    profiling_agent.analyze_user(user_id)
    
    # Expliquer le profil
    explanations = xai_agent.explain_full_system(user_id)
    profile_exp = explanations['profile_explanation']
    
    # Vérifications
    assert 'level_reasoning' in profile_exp, "❌ Raisonnement niveau manquant"
    assert 'style_reasoning' in profile_exp, "❌ Raisonnement style manquant"
    assert 'improvement_suggestions' in profile_exp, "❌ Suggestions manquantes"
    
    print(f"\n📝 EXPLICATION DU PROFIL:")
    print(f"\n  Niveau ({bb.read('profiles', user_id)['level']}):")
    print(f"  {profile_exp['level_reasoning']}")
    
    print(f"\n  Style ({bb.read('profiles', user_id)['learning_style']}):")
    print(f"  {profile_exp['style_reasoning']}")
    
    print(f"\n  Suggestions d'amélioration:")
    print(f"  {profile_exp['improvement_suggestions']}")
    
    print("\n✅ TEST 2 RÉUSSI - Explication du profil détaillée")


def test_counterfactuals():
    """Test des explications contrefactuelles"""
    print("\n" + "="*70)
    print("TEST 3 : Explications CONTREFACTUELLES")
    print("="*70)
    
    bb = Blackboard()
    user_id = "test_counterfactual"
    
    setup_complete_system(bb, user_id)
    
    xai_agent = XAIAgent(bb)
    explanations = xai_agent.explain_full_system(user_id)
    
    counterfactuals = explanations['counterfactuals']
    
    # Vérifications
    assert 'if_higher_level' in counterfactuals, "❌ Contrefactuel niveau manquant"
    assert 'if_different_style' in counterfactuals, "❌ Contrefactuel style manquant"
    assert 'if_more_practice' in counterfactuals, "❌ Contrefactuel pratique manquant"
    
    print(f"\n💭 SCÉNARIOS CONTREFACTUELS:")
    
    print(f"\n  🔼 Si niveau supérieur:")
    print(f"     {counterfactuals['if_higher_level']}")
    
    print(f"\n  🔄 Si style différent:")
    print(f"     {counterfactuals['if_different_style']}")
    
    print(f"\n  📈 Si plus de pratique:")
    print(f"     {counterfactuals['if_more_practice']}")
    
    print("\n✅ TEST 3 RÉUSSI - Contrefactuels générés")


def test_feature_importance():
    """Test de l'importance des features"""
    print("\n" + "="*70)
    print("TEST 4 : IMPORTANCE DES FEATURES")
    print("="*70)
    
    bb = Blackboard()
    user_id = "test_features"
    
    setup_complete_system(bb, user_id)
    
    xai_agent = XAIAgent(bb)
    importance = xai_agent.get_feature_importance(user_id)
    
    # Vérifications
    assert "feature_importance" in importance, "❌ Importance des features manquante"
    
    print(f"\n⚖️  IMPORTANCE DES FEATURES:")
    
    # Trier par importance décroissante
    sorted_features = sorted(
        importance['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for feature, score in sorted_features:
        bar = "█" * int(score * 40)  # Barre visuelle
        print(f"  {feature:20s} : {bar} {score:.0%}")
    
    print(f"\n  💬 {importance['explanation']}")
    
    print("\n✅ TEST 4 RÉUSSI - Importance des features calculée")


def test_specific_decision_explanation():
    """Test d'explication de décisions spécifiques"""
    print("\n" + "="*70)
    print("TEST 5 : Explications de DÉCISIONS SPÉCIFIQUES")
    print("="*70)
    
    bb = Blackboard()
    user_id = "test_decision"
    
    setup_complete_system(bb, user_id)
    
    xai_agent = XAIAgent(bb)
    
    # Test 1: Explication du profil
    print(f"\n1️⃣ Explication: PROFIL")
    profile_explanation = xai_agent.explain_decision("profile", user_id)
    print(f"   {profile_explanation[:150]}...")
    assert len(profile_explanation) > 0, "❌ Explication vide"
    
    # Test 2: Explication du parcours
    print(f"\n2️⃣ Explication: PARCOURS")
    path_explanation = xai_agent.explain_decision("path", user_id)
    print(f"   {path_explanation[:150]}...")
    assert len(path_explanation) > 0, "❌ Explication vide"
    
    # Test 3: Explication des recommandations
    print(f"\n3️⃣ Explication: RECOMMANDATIONS")
    rec_explanation = xai_agent.explain_decision("recommendation", user_id)
    print(f"   {rec_explanation[:150]}...")
    assert len(rec_explanation) > 0, "❌ Explication vide"
    
    print("\n✅ TEST 5 RÉUSSI - Explications spécifiques générées")


def test_explanation_caching():
    """Test de la mise en cache des explications"""
    print("\n" + "="*70)
    print("TEST 6 : Mise en cache des EXPLICATIONS")
    print("="*70)
    
    bb = Blackboard()
    user_id = "test_cache"
    
    setup_complete_system(bb, user_id)
    
    xai_agent = XAIAgent(bb)
    xai_agent.explain_full_system(user_id)
    
    # Vérifier dans le Blackboard
    cached_exp = bb.read("explanations", user_id)
    
    assert cached_exp is not None, "❌ Explications non mises en cache"
    assert cached_exp['user_id'] == user_id, "❌ User ID incorrect"
    
    print(f"\n💾 Explications en cache:")
    print(f"  • User ID  : {cached_exp['user_id']}")
    print(f"  • Timestamp: {cached_exp['timestamp']}")
    print(f"  • Sections : {list(cached_exp.keys())}")
    
    print("\n✅ TEST 6 RÉUSSI - Explications mises en cache")


def run_all_tests():
    """Exécuter tous les tests du XAI Agent"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - XAI AGENT")
    print("#"*70)
    
    try:
        test_full_explanation()
        test_profile_explanation()
        test_counterfactuals()
        test_feature_importance()
        test_specific_decision_explanation()
        test_explanation_caching()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DU XAI AGENT SONT RÉUSSIS !")
        print("="*70)
        print("\n✨ FÉLICITATIONS ! Tous les agents sont maintenant fonctionnels :")
        print("   ✅ Profiling Agent")
        print("   ✅ Path Planning Agent")
        print("   ✅ Content Generator Agent")
        print("   ✅ Recommendation Agent")
        print("   ✅ XAI Agent")
        print("\n🎯 Le système multi-agents complet est opérationnel !\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()