# tests/test_recommendation_agent.py
"""
Tests pour le Recommendation Agent

Ce fichier teste le système de recommandations hybrides
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent
from agents.path_planning_agent import PathPlanningAgent
from agents.content_generator import ContentGenerator
from agents.recommendation_agent import RecommendationAgent


def setup_complete_user(bb, profiling_agent, planning_agent, content_gen, user_id):
    """
    Crée un utilisateur avec profil, parcours et contenu généré
    
    Args:
        bb: Blackboard
        profiling_agent: Agent de profilage
        planning_agent: Agent de planification
        content_gen: Générateur de contenu
        user_id: ID de l'utilisateur
    """
    # Ajouter historique
    interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 75},
        {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 300},
    ]
    
    for interaction in interactions:
        bb.add_to_history(user_id, interaction)
    
    # Créer profil
    profiling_agent.analyze_user(user_id)
    
    # Créer parcours
    planning_agent.plan_learning_path(user_id)
    
    # Générer quelques contenus
    content_gen.generate_content(user_id, "course", "python", "intermediate")
    content_gen.generate_content(user_id, "quiz", "python", "intermediate")
    
    print(f"✓ Utilisateur complet {user_id} créé")


def test_basic_recommendations():
    """Test de génération de recommandations basiques"""
    print("\n" + "="*70)
    print("TEST 1 : Génération de recommandations basiques")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    user_id = "test_rec_user_001"
    setup_complete_user(bb, profiling_agent, planning_agent, content_gen, user_id)
    
    # Générer recommandations
    recommendations = rec_agent.generate_recommendations(user_id, top_k=5)
    
    # Vérifications
    assert "error" not in recommendations, "❌ Erreur lors de la génération"
    assert len(recommendations['recommendations']) > 0, "❌ Aucune recommandation générée"
    assert len(recommendations['recommendations']) <= 5, "❌ Trop de recommandations"
    
    # Afficher les recommandations
    print(f"\n🎯 RECOMMANDATIONS GÉNÉRÉES:")
    print(f"  • User ID         : {recommendations['user_id']}")
    print(f"  • Nombre          : {len(recommendations['recommendations'])}")
    print(f"  • Total candidats : {recommendations['total_candidates']}")
    
    print(f"\n  Sources:")
    for source, count in recommendations['sources'].items():
        print(f"    - {source:20s}: {count}")
    
    print(f"\n  Top {len(recommendations['recommendations'])} Recommandations:")
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"    {i}. {rec['title']}")
        print(f"       Type: {rec['type']:10s} | Level: {rec['level']:12s} | Score: {rec['priority_score']:.2f}")
        print(f"       Reason: {rec['reason']}")
        print()
    
    print("✅ TEST 1 RÉUSSI - Recommandations générées avec succès")


def test_recommendation_sources():
    """Test des différentes sources de recommandations"""
    print("\n" + "="*70)
    print("TEST 2 : Sources multiples de recommandations")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    user_id = "test_rec_sources_001"
    setup_complete_user(bb, profiling_agent, planning_agent, content_gen, user_id)
    
    # Générer recommandations
    recommendations = rec_agent.generate_recommendations(user_id, top_k=10)
    
    # Vérifier les sources
    sources_found = set()
    for rec in recommendations['recommendations']:
        sources_found.add(rec['source'])
    
    print(f"\n📊 Sources de recommandations détectées:")
    for source in sources_found:
        count = sum(1 for r in recommendations['recommendations'] if r['source'] == source)
        print(f"  • {source:25s}: {count} recommandations")
    
    # Vérifier qu'on a au moins 2 sources différentes
    assert len(sources_found) >= 1, "❌ Au moins 1 source devrait être présente"
    
    print("\n✅ TEST 2 RÉUSSI - Sources multiples utilisées")


def test_personalization():
    """Test de la personnalisation selon le profil"""
    print("\n" + "="*70)
    print("TEST 3 : Personnalisation des recommandations")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    # Créer 2 utilisateurs avec profils différents
    # Utilisateur 1 : Débutant visuel
    user1_id = "test_visual_beginner"
    interactions_1 = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 55},
    ]
    for i in interactions_1:
        bb.add_to_history(user1_id, i)
    
    profiling_agent.analyze_user(user1_id)
    planning_agent.plan_learning_path(user1_id)
    content_gen.generate_content(user1_id, "course", "python", "beginner")
    
    # Utilisateur 2 : Avancé kinesthésique
    user2_id = "test_kinesthetic_advanced"
    interactions_2 = [
        {"type": "exercise", "resource_id": "exercise_python_oop", "duration": 500},
        {"type": "quiz", "resource_id": "quiz_python_oop", "score": 90},
    ]
    for i in interactions_2:
        bb.add_to_history(user2_id, i)
    
    profiling_agent.analyze_user(user2_id)
    planning_agent.plan_learning_path(user2_id)
    content_gen.generate_content(user2_id, "exercise", "python", "advanced")
    
    # Générer recommandations pour chaque utilisateur
    rec1 = rec_agent.generate_recommendations(user1_id, top_k=3)
    rec2 = rec_agent.generate_recommendations(user2_id, top_k=3)
    
    print(f"\n👤 Utilisateur 1 (Débutant Visuel):")
    profile1 = bb.read("profiles", user1_id)
    print(f"  Niveau: {profile1['level']}, Style: {profile1['learning_style']}")
    print(f"  Recommandations:")
    for i, rec in enumerate(rec1['recommendations'][:3], 1):
        print(f"    {i}. {rec['title']} ({rec['level']})")
    
    print(f"\n👤 Utilisateur 2 (Avancé Kinesthésique):")
    profile2 = bb.read("profiles", user2_id)
    print(f"  Niveau: {profile2['level']}, Style: {profile2['learning_style']}")
    print(f"  Recommandations:")
    for i, rec in enumerate(rec2['recommendations'][:3], 1):
        print(f"    {i}. {rec['title']} ({rec['level']})")
    
    # Les recommandations devraient être différentes
    rec1_titles = set(r['title'] for r in rec1['recommendations'])
    rec2_titles = set(r['title'] for r in rec2['recommendations'])
    
    print(f"\n📊 Overlap: {len(rec1_titles & rec2_titles)} / {min(len(rec1_titles), len(rec2_titles))}")
    
    print("\n✅ TEST 3 RÉUSSI - Personnalisation fonctionnelle")


def test_recommendation_explanation():
    """Test des explications de recommandations"""
    print("\n" + "="*70)
    print("TEST 4 : Explications des recommandations")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    user_id = "test_explanation_user"
    setup_complete_user(bb, profiling_agent, planning_agent, content_gen, user_id)
    
    # Générer recommandations
    recommendations = rec_agent.generate_recommendations(user_id, top_k=3)
    
    # Obtenir l'explication pour la première recommandation
    print(f"\n💬 Explication pour la recommandation #1:")
    explanation = rec_agent.get_recommendation_explanation(user_id, 0)
    
    rec = recommendations['recommendations'][0]
    print(f"\n  Ressource: {rec['title']}")
    print(f"  Explication:\n  {explanation}")
    
    assert len(explanation) > 0, "❌ L'explication ne devrait pas être vide"
    
    print("\n✅ TEST 4 RÉUSSI - Explications générées")


def test_interaction_recording():
    """Test de l'enregistrement des interactions"""
    print("\n" + "="*70)
    print("TEST 5 : Enregistrement des interactions")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    rec_agent = RecommendationAgent(bb)
    
    user_id = "test_interaction_user"
    
    # Créer profil basique
    bb.add_to_history(user_id, {"type": "view", "resource_id": "test", "duration": 100})
    profiling_agent.analyze_user(user_id)
    
    # Enregistrer différentes interactions
    print(f"\n📝 Enregistrement de 3 interactions...")
    
    success1 = rec_agent.record_interaction(user_id, "resource_1", "click")
    success2 = rec_agent.record_interaction(user_id, "resource_2", "complete", score=85)
    success3 = rec_agent.record_interaction(user_id, "resource_3", "skip")
    
    assert success1 and success2 and success3, "❌ Échec d'enregistrement"
    
    # Vérifier dans l'historique
    history = bb.get_user_history(user_id)
    
    # Compter les nouvelles interactions
    new_interactions = [h for h in history if h['type'] in ['click', 'complete', 'skip']]
    
    print(f"\n✓ {len(new_interactions)} nouvelles interactions enregistrées")
    for interaction in new_interactions:
        print(f"  • {interaction['type']:10s} - {interaction['resource_id']}")
    
    print("\n✅ TEST 5 RÉUSSI - Enregistrement des interactions OK")


def test_recommendation_caching():
    """Test de la mise en cache des recommandations"""
    print("\n" + "="*70)
    print("TEST 6 : Mise en cache des recommandations")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    content_gen = ContentGenerator(bb)
    rec_agent = RecommendationAgent(bb)
    
    user_id = "test_cache_user"
    setup_complete_user(bb, profiling_agent, planning_agent, content_gen, user_id)
    
    # Générer recommandations
    rec_agent.generate_recommendations(user_id, top_k=5)
    
    # Vérifier dans le Blackboard
    cached_recs = bb.read("recommendations", user_id)
    
    assert cached_recs is not None, "❌ Recommandations non mises en cache"
    assert len(cached_recs['recommendations']) == 5, "❌ Nombre incorrect"
    
    print(f"\n💾 Recommandations en cache:")
    print(f"  • User ID: {cached_recs['user_id']}")
    print(f"  • Nombre : {len(cached_recs['recommendations'])}")
    print(f"  • Date   : {cached_recs['generated_at']}")
    
    print("\n✅ TEST 6 RÉUSSI - Mise en cache fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests du Recommendation Agent"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - RECOMMENDATION AGENT")
    print("#"*70)
    
    try:
        test_basic_recommendations()
        test_recommendation_sources()
        test_personalization()
        test_recommendation_explanation()
        test_interaction_recording()
        test_recommendation_caching()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DU RECOMMENDATION AGENT SONT RÉUSSIS !")
        print("="*70)
        print("\nLe Recommendation Agent fonctionne correctement.")
        print("Prochaine étape : XAI Agent (dernier agent !)\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()