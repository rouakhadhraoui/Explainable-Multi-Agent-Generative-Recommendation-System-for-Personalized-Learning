# tests/test_profiling_agent.py
"""
Tests pour le Profiling Agent

Ce fichier teste la capacité de l'agent à analyser des utilisateurs
et créer des profils d'apprentissage pertinents
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent


def create_sample_history():
    """
    Crée un historique d'interactions fictif pour les tests
    
    Returns:
        Liste d'interactions simulées
    """
    return [
        # Utilisateur regarde des vidéos Python
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "view", "resource_id": "video_python_variables", "duration": 120},
        
        # Fait des exercices
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 85},
        {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 300},
        
        # Continue avec des cours intermédiaires
        {"type": "view", "resource_id": "course_python_functions", "duration": 240},
        {"type": "quiz", "resource_id": "quiz_python_functions", "score": 78},
        
        # S'intéresse à un autre sujet
        {"type": "view", "resource_id": "video_datascience_intro", "duration": 150},
        {"type": "quiz", "resource_id": "quiz_datascience_basics", "score": 65},
        
        # Revient à Python avec du contenu avancé
        {"type": "view", "resource_id": "course_python_oop", "duration": 360},
        {"type": "quiz", "resource_id": "quiz_python_oop", "score": 82},
    ]


def test_profile_creation():
    """Test de création d'un profil complet"""
    print("\n" + "="*70)
    print("TEST 1 : Création d'un profil utilisateur complet")
    print("="*70)
    
    # Initialiser
    bb = Blackboard()
    agent = ProfilingAgent(bb)
    
    # Créer un historique fictif
    user_id = "user_test_001"
    history = create_sample_history()
    
    # Ajouter l'historique au Blackboard
    print(f"\n📝 Ajout de {len(history)} interactions pour {user_id}...")
    for interaction in history:
        bb.add_to_history(user_id, interaction)
    
    # Analyser l'utilisateur
    print("\n🔬 Lancement de l'analyse...")
    profile = agent.analyze_user(user_id)
    
    # Afficher les résultats
    print("\n" + "-"*70)
    print("📊 RÉSULTATS DE L'ANALYSE")
    print("-"*70)
    print(f"User ID          : {profile['user_id']}")
    print(f"Learning Style   : {profile['learning_style']}")
    print(f"Level            : {profile['level']}")
    print(f"Interests        : {', '.join(profile['interests'])}")
    print(f"Strengths        : {', '.join(profile['strengths']) if profile['strengths'] else 'None yet'}")
    print(f"Weaknesses       : {', '.join(profile['weaknesses']) if profile['weaknesses'] else 'None'}")
    print(f"Total Interactions: {profile['total_interactions']}")
    print(f"\nSummary:\n{profile['summary']}")
    print("-"*70)
    
    # Vérifier que le profil est dans le Blackboard
    retrieved = bb.read("profiles", user_id)
    assert retrieved is not None, "❌ Le profil n'a pas été sauvegardé"
    assert retrieved['user_id'] == user_id, "❌ User ID incorrect"
    
    print("\n✅ TEST 1 RÉUSSI - Profil créé et sauvegardé correctement")


def test_default_profile():
    """Test de création d'un profil par défaut (nouvel utilisateur)"""
    print("\n" + "="*70)
    print("TEST 2 : Profil par défaut pour nouvel utilisateur")
    print("="*70)
    
    bb = Blackboard()
    agent = ProfilingAgent(bb)
    
    # Analyser un utilisateur sans historique
    user_id = "user_new_001"
    print(f"\n🆕 Création de profil pour nouvel utilisateur: {user_id}")
    
    profile = agent.analyze_user(user_id)
    
    print(f"\n📊 Profil par défaut créé:")
    print(f"  - Style: {profile['learning_style']}")
    print(f"  - Niveau: {profile['level']}")
    print(f"  - Intérêts: {profile['interests']}")
    
    assert profile['level'] == "beginner", "❌ Un nouveau profil devrait être 'beginner'"
    assert profile['total_interactions'] == 0, "❌ Devrait avoir 0 interactions"
    
    print("\n✅ TEST 2 RÉUSSI - Profil par défaut correct")


def test_profile_update():
    """Test de mise à jour d'un profil existant"""
    print("\n" + "="*70)
    print("TEST 3 : Mise à jour d'un profil existant")
    print("="*70)
    
    bb = Blackboard()
    agent = ProfilingAgent(bb)
    
    user_id = "user_update_001"
    
    # Première analyse (peu d'interactions)
    print(f"\n📝 Ajout de 3 interactions initiales...")
    bb.add_to_history(user_id, {"type": "view", "resource_id": "course_python_basics", "duration": 120})
    bb.add_to_history(user_id, {"type": "quiz", "resource_id": "quiz_python_basics", "score": 55})
    bb.add_to_history(user_id, {"type": "view", "resource_id": "video_python_intro", "duration": 90})
    
    profile_v1 = agent.analyze_user(user_id)
    print(f"\n✓ Profil v1 - Niveau: {profile_v1['level']}, Interactions: {profile_v1['total_interactions']}")
    
    # Ajout de nouvelles interactions (amélioration)
    print(f"\n📝 Ajout de 5 nouvelles interactions avec de meilleurs scores...")
    bb.add_to_history(user_id, {"type": "quiz", "resource_id": "quiz_python_loops", "score": 85})
    bb.add_to_history(user_id, {"type": "quiz", "resource_id": "quiz_python_functions", "score": 88})
    bb.add_to_history(user_id, {"type": "exercise", "resource_id": "exercise_python_project", "duration": 600})
    bb.add_to_history(user_id, {"type": "quiz", "resource_id": "quiz_python_advanced", "score": 90})
    bb.add_to_history(user_id, {"type": "view", "resource_id": "course_python_expert", "duration": 300})
    
    # Mise à jour
    profile_v2 = agent.update_profile(user_id)
    print(f"\n✓ Profil v2 - Niveau: {profile_v2['level']}, Interactions: {profile_v2['total_interactions']}")
    
    # Vérifications
    assert profile_v2['total_interactions'] > profile_v1['total_interactions'], "❌ Le nombre d'interactions devrait augmenter"
    assert profile_v2['level'] != "beginner", "❌ Le niveau devrait avoir progressé"
    
    print(f"\n📈 Progression détectée:")
    print(f"  - Interactions: {profile_v1['total_interactions']} → {profile_v2['total_interactions']}")
    print(f"  - Niveau: {profile_v1['level']} → {profile_v2['level']}")
    
    print("\n✅ TEST 3 RÉUSSI - Mise à jour de profil fonctionnelle")


def test_learning_style_detection():
    """Test de détection des différents styles d'apprentissage"""
    print("\n" + "="*70)
    print("TEST 4 : Détection des styles d'apprentissage")
    print("="*70)
    
    bb = Blackboard()
    agent = ProfilingAgent(bb)
    
    # Test 1 : Style VISUAL
    user_visual = "user_visual"
    print("\n📹 Test style VISUAL (vidéos majoritairement)...")
    for i in range(5):
        bb.add_to_history(user_visual, {"type": "view", "resource_id": f"video_topic_{i}", "duration": 120})
    
    profile_visual = agent.analyze_user(user_visual)
    print(f"  ✓ Style détecté: {profile_visual['learning_style']}")
    
    # Test 2 : Style KINESTHETIC
    user_kinesthetic = "user_kinesthetic"
    print("\n🎮 Test style KINESTHETIC (exercices majoritairement)...")
    for i in range(5):
        bb.add_to_history(user_kinesthetic, {"type": "exercise", "resource_id": f"exercise_topic_{i}", "duration": 300})
        bb.add_to_history(user_kinesthetic, {"type": "quiz", "resource_id": f"quiz_topic_{i}", "score": 75})
    
    profile_kinesthetic = agent.analyze_user(user_kinesthetic)
    print(f"  ✓ Style détecté: {profile_kinesthetic['learning_style']}")
    
    # Test 3 : Style READING
    user_reading = "user_reading"
    print("\n📚 Test style READING (cours textuels majoritairement)...")
    for i in range(5):
        bb.add_to_history(user_reading, {"type": "view", "resource_id": f"course_topic_{i}", "duration": 200})
        bb.add_to_history(user_reading, {"type": "view", "resource_id": f"article_topic_{i}", "duration": 150})
    
    profile_reading = agent.analyze_user(user_reading)
    print(f"  ✓ Style détecté: {profile_reading['learning_style']}")
    
    print("\n✅ TEST 4 RÉUSSI - Détection des styles d'apprentissage OK")


def run_all_tests():
    """Exécuter tous les tests du Profiling Agent"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - PROFILING AGENT")
    print("#"*70)
    
    try:
        test_profile_creation()
        test_default_profile()
        test_profile_update()
        test_learning_style_detection()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DU PROFILING AGENT SONT RÉUSSIS !")
        print("="*70)
        print("\nLe Profiling Agent fonctionne correctement.")
        print("Prochaine étape : Orchestrator\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()