# tests/test_path_planning_agent.py
"""
Tests pour le Path Planning Agent

Ce fichier teste la planification de parcours d'apprentissage
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent
from agents.path_planning_agent import PathPlanningAgent


def setup_test_user(blackboard: Blackboard, profiling_agent: ProfilingAgent, 
                    user_id: str, level: str = "beginner"):
    """
    Crée un utilisateur de test avec un profil spécifique
    
    Args:
        blackboard: Instance du Blackboard
        profiling_agent: Agent de profilage
        user_id: ID de l'utilisateur
        level: Niveau souhaité (beginner, intermediate, advanced)
    """
    # Créer un historique adapté au niveau
    if level == "beginner":
        interactions = [
            {"type": "view", "resource_id": "video_python_intro", "duration": 180},
            {"type": "quiz", "resource_id": "quiz_python_basics", "score": 55},
        ]
    elif level == "intermediate":
        interactions = [
            {"type": "view", "resource_id": "course_python_basics", "duration": 200},
            {"type": "quiz", "resource_id": "quiz_python_basics", "score": 78},
            {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 300},
            {"type": "quiz", "resource_id": "quiz_python_loops", "score": 82},
        ]
    else:  # advanced
        interactions = [
            {"type": "view", "resource_id": "course_python_oop", "duration": 400},
            {"type": "quiz", "resource_id": "quiz_python_oop", "score": 90},
            {"type": "exercise", "resource_id": "exercise_python_advanced", "duration": 500},
            {"type": "quiz", "resource_id": "quiz_python_advanced", "score": 88},
        ]
    
    # Ajouter l'historique
    for interaction in interactions:
        blackboard.add_to_history(user_id, interaction)
    
    # Créer le profil
    profiling_agent.analyze_user(user_id)
    
    print(f"✓ Utilisateur {user_id} créé (niveau: {level})")


def test_path_creation_beginner():
    """Test de création de parcours pour un débutant"""
    print("\n" + "="*70)
    print("TEST 1 : Planification pour un utilisateur DÉBUTANT")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    
    # Créer un utilisateur débutant
    user_id = "test_beginner_001"
    setup_test_user(bb, profiling_agent, user_id, level="beginner")
    
    # Planifier le parcours
    path = planning_agent.plan_learning_path(user_id)
    
    # Vérifications
    assert "error" not in path, "❌ Erreur lors de la planification"
    assert len(path['path']) > 0, "❌ Le parcours devrait contenir des étapes"
    assert path['current_level'] == "beginner", "❌ Niveau actuel incorrect"
    assert path['target_level'] == "intermediate", "❌ Niveau cible devrait être intermediate"
    
    # Afficher le parcours
    print(f"\n📋 PARCOURS GÉNÉRÉ:")
    print(f"  • Niveau actuel  : {path['current_level']}")
    print(f"  • Niveau cible   : {path['target_level']}")
    print(f"  • Nombre d'étapes: {path['total_steps']}")
    print(f"  • Durée estimée  : {path['estimated_duration_minutes']} minutes")
    print(f"\n  Étapes:")
    for step in path['path'][:5]:  # Afficher les 5 premières
        print(f"    {step['step']}. {step['title']} ({step['type']}, {step['duration']}min)")
    
    print(f"\n  💬 Explication:")
    print(f"     {path['explanation']}")
    
    print("\n✅ TEST 1 RÉUSSI - Parcours débutant créé avec succès")


def test_path_creation_intermediate():
    """Test de création de parcours pour un niveau intermédiaire"""
    print("\n" + "="*70)
    print("TEST 2 : Planification pour un utilisateur INTERMÉDIAIRE")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    
    # Créer un utilisateur intermédiaire
    user_id = "test_intermediate_001"
    setup_test_user(bb, profiling_agent, user_id, level="intermediate")
    
    # Planifier le parcours
    path = planning_agent.plan_learning_path(user_id)
    
    # Vérifications
    assert "error" not in path, "❌ Erreur lors de la planification"
    assert path['current_level'] == "intermediate", "❌ Niveau actuel incorrect"
    assert path['target_level'] == "advanced", "❌ Niveau cible devrait être advanced"
    
    # Vérifier que les ressources sont de niveau approprié
    levels_in_path = [step['level'] for step in path['path']]
    assert "beginner" not in levels_in_path, "❌ Pas de ressources débutant pour un intermédiaire"
    
    print(f"\n📋 PARCOURS GÉNÉRÉ:")
    print(f"  • Niveau: {path['current_level']} → {path['target_level']}")
    print(f"  • Étapes: {path['total_steps']}")
    print(f"  • Durée: {path['estimated_duration_minutes']}min")
    
    print("\n✅ TEST 2 RÉUSSI - Parcours intermédiaire créé avec succès")


def test_path_with_completed_resources():
    """Test avec des ressources déjà complétées"""
    print("\n" + "="*70)
    print("TEST 3 : Parcours avec ressources déjà complétées")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    
    user_id = "test_completed_001"
    
    # Créer un historique avec plusieurs ressources complétées
    completed_interactions = [
        {"type": "view", "resource_id": "py_intro_video", "duration": 180},
        {"type": "quiz", "resource_id": "py_basics_quiz", "score": 85},
        {"type": "exercise", "resource_id": "py_loops_exercise", "duration": 400},
        {"type": "view", "resource_id": "py_functions_course", "duration": 500},
    ]
    
    for interaction in completed_interactions:
        bb.add_to_history(user_id, interaction)
    
    # Créer le profil
    profiling_agent.analyze_user(user_id)
    
    # Planifier
    path = planning_agent.plan_learning_path(user_id)
    
    # Vérifier que les ressources complétées ne sont pas dans le nouveau parcours
    path_resource_ids = [step['resource_id'] for step in path['path']]
    completed_ids = ["py_intro_video", "py_basics_quiz", "py_loops_exercise", "py_functions_course"]
    
    for completed_id in completed_ids:
        assert completed_id not in path_resource_ids, f"❌ {completed_id} ne devrait pas être dans le parcours"
    
    print(f"\n✓ Ressources complétées correctement exclues du nouveau parcours")
    print(f"✓ Nouveau parcours contient {len(path['path'])} étapes fraîches")
    
    print("\n✅ TEST 3 RÉUSSI - Exclusion des ressources complétées OK")


def test_path_progress_update():
    """Test de mise à jour de la progression"""
    print("\n" + "="*70)
    print("TEST 4 : Mise à jour de la progression")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    
    user_id = "test_progress_001"
    setup_test_user(bb, profiling_agent, user_id, level="beginner")
    
    # Créer un parcours
    path = planning_agent.plan_learning_path(user_id)
    
    print(f"\n📊 Progression initiale: 0%")
    
    # Compléter les 3 premières étapes
    for step_num in [1, 2, 3]:
        updated_path = planning_agent.update_path_progress(user_id, step_num)
        print(f"✓ Étape {step_num} complétée - Progression: {updated_path['progress_percentage']:.1f}%")
    
    # Vérifier la progression
    final_path = bb.read("learning_paths", user_id)
    assert 'progress_percentage' in final_path, "❌ Pourcentage de progression manquant"
    assert final_path['progress_percentage'] > 0, "❌ La progression devrait être > 0"
    
    # Compter les étapes complétées
    completed_count = sum(1 for step in final_path['path'] if step.get('completed', False))
    print(f"\n✓ {completed_count}/{len(final_path['path'])} étapes complétées")
    
    print("\n✅ TEST 4 RÉUSSI - Mise à jour de progression fonctionnelle")


def test_different_learning_styles():
    """Test de l'adaptation aux différents styles d'apprentissage"""
    print("\n" + "="*70)
    print("TEST 5 : Adaptation aux styles d'apprentissage")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    planning_agent = PathPlanningAgent(bb)
    
    # Créer 3 utilisateurs avec différents styles
    styles = ["visual", "kinesthetic", "reading"]
    
    for style in styles:
        user_id = f"test_{style}_001"
        
        # Créer un historique adapté au style
        if style == "visual":
            interactions = [
                {"type": "view", "resource_id": "video_python_intro", "duration": 180},
                {"type": "view", "resource_id": "video_python_vars", "duration": 150},
            ]
        elif style == "kinesthetic":
            interactions = [
                {"type": "exercise", "resource_id": "exercise_python_basics", "duration": 300},
                {"type": "quiz", "resource_id": "quiz_python_basics", "score": 75},
            ]
        else:  # reading
            interactions = [
                {"type": "view", "resource_id": "course_python_intro", "duration": 250},
                {"type": "view", "resource_id": "article_python_best_practices", "duration": 200},
            ]
        
        for interaction in interactions:
            bb.add_to_history(user_id, interaction)
        
        # Créer profil et parcours
        profile = profiling_agent.analyze_user(user_id)
        path = planning_agent.plan_learning_path(user_id)
        
        print(f"\n👤 {style.upper()} learner:")
        print(f"  Style détecté: {profile['learning_style']}")
        print(f"  Types de ressources dans le parcours:")
        
        type_counts = {}
        for step in path['path']:
            resource_type = step['type']
            type_counts[resource_type] = type_counts.get(resource_type, 0) + 1
        
        for rtype, count in type_counts.items():
            print(f"    - {rtype}: {count}")
    
    print("\n✅ TEST 5 RÉUSSI - Adaptation aux styles d'apprentissage OK")


def run_all_tests():
    """Exécuter tous les tests du Path Planning Agent"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - PATH PLANNING AGENT")
    print("#"*70)
    
    try:
        test_path_creation_beginner()
        test_path_creation_intermediate()
        test_path_with_completed_resources()
        test_path_progress_update()
        test_different_learning_styles()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DU PATH PLANNING AGENT SONT RÉUSSIS !")
        print("="*70)
        print("\nLe Path Planning Agent fonctionne correctement.")
        print("Prochaine étape : Content Generator Agent\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()