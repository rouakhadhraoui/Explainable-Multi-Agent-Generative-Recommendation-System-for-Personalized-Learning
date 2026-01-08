# tests/test_content_generator.py
"""
Tests pour le Content Generator Agent

Ce fichier teste la génération de contenu pédagogique personnalisé
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent
from agents.content_generator import ContentGenerator


def setup_test_user(blackboard: Blackboard, profiling_agent: ProfilingAgent, user_id: str):
    """
    Crée un utilisateur de test avec un profil
    
    Args:
        blackboard: Instance du Blackboard
        profiling_agent: Agent de profilage
        user_id: ID de l'utilisateur
    """
    interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 75},
        {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 300},
    ]
    
    for interaction in interactions:
        blackboard.add_to_history(user_id, interaction)
    
    profiling_agent.analyze_user(user_id)
    print(f"✓ Utilisateur {user_id} créé")


def test_course_generation():
    """Test de génération d'un cours"""
    print("\n" + "="*70)
    print("TEST 1 : Génération d'un COURS")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_course_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    # Générer un cours
    content = content_gen.generate_content(
        user_id=user_id,
        content_type="course",
        topic="python",
        level="beginner"
    )
    
    # Vérifications
    assert "error" not in content, "❌ Erreur lors de la génération"
    assert content['type'] == "course", "❌ Type incorrect"
    assert content['topic'] == "python", "❌ Topic incorrect"
    assert 'content' in content, "❌ Contenu manquant"
    
    # Afficher le contenu généré
    print(f"\n📚 COURS GÉNÉRÉ:")
    print(f"  • ID          : {content['content_id']}")
    print(f"  • Type        : {content['type']}")
    print(f"  • Topic       : {content['topic']}")
    print(f"  • Level       : {content['level']}")
    print(f"  • Style       : {content['learning_style']}")
    print(f"  • Sources RAG : {', '.join(content['rag_sources'])}")
    
    course_content = content['content']
    print(f"\n  📖 Contenu du cours:")
    print(f"     Title: {course_content.get('title', 'N/A')}")
    print(f"     Intro: {course_content.get('introduction', 'N/A')[:100]}...")
    
    # Vérifier qu'il est dans le cache
    cached = bb.read("cached_content", content['content_id'])
    assert cached is not None, "❌ Contenu non mis en cache"
    
    print("\n✅ TEST 1 RÉUSSI - Cours généré avec succès")


def test_exercise_generation():
    """Test de génération d'un exercice"""
    print("\n" + "="*70)
    print("TEST 2 : Génération d'un EXERCICE")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_exercise_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    # Générer un exercice
    content = content_gen.generate_content(
        user_id=user_id,
        content_type="exercise",
        topic="python",
        level="intermediate"
    )
    
    # Vérifications
    assert content['type'] == "exercise", "❌ Type incorrect"
    assert 'content' in content, "❌ Contenu manquant"
    
    exercise_content = content['content']
    print(f"\n🎮 EXERCICE GÉNÉRÉ:")
    print(f"  • Title      : {exercise_content.get('title', 'N/A')}")
    print(f"  • Description: {exercise_content.get('description', 'N/A')[:100]}...")
    print(f"  • Hints      : {exercise_content.get('hints', 'N/A')[:100]}...")
    
    print("\n✅ TEST 2 RÉUSSI - Exercice généré avec succès")


def test_quiz_generation():
    """Test de génération d'un quiz"""
    print("\n" + "="*70)
    print("TEST 3 : Génération d'un QUIZ")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_quiz_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    # Générer un quiz
    content = content_gen.generate_content(
        user_id=user_id,
        content_type="quiz",
        topic="python",
        level="beginner"
    )
    
    # Vérifications
    assert content['type'] == "quiz", "❌ Type incorrect"
    assert 'content' in content, "❌ Contenu manquant"
    
    quiz_content = content['content']
    print(f"\n❓ QUIZ GÉNÉRÉ:")
    print(f"  • Title     : {quiz_content.get('title', 'N/A')}")
    print(f"  • Questions : {quiz_content.get('total_questions', 0)}")
    
    if quiz_content.get('questions'):
        print(f"\n  Exemple de question:")
        q1 = quiz_content['questions'][0]
        print(f"     Q{q1['question_number']}: {q1['question']}")
    
    print("\n✅ TEST 3 RÉUSSI - Quiz généré avec succès")


def test_different_levels():
    """Test de génération pour différents niveaux"""
    print("\n" + "="*70)
    print("TEST 4 : Génération pour différents NIVEAUX")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_levels_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    levels = ["beginner", "intermediate", "advanced"]
    
    for level in levels:
        print(f"\n📊 Génération niveau {level.upper()}...")
        
        content = content_gen.generate_content(
            user_id=user_id,
            content_type="course",
            topic="python",
            level=level
        )
        
        assert content['level'] == level, f"❌ Niveau {level} incorrect"
        print(f"  ✓ Cours de niveau {level} généré")
    
    print("\n✅ TEST 4 RÉUSSI - Génération multi-niveaux OK")


def test_rag_context_usage():
    """Test de l'utilisation du contexte RAG"""
    print("\n" + "="*70)
    print("TEST 5 : Utilisation du contexte RAG")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_rag_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    # Générer du contenu
    content = content_gen.generate_content(
        user_id=user_id,
        content_type="course",
        topic="python",
        level="intermediate"
    )
    
    # Vérifier que des sources RAG ont été utilisées
    assert len(content['rag_sources']) > 0, "❌ Aucune source RAG utilisée"
    
    print(f"\n📚 Sources RAG utilisées:")
    for source in content['rag_sources']:
        print(f"  • {source}")
    
    print(f"\n✓ {len(content['rag_sources'])} sources RAG intégrées au contenu")
    
    print("\n✅ TEST 5 RÉUSSI - Contexte RAG correctement utilisé")


def test_content_caching():
    """Test de la mise en cache du contenu"""
    print("\n" + "="*70)
    print("TEST 6 : Mise en cache du contenu")
    print("="*70)
    
    bb = Blackboard()
    profiling_agent = ProfilingAgent(bb)
    content_gen = ContentGenerator(bb)
    
    user_id = "test_cache_user"
    setup_test_user(bb, profiling_agent, user_id)
    
    # Générer plusieurs contenus
    print(f"\n🔄 Génération de 3 contenus différents...")
    
    contents = []
    for i, ctype in enumerate(["course", "exercise", "quiz"], 1):
        content = content_gen.generate_content(
            user_id=user_id,
            content_type=ctype,
            topic="python",
            level="beginner"
        )
        contents.append(content)
        print(f"  {i}. {ctype} généré")
    
    # Vérifier le cache
    cached_section = bb.read_section("cached_content")
    print(f"\n💾 Contenus en cache: {len(cached_section)}")
    
    assert len(cached_section) >= 3, "❌ Tous les contenus ne sont pas en cache"
    
    # Vérifier qu'on peut récupérer les contenus
    for content in contents:
        cached = bb.read("cached_content", content['content_id'])
        assert cached is not None, f"❌ Contenu {content['content_id']} introuvable"
        print(f"  ✓ {content['type']} récupéré du cache")
    
    print("\n✅ TEST 6 RÉUSSI - Mise en cache fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests du Content Generator"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - CONTENT GENERATOR")
    print("#"*70)
    
    try:
        test_course_generation()
        test_exercise_generation()
        test_quiz_generation()
        test_different_levels()
        test_rag_context_usage()
        test_content_caching()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS DU CONTENT GENERATOR SONT RÉUSSIS !")
        print("="*70)
        print("\nLe Content Generator fonctionne correctement.")
        print("Prochaine étape : Recommendation Agent\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()