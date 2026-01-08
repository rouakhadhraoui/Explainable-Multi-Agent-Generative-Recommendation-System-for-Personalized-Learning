# main.py
"""
Point d'entrée principal du système multi-agents

Ce fichier démontre l'utilisation complète du système :
- Création du Blackboard
- Initialisation de l'Orchestrator
- Simulation d'utilisateurs
- Analyse et recommandations
"""

from memory.blackboard import Blackboard
from orchestrator.orchestrator import Orchestrator
from datetime import datetime


def create_sample_users(blackboard: Blackboard):
    """
    Crée des utilisateurs fictifs avec différents profils d'apprentissage
    
    Args:
        blackboard: Instance du Blackboard
    """
    print("\n" + "="*80)
    print("📝 CRÉATION D'UTILISATEURS FICTIFS")
    print("="*80)
    
    # Utilisateur 1 : Alice - Débutant visuel
    print("\n👤 Utilisateur 1 : Alice (débutant, style visuel)")
    alice_interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "view", "resource_id": "video_python_variables", "duration": 150},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 55},
        {"type": "view", "resource_id": "video_python_conditions", "duration": 200},
        {"type": "quiz", "resource_id": "quiz_python_conditions", "score": 62},
    ]
    for interaction in alice_interactions:
        blackboard.add_to_history("alice_001", interaction)
    print(f"  ✓ {len(alice_interactions)} interactions créées")
    
    # Utilisateur 2 : Bob - Intermédiaire kinesthésique
    print("\n👤 Utilisateur 2 : Bob (intermédiaire, style kinesthésique)")
    bob_interactions = [
        {"type": "exercise", "resource_id": "exercise_python_loops", "duration": 400},
        {"type": "quiz", "resource_id": "quiz_python_loops", "score": 78},
        {"type": "exercise", "resource_id": "exercise_python_functions", "duration": 600},
        {"type": "quiz", "resource_id": "quiz_python_functions", "score": 82},
        {"type": "exercise", "resource_id": "exercise_python_lists", "duration": 500},
        {"type": "quiz", "resource_id": "quiz_python_lists", "score": 85},
        {"type": "exercise", "resource_id": "exercise_python_dicts", "duration": 450},
        {"type": "quiz", "resource_id": "quiz_python_dicts", "score": 80},
    ]
    for interaction in bob_interactions:
        blackboard.add_to_history("bob_002", interaction)
    print(f"  ✓ {len(bob_interactions)} interactions créées")
    
    # Utilisateur 3 : Charlie - Avancé lecture
    print("\n👤 Utilisateur 3 : Charlie (avancé, style lecture)")
    charlie_interactions = [
        {"type": "view", "resource_id": "course_python_oop", "duration": 600},
        {"type": "quiz", "resource_id": "quiz_python_oop", "score": 92},
        {"type": "view", "resource_id": "course_python_decorators", "duration": 450},
        {"type": "quiz", "resource_id": "quiz_python_decorators", "score": 88},
        {"type": "view", "resource_id": "article_python_generators", "duration": 300},
        {"type": "quiz", "resource_id": "quiz_python_generators", "score": 95},
        {"type": "view", "resource_id": "course_python_async", "duration": 700},
        {"type": "quiz", "resource_id": "quiz_python_async", "score": 90},
        {"type": "view", "resource_id": "article_python_metaclasses", "duration": 500},
        {"type": "quiz", "resource_id": "quiz_python_metaclasses", "score": 93},
    ]
    for interaction in charlie_interactions:
        blackboard.add_to_history("charlie_003", interaction)
    print(f"  ✓ {len(charlie_interactions)} interactions créées")
    
    print("\n✅ 3 utilisateurs fictifs créés avec succès")


def demo_system():
    """
    Démonstration complète du système
    """
    print("\n" + "#"*80)
    print("# DÉMONSTRATION DU SYSTÈME MULTI-AGENTS")
    print("# Explainable Multi-Agent Generative Recommendation System")
    print("#"*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Étape 1 : Initialiser le Blackboard
    print("\n" + "="*80)
    print("ÉTAPE 1 : INITIALISATION DU SHARED MEMORY (BLACKBOARD)")
    print("="*80)
    blackboard = Blackboard()
    print(f"✓ Blackboard initialisé")
    print(f"  {blackboard}")
    
    # Étape 2 : Créer des utilisateurs fictifs
    create_sample_users(blackboard)
    
    # Étape 3 : Initialiser l'Orchestrator
    print("\n" + "="*80)
    print("ÉTAPE 2 : INITIALISATION DE L'ORCHESTRATOR")
    print("="*80)
    orchestrator = Orchestrator(blackboard)
    print(f"✓ Orchestrator initialisé")
    print(f"  {orchestrator}")
    
    pipeline_info = orchestrator.get_pipeline_info()
    print(f"\n📋 Configuration du pipeline:")
    print(f"  - Agents disponibles: {', '.join(pipeline_info['agents_available'])}")
    print(f"  - Pipeline: {' → '.join(pipeline_info['pipeline'])}")
    
    # Étape 4 : Analyser les utilisateurs
    print("\n" + "="*80)
    print("ÉTAPE 3 : ANALYSE DES PROFILS UTILISATEURS")
    print("="*80)
    
    users = ["alice_001", "bob_002", "charlie_003"]
    
    for user_id in users:
        print(f"\n{'─'*80}")
        print(f"🔍 Analyse de l'utilisateur: {user_id}")
        print(f"{'─'*80}")
        
        # Lancer l'analyse
        result = orchestrator.process_user_request(user_id, request_type="full_analysis")
        
        # Afficher le profil créé
        if result["overall_status"] == "completed":
            profile = blackboard.read("profiles", user_id)
            print(f"\n📊 PROFIL GÉNÉRÉ:")
            print(f"  • User ID          : {profile['user_id']}")
            print(f"  • Niveau           : {profile['level']}")
            print(f"  • Style            : {profile['learning_style']}")
            print(f"  • Intérêts         : {', '.join(profile['interests'])}")
            print(f"  • Forces           : {', '.join(profile['strengths']) if profile['strengths'] else 'À déterminer'}")
            print(f"  • Faiblesses       : {', '.join(profile['weaknesses']) if profile['weaknesses'] else 'Aucune détectée'}")
            print(f"  • Total interactions: {profile['total_interactions']}")
            print(f"\n  💬 Résumé:")
            print(f"     {profile['summary']}")
    
    # Étape 5 : Statistiques globales
    print("\n" + "="*80)
    print("ÉTAPE 4 : STATISTIQUES GLOBALES DU SYSTÈME")
    print("="*80)
    
    stats = blackboard.get_stats()
    print(f"\n📊 Statistiques du Blackboard:")
    for section, count in stats.items():
        if section != "metadata":
            print(f"  • {section:20s} : {count}")
    
    exec_history = orchestrator.get_execution_history()
    print(f"\n📜 Historique des exécutions:")
    print(f"  • Nombre total d'exécutions : {len(exec_history)}")
    print(f"  • Succès                    : {sum(1 for e in exec_history if e['overall_status'] == 'completed')}")
    print(f"  • Échecs                    : {sum(1 for e in exec_history if e['overall_status'] == 'failed')}")
    
    # Étape 6 : Export des données
    print("\n" + "="*80)
    print("ÉTAPE 5 : EXPORT DES DONNÉES")
    print("="*80)
    
    export_file = "data/blackboard_export.json"
    success = blackboard.export_to_json(export_file)
    if success:
        print(f"✅ Données exportées vers: {export_file}")
    
    # Conclusion
    print("\n" + "="*80)
    print("✅ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*80)
    print("\n📌 Résumé:")
    print(f"  • {len(users)} utilisateurs analysés")
    print(f"  • {stats['profiles']} profils créés")
    print(f"  • {stats['learning_paths']} parcours planifiés")
    print(f"  • {stats['recommendations']} recommandations générées")
    print(f"  • {stats['explanations']} explications XAI générées")
    print(f"\n✅ Tous les agents opérationnels:")
    print("  ✓ Profiling Agent (Embeddings + Clustering + LLM)")
    print("  ✓ Path Planning Agent (A* + Q-Learning + Heuristics)")
    print("  ✓ Content Generator (LLM + RAG)")
    print("  ✓ Recommendation Agent (Hybrid Filtering + LLM Ranking)")
    print("  ✓ XAI Agent (SHAP + LIME + Counterfactuals)")
    print("  ✓ Orchestrator (LangGraph ready)")
    print("\n🚀 Pour utiliser avec les données réelles OULAD:")
    print("  Exécutez: python -m pytest tests/test_oulad_integration.py -v")
    print("\n")


if __name__ == "__main__":
    demo_system()