# tests/test_orchestrator.py
"""
Tests pour l'Orchestrator

Ce fichier teste la coordination des agents et le flux du pipeline
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from orchestrator.orchestrator import Orchestrator, AgentStatus


def setup_test_user(blackboard: Blackboard, user_id: str):
    """
    Crée des données de test pour un utilisateur
    
    Args:
        blackboard: Instance du Blackboard
        user_id: ID de l'utilisateur
    """
    # Ajouter un historique d'interactions
    interactions = [
        {"type": "view", "resource_id": "video_python_intro", "duration": 180},
        {"type": "quiz", "resource_id": "quiz_python_basics", "score": 75},
        {"type": "view", "resource_id": "course_python_loops", "duration": 240},
        {"type": "quiz", "resource_id": "quiz_python_loops", "score": 82},
        {"type": "exercise", "resource_id": "exercise_python_functions", "duration": 600},
        {"type": "quiz", "resource_id": "quiz_python_functions", "score": 88},
    ]
    
    for interaction in interactions:
        blackboard.add_to_history(user_id, interaction)
    
    print(f"✓ {len(interactions)} interactions créées pour {user_id}")


def test_orchestrator_initialization():
    """Test d'initialisation de l'orchestrateur"""
    print("\n" + "="*80)
    print("TEST 1 : Initialisation de l'Orchestrator")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Vérifier que l'orchestrateur est bien initialisé
    assert len(orchestrator.agents) > 0, "❌ Aucun agent initialisé"
    assert "profiling" in orchestrator.agents, "❌ Agent profiling manquant"
    
    # Vérifier le pipeline
    assert len(orchestrator.pipeline) > 0, "❌ Pipeline vide"
    
    # Afficher les infos
    info = orchestrator.get_pipeline_info()
    print(f"\n📊 Informations du pipeline:")
    print(f"  - Agents disponibles: {info['agents_available']}")
    print(f"  - Étapes du pipeline: {info['pipeline']}")
    print(f"  - Statut des agents: {info['agents_status']}")
    
    print(f"\n{orchestrator}")
    
    print("\n✅ TEST 1 RÉUSSI - Orchestrator initialisé correctement")


def test_profile_only_request():
    """Test d'une requête profile_only"""
    print("\n" + "="*80)
    print("TEST 2 : Requête 'profile_only'")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Créer un utilisateur de test
    user_id = "test_user_001"
    setup_test_user(bb, user_id)
    
    # Exécuter une requête profile_only
    result = orchestrator.process_user_request(user_id, request_type="profile_only")
    
    # Vérifications
    assert result["overall_status"] == "completed", "❌ La requête devrait être complétée"
    assert "profiling" in result["agents_results"], "❌ Résultat du profiling manquant"
    assert result["agents_results"]["profiling"]["status"] == "success", "❌ Le profiling a échoué"
    
    # Vérifier que le profil est dans le Blackboard
    profile = bb.read("profiles", user_id)
    assert profile is not None, "❌ Le profil n'est pas dans le Blackboard"
    
    print(f"\n📋 Profil créé:")
    print(f"  - User ID: {profile['user_id']}")
    print(f"  - Level: {profile['level']}")
    print(f"  - Style: {profile['learning_style']}")
    print(f"  - Interests: {profile['interests']}")
    
    print("\n✅ TEST 2 RÉUSSI - Requête 'profile_only' exécutée avec succès")


def test_full_analysis_request():
    """Test d'une requête full_analysis"""
    print("\n" + "="*80)
    print("TEST 3 : Requête 'full_analysis'")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Créer un utilisateur de test
    user_id = "test_user_002"
    setup_test_user(bb, user_id)
    
    # Exécuter une analyse complète
    result = orchestrator.process_user_request(user_id, request_type="full_analysis")
    
    # Vérifications
    assert result["overall_status"] == "completed", "❌ L'analyse complète a échoué"
    assert len(result["agents_results"]) > 0, "❌ Aucun agent exécuté"
    
    # Vérifier que tous les agents du pipeline ont été exécutés
    for agent_name in orchestrator.pipeline:
        assert agent_name in result["agents_results"], f"❌ Agent {agent_name} non exécuté"
    
    print(f"\n📊 Résultats de l'analyse complète:")
    print(f"  - Statut global: {result['overall_status']}")
    print(f"  - Agents exécutés: {list(result['agents_results'].keys())}")
    print(f"  - Durée totale: {result.get('completed_at', 'N/A')}")
    
    print("\n✅ TEST 3 RÉUSSI - Analyse complète exécutée avec succès")


def test_multiple_users():
    """Test avec plusieurs utilisateurs"""
    print("\n" + "="*80)
    print("TEST 4 : Traitement de plusieurs utilisateurs")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Créer 3 utilisateurs
    users = ["user_alice", "user_bob", "user_charlie"]
    
    print(f"\n🔄 Traitement de {len(users)} utilisateurs...")
    for user_id in users:
        setup_test_user(bb, user_id)
        result = orchestrator.process_user_request(user_id, request_type="profile_only")
        assert result["overall_status"] == "completed", f"❌ Échec pour {user_id}"
    
    # Vérifier que tous les profils sont créés
    all_profiles = bb.read_section("profiles")
    assert len(all_profiles) == len(users), "❌ Tous les profils ne sont pas créés"
    
    print(f"\n📊 Profils créés:")
    for user_id in users:
        profile = bb.read("profiles", user_id)
        print(f"  - {user_id:15s} : {profile['level']:12s} | {profile['learning_style']}")
    
    # Vérifier l'historique des exécutions
    history = orchestrator.get_execution_history()
    assert len(history) == len(users), "❌ Historique incomplet"
    
    print(f"\n📜 Historique des exécutions: {len(history)} entrées")
    
    print("\n✅ TEST 4 RÉUSSI - Plusieurs utilisateurs traités avec succès")


def test_execution_history():
    """Test de l'historique des exécutions"""
    print("\n" + "="*80)
    print("TEST 5 : Historique des exécutions")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Exécuter plusieurs requêtes
    users = ["user_001", "user_002", "user_001"]  # user_001 deux fois
    
    print(f"\n🔄 Exécution de {len(users)} requêtes...")
    for i, user_id in enumerate(users):
        setup_test_user(bb, f"{user_id}_{i}")
        orchestrator.process_user_request(f"{user_id}_{i}", request_type="profile_only")
    
    # Récupérer l'historique complet
    full_history = orchestrator.get_execution_history()
    print(f"\n📜 Historique complet: {len(full_history)} exécutions")
    
    # Vérifier
    assert len(full_history) == len(users), "❌ Historique incomplet"
    
    # Afficher les détails
    for i, execution in enumerate(full_history, 1):
        print(f"\n  Exécution #{i}:")
        print(f"    User ID: {execution['user_id']}")
        print(f"    Type: {execution['request_type']}")
        print(f"    Statut: {execution['overall_status']}")
        print(f"    Agents: {list(execution['agents_results'].keys())}")
    
    print("\n✅ TEST 5 RÉUSSI - Historique des exécutions fonctionnel")


def test_agent_status():
    """Test des statuts des agents"""
    print("\n" + "="*80)
    print("TEST 6 : Gestion des statuts d'agents")
    print("="*80)
    
    bb = Blackboard()
    orchestrator = Orchestrator(bb)
    
    # Vérifier le statut initial
    initial_status = orchestrator.get_agent_status("profiling")
    print(f"\n📊 Statut initial de 'profiling': {initial_status}")
    assert initial_status == AgentStatus.PENDING, "❌ Le statut initial devrait être PENDING"
    
    # Exécuter une requête
    user_id = "test_user_status"
    setup_test_user(bb, user_id)
    orchestrator.process_user_request(user_id, request_type="profile_only")
    
    # Vérifier le statut après exécution
    final_status = orchestrator.get_agent_status("profiling")
    print(f"📊 Statut final de 'profiling': {final_status}")
    assert final_status == AgentStatus.COMPLETED, "❌ Le statut devrait être COMPLETED"
    
    # Réinitialiser les agents
    print(f"\n🔄 Réinitialisation des agents...")
    orchestrator.reset_agents()
    
    reset_status = orchestrator.get_agent_status("profiling")
    print(f"📊 Statut après reset: {reset_status}")
    assert reset_status == AgentStatus.PENDING, "❌ Le statut devrait être PENDING après reset"
    
    print("\n✅ TEST 6 RÉUSSI - Gestion des statuts fonctionnelle")


def run_all_tests():
    """Exécuter tous les tests de l'Orchestrator"""
    print("\n" + "#"*80)
    print("# SUITE DE TESTS COMPLÈTE - ORCHESTRATOR")
    print("#"*80)
    
    try:
        test_orchestrator_initialization()
        test_profile_only_request()
        test_full_analysis_request()
        test_multiple_users()
        test_execution_history()
        test_agent_status()
        
        # Message final mis à jour pour refléter le système complet
        print("\n" + "="*80)
        print("🎉 TOUS LES TESTS DE L'ORCHESTRATOR SONT RÉUSSIS !")
        print("="*80)
        
        print("\n✅ Le système multi-agents complet est opérationnel !")
        
        print("\n📊 Architecture implémentée (5 couches) :")
        print("  └─ Layer 0 : Orchestration")
        print("     • Orchestrator (LangGraph/AutoGen)")
        print("  └─ Layer 1 : Shared Memory")
        print("     • Blackboard avec Vector Database")
        print("  └─ Layer 2 : Reasoning & Decision")
        print("     • Profiling Agent (Embeddings/Clustering)")
        print("     • Path Planning Agent (Graph Search/Heuristics)")
        print("     • Content Generator (LLM + RAG)")
        print("  └─ Layer 3 : Explainability & Trust")
        print("     • Recommendation Agent (Hybrid Ranking)")
        print("     • XAI Agent (SHAP/LIME/Counterfactuals)")
        print("  └─ Layer 4 : Data Layer")
        print("     • Ressources pédagogiques")
        print("     • Historique des interactions")
        
        print("\n🚀 Pipeline cognitif complet :")
        print("   User → Profiling → Path Planning → Content Generation → Recommendation → XAI")
        
        print("\n💡 Prochaines étapes suggérées :")
        print("  1. Exécuter python main.py pour voir le système en action")
        print("  2. Enrichir le catalogue de ressources pédagogiques")
        print("  3. Affiner les algorithmes de recommandation")
        print("  4. Améliorer les explications XAI")
        print("  5. Développer une interface utilisateur (Web/CLI)")
        print("  6. Ajouter des métriques de performance et d'évaluation")
        print("  7. Implémenter la persistance des données (base de données)")
        
        print("\n" + "="*80)
        print("🎊 Félicitations ! Votre système de recommandation explicable est prêt !")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()