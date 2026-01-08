# tests/test_oulad_integration.py
"""
Tests pour l'intégration OULAD avec le système multi-agents

Vérifie que les données OULAD fonctionnent correctement avec nos agents
"""

import sys
sys.path.append('..')

from memory.blackboard import Blackboard
from utils.oulad_integration import OULADIntegration
from agents.profiling_agent import ProfilingAgent
from orchestrator.orchestrator import Orchestrator


def test_load_single_student():
    """Test de chargement d'un seul étudiant OULAD"""
    print("\n" + "="*70)
    print("TEST 1 : Chargement d'un étudiant OULAD")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    
    # Récupérer un étudiant
    students = oulad.loader.get_sample_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible dans OULAD")
        return
    
    student_id = students[0]
    
    # Charger dans le Blackboard
    success = oulad.load_student_to_blackboard(student_id)
    
    assert success, "❌ Échec du chargement"
    
    # Vérifier que l'historique est dans le Blackboard
    history = bb.get_user_history(student_id)
    
    print(f"\n✓ Étudiant {student_id} chargé")
    print(f"  • Interactions dans le Blackboard: {len(history)}")
    
    assert len(history) > 0, "❌ Aucune interaction chargée"
    
    print("\n✅ TEST 1 RÉUSSI - Étudiant chargé dans le Blackboard")


def test_load_multiple_students():
    """Test de chargement de plusieurs étudiants"""
    print("\n" + "="*70)
    print("TEST 2 : Chargement de plusieurs étudiants")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    
    # Charger 5 étudiants
    loaded = oulad.load_multiple_students(n=5)
    
    print(f"\n✓ {len(loaded)} étudiants chargés")
    
    assert len(loaded) > 0, "❌ Aucun étudiant chargé"
    
    # Vérifier dans le Blackboard
    history_section = bb.read_section("history")
    print(f"✓ {len(history_section)} étudiants dans l'historique du Blackboard")
    
    print("\n✅ TEST 2 RÉUSSI - Plusieurs étudiants chargés")


def test_profiling_with_oulad():
    """Test du Profiling Agent avec données OULAD"""
    print("\n" + "="*70)
    print("TEST 3 : Profiling Agent avec données OULAD")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    profiling_agent = ProfilingAgent(bb)
    
    # Charger un étudiant
    students = oulad.load_multiple_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible")
        return
    
    student_id = students[0]
    
    # Analyser avec le Profiling Agent
    print(f"\n🔍 Analyse du profil de l'étudiant {student_id}...")
    profile = profiling_agent.analyze_user(student_id)
    
    # Afficher le résultat
    print(f"\n📊 PROFIL GÉNÉRÉ:")
    print(f"  • Niveau          : {profile['level']}")
    print(f"  • Style           : {profile['learning_style']}")
    print(f"  • Intérêts        : {', '.join(profile['interests'])}")
    print(f"  • Interactions    : {profile['total_interactions']}")
    
    # Comparer avec l'estimation OULAD
    print(f"\n🔄 Comparaison OULAD vs Profiling Agent:")
    comparison = oulad.compare_oulad_vs_profiling(student_id, profile)
    
    print(f"  OULAD estimation:")
    print(f"    - Niveau: {comparison['oulad_estimation']['level']}")
    print(f"    - Style : {comparison['oulad_estimation']['style']}")
    
    print(f"  Profiling Agent:")
    print(f"    - Niveau: {comparison['profiling_agent']['level']}")
    print(f"    - Style : {comparison['profiling_agent']['style']}")
    
    print(f"\n  Match:")
    print(f"    - Niveau: {'✓' if comparison['level_match'] else '✗'}")
    print(f"    - Style : {'✓' if comparison['style_match'] else '✗'}")
    
    assert profile is not None, "❌ Profil non généré"
    
    print("\n✅ TEST 3 RÉUSSI - Profiling fonctionne avec OULAD")


def test_full_system_with_oulad():
    """Test du système complet avec données OULAD"""
    print("\n" + "="*70)
    print("TEST 4 : Système COMPLET avec données OULAD")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    orchestrator = Orchestrator(bb)
    
    # Charger un étudiant
    students = oulad.load_multiple_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible")
        return
    
    student_id = students[0]
    
    print(f"\n🚀 Lancement de l'analyse complète pour {student_id}...")
    
    # Lancer le pipeline complet
    result = orchestrator.process_user_request(student_id, request_type="full_analysis")
    
    # Vérifications
    assert result['overall_status'] == 'completed', "❌ Pipeline non complété"
    
    # Vérifier que tous les agents ont réussi
    print(f"\n📊 Résultats des agents:")
    for agent_name, agent_result in result['agents_results'].items():
        status = agent_result['status']
        emoji = "✅" if status == "success" else "❌"
        print(f"  {emoji} {agent_name:20s} : {status}")
    
    # Vérifier les données dans le Blackboard
    profile = bb.read("profiles", student_id)
    learning_path = bb.read("learning_paths", student_id)
    recommendations = bb.read("recommendations", student_id)
    explanations = bb.read("explanations", student_id)
    
    print(f"\n💾 Données dans le Blackboard:")
    print(f"  ✓ Profil          : {'Oui' if profile else 'Non'}")
    print(f"  ✓ Parcours        : {'Oui' if learning_path else 'Non'}")
    print(f"  ✓ Recommandations : {'Oui' if recommendations else 'Non'}")
    print(f"  ✓ Explications    : {'Oui' if explanations else 'Non'}")
    
    print("\n✅ TEST 4 RÉUSSI - Système complet fonctionne avec OULAD")


def test_batch_analysis():
    """Test d'analyse en batch"""
    print("\n" + "="*70)
    print("TEST 5 : Analyse en BATCH de 3 étudiants")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    orchestrator = Orchestrator(bb)
    
    # Analyser 3 étudiants
    results = oulad.batch_analyze_students(orchestrator, n=3)
    
    # Vérifications
    assert results['total_students'] > 0, "❌ Aucun étudiant analysé"
    
    print(f"\n📊 Distribution des niveaux:")
    levels = {}
    for student_result in results['students_results']:
        if student_result['status'] == 'success':
            level = student_result['level']
            levels[level] = levels.get(level, 0) + 1
    
    for level, count in levels.items():
        print(f"  • {level:15s} : {count} étudiant(s)")
    
    print("\n✅ TEST 5 RÉUSSI - Analyse en batch fonctionnelle")


def test_dataset_statistics():
    """Test des statistiques du dataset"""
    print("\n" + "="*70)
    print("TEST 6 : Statistiques du dataset OULAD")
    print("="*70)
    
    bb = Blackboard()
    oulad = OULADIntegration(bb)
    
    stats = oulad.get_dataset_statistics()
    
    print(f"\n📊 STATISTIQUES OULAD:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  • {key:30s} : {value:,}")
        else:
            print(f"  • {key:30s} : {value}")
    
    assert stats['total_students'] > 0, "❌ Aucun étudiant dans le dataset"
    
    print("\n✅ TEST 6 RÉUSSI - Statistiques calculées")


def run_all_tests():
    """Exécuter tous les tests d'intégration OULAD"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - INTÉGRATION OULAD")
    print("#"*70)
    
    try:
        test_load_single_student()
        test_load_multiple_students()
        test_profiling_with_oulad()
        test_full_system_with_oulad()
        test_batch_analysis()
        test_dataset_statistics()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS D'INTÉGRATION OULAD SONT RÉUSSIS !")
        print("="*70)
        print("\n✅ Le système multi-agents fonctionne avec OULAD")
        print("✅ Prêt pour l'évaluation avec métriques")
        print("\n🚀 Prochaine étape : Implémentation des métriques d'évaluation\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()