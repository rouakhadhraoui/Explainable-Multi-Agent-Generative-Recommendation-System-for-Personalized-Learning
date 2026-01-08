# tests/test_blackboard.py
"""
Tests pour valider le fonctionnement du Blackboard

Ce fichier teste toutes les fonctionnalités du Shared Memory
"""

import sys
sys.path.append('..')  # Pour importer depuis le dossier parent

from memory.blackboard import Blackboard


def test_basic_operations():
    """Test des opérations de base (read/write)"""
    print("\n" + "="*60)
    print("TEST 1 : Opérations de base (Read/Write)")
    print("="*60)
    
    # Créer un blackboard
    bb = Blackboard()
    
    # Test 1 : Écrire un profil utilisateur
    print("\n--- Test écriture profil ---")
    profile = {
        "user_id": "user_001",
        "name": "Alice",
        "learning_style": "visual",
        "level": "intermediate"
    }
    bb.write("profiles", "user_001", profile)
    
    # Test 2 : Lire le profil
    print("\n--- Test lecture profil ---")
    retrieved_profile = bb.read("profiles", "user_001")
    print(f"Profil récupéré: {retrieved_profile}")
    
    # Test 3 : Lire une clé inexistante
    print("\n--- Test lecture clé inexistante ---")
    result = bb.read("profiles", "user_999")
    
    # Validation
    assert retrieved_profile == profile, "❌ Le profil récupéré ne correspond pas"
    assert result is None, "❌ Une clé inexistante devrait retourner None"
    
    print("\n✅ TEST 1 RÉUSSI")


def test_history():
    """Test de l'historique des interactions"""
    print("\n" + "="*60)
    print("TEST 2 : Gestion de l'historique")
    print("="*60)
    
    bb = Blackboard()
    
    # Ajouter plusieurs interactions
    print("\n--- Ajout d'interactions ---")
    bb.add_to_history("user_001", {
        "type": "view",
        "resource_id": "course_python_101",
        "duration": 120
    })
    
    bb.add_to_history("user_001", {
        "type": "quiz",
        "resource_id": "quiz_python_basics",
        "score": 85
    })
    
    bb.add_to_history("user_001", {
        "type": "view",
        "resource_id": "course_python_102",
        "duration": 90
    })
    
    # Récupérer l'historique complet
    print("\n--- Récupération historique complet ---")
    full_history = bb.get_user_history("user_001")
    print(f"Nombre d'interactions: {len(full_history)}")
    for i, interaction in enumerate(full_history, 1):
        print(f"  {i}. {interaction['type']} - {interaction['resource_id']}")
    
    # Récupérer les 2 dernières interactions
    print("\n--- Récupération 2 dernières interactions ---")
    recent_history = bb.get_user_history("user_001", limit=2)
    print(f"Nombre: {len(recent_history)}")
    
    # Validation
    assert len(full_history) == 3, "❌ Devrait y avoir 3 interactions"
    assert len(recent_history) == 2, "❌ Devrait y avoir 2 interactions récentes"
    
    print("\n✅ TEST 2 RÉUSSI")


def test_sections():
    """Test de lecture complète des sections"""
    print("\n" + "="*60)
    print("TEST 3 : Lecture de sections complètes")
    print("="*60)
    
    bb = Blackboard()
    
    # Ajouter plusieurs profils
    bb.write("profiles", "user_001", {"name": "Alice", "level": "beginner"})
    bb.write("profiles", "user_002", {"name": "Bob", "level": "advanced"})
    bb.write("profiles", "user_003", {"name": "Charlie", "level": "intermediate"})
    
    # Lire toute la section
    print("\n--- Lecture section 'profiles' ---")
    all_profiles = bb.read_section("profiles")
    print(f"Nombre de profils: {len(all_profiles)}")
    for user_id, profile in all_profiles.items():
        print(f"  - {user_id}: {profile['name']} ({profile['level']})")
    
    # Validation
    assert len(all_profiles) == 3, "❌ Devrait y avoir 3 profils"
    
    print("\n✅ TEST 3 RÉUSSI")


def test_delete():
    """Test de suppression"""
    print("\n" + "="*60)
    print("TEST 4 : Suppression de données")
    print("="*60)
    
    bb = Blackboard()
    
    # Ajouter et supprimer
    bb.write("profiles", "user_temp", {"name": "Temp"})
    print("\n--- Avant suppression ---")
    print(f"Profil existe: {bb.read('profiles', 'user_temp') is not None}")
    
    bb.delete("profiles", "user_temp")
    print("\n--- Après suppression ---")
    print(f"Profil existe: {bb.read('profiles', 'user_temp') is not None}")
    
    # Validation
    assert bb.read("profiles", "user_temp") is None, "❌ Le profil devrait être supprimé"
    
    print("\n✅ TEST 4 RÉUSSI")


def test_stats():
    """Test des statistiques"""
    print("\n" + "="*60)
    print("TEST 5 : Statistiques du Blackboard")
    print("="*60)
    
    bb = Blackboard()
    
    # Ajouter des données
    bb.write("profiles", "user_001", {"name": "Alice"})
    bb.write("profiles", "user_002", {"name": "Bob"})
    bb.write("cached_content", "content_001", {"title": "Python Intro"})
    bb.add_to_history("user_001", {"type": "view"})
    bb.add_to_history("user_001", {"type": "quiz"})
    
    # Obtenir les stats
    print("\n--- Statistiques ---")
    stats = bb.get_stats()
    for section, count in stats.items():
        print(f"  {section:20s}: {count}")
    
    # Afficher la représentation
    print(f"\n{bb}")
    
    print("\n✅ TEST 5 RÉUSSI")


def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "#"*60)
    print("# SUITE DE TESTS COMPLÈTE DU BLACKBOARD")
    print("#"*60)
    
    try:
        test_basic_operations()
        test_history()
        test_sections()
        test_delete()
        test_stats()
        
        print("\n" + "="*60)
        print("🎉 TOUS LES TESTS SONT RÉUSSIS !")
        print("="*60)
        print("\nLe Blackboard fonctionne correctement.")
        print("Tu peux passer à l'étape suivante.\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")


if __name__ == "__main__":
    run_all_tests()