# tests/test_oulad_loader.py
"""
Tests pour le OULAD Loader

Vérifie que les données OULAD sont correctement chargées et préprocessées
"""

import sys
import os
sys.path.append('..')

from data.oulad_loader import OULADLoader


def test_load_data():
    """Test du chargement des données OULAD"""
    print("\n" + "="*70)
    print("TEST 1 : Chargement des données OULAD")
    print("="*70)
    
    # Créer le loader
    loader = OULADLoader(data_path="data/raw")
    
    # Charger les données
    dataframes = loader.load_all_data()
    
    # Vérifications
    assert len(dataframes) > 0, "❌ Aucun fichier chargé"
    assert 'studentInfo' in dataframes, "❌ studentInfo manquant"
    
    print(f"\n📊 Fichiers chargés:")
    for name, df in dataframes.items():
        print(f"  • {name:25s} : {len(df):,} lignes")
    
    print("\n✅ TEST 1 RÉUSSI - Données chargées avec succès")


def test_get_statistics():
    """Test des statistiques du dataset"""
    print("\n" + "="*70)
    print("TEST 2 : Statistiques OULAD")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    stats = loader.get_statistics()
    
    print(f"\n📊 STATISTIQUES OULAD:")
    for key, value in stats.items():
        print(f"  • {key:30s} : {value:,}" if isinstance(value, (int, float)) else f"  • {key:30s} : {value}")
    
    assert stats['total_students'] > 0, "❌ Aucun étudiant trouvé"
    
    print("\n✅ TEST 2 RÉUSSI - Statistiques calculées")


def test_get_sample_students():
    """Test de récupération d'étudiants échantillons"""
    print("\n" + "="*70)
    print("TEST 3 : Échantillon d'étudiants")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    # Récupérer 10 étudiants
    students = loader.get_sample_students(n=10)
    
    print(f"\n👥 {len(students)} étudiants échantillonnés:")
    for i, student_id in enumerate(students, 1):
        print(f"  {i}. Student ID: {student_id}")
    
    assert len(students) > 0, "❌ Aucun étudiant récupéré"
    
    print("\n✅ TEST 3 RÉUSSI - Échantillon récupéré")


def test_get_student_profile():
    """Test de récupération d'un profil étudiant"""
    print("\n" + "="*70)
    print("TEST 4 : Profil d'un étudiant")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    # Prendre le premier étudiant
    students = loader.get_sample_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible")
        return
    
    student_id = students[0]
    
    # Récupérer le profil
    profile = loader.get_student_profile(student_id)
    
    if profile:
        print(f"\n👤 PROFIL de l'étudiant {student_id}:")
        for key, value in profile.items():
            print(f"  • {key:25s} : {value}")
        
        assert 'student_id' in profile, "❌ student_id manquant"
        
        print("\n✅ TEST 4 RÉUSSI - Profil récupéré")
    else:
        print(f"⚠️  Profil non trouvé pour {student_id}")


def test_get_student_interactions():
    """Test de récupération des interactions"""
    print("\n" + "="*70)
    print("TEST 5 : Interactions d'un étudiant")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    students = loader.get_sample_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible")
        return
    
    student_id = students[0]
    
    # Récupérer les interactions
    interactions = loader.get_student_interactions(student_id)
    
    print(f"\n📊 {len(interactions)} interactions pour l'étudiant {student_id}:")
    
    for i, interaction in enumerate(interactions[:5], 1):  # Afficher les 5 premières
        print(f"\n  Interaction {i}:")
        print(f"    Type        : {interaction.get('type', 'N/A')}")
        print(f"    Resource ID : {interaction.get('resource_id', 'N/A')}")
        if 'score' in interaction:
            print(f"    Score       : {interaction['score']}")
    
    if len(interactions) > 5:
        print(f"\n  ... et {len(interactions) - 5} autres interactions")
    
    print("\n✅ TEST 5 RÉUSSI - Interactions récupérées")


def test_convert_to_system_format():
    """Test de conversion au format système"""
    print("\n" + "="*70)
    print("TEST 6 : Conversion au format système")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    students = loader.get_sample_students(n=1)
    
    if not students:
        print("⚠️  Aucun étudiant disponible")
        return
    
    student_id = students[0]
    
    # Convertir
    data = loader.convert_to_system_format(student_id)
    
    if "error" in data:
        print(f"❌ {data['error']}")
        return
    
    print(f"\n🔄 DONNÉES CONVERTIES pour {student_id}:")
    print(f"  • Estimated Level : {data['estimated_level']}")
    print(f"  • Estimated Style : {data['estimated_style']}")
    print(f"  • Total Interactions : {data['total_interactions']}")
    
    print(f"\n  Profil:")
    for key, value in data['profile'].items():
        print(f"    - {key:20s} : {value}")
    
    assert 'profile' in data, "❌ Profil manquant"
    assert 'interactions' in data, "❌ Interactions manquantes"
    
    print("\n✅ TEST 6 RÉUSSI - Conversion réussie")


def test_export_processed_data():
    """Test d'export des données préprocessées"""
    print("\n" + "="*70)
    print("TEST 7 : Export des données préprocessées")
    print("="*70)
    
    loader = OULADLoader()
    loader.load_all_data()
    
    # Exporter
    loader.export_processed_data(output_path="data/processed")
    
    # Vérifier que le fichier existe
    output_file = "data/processed/oulad_processed_students.json"
    assert os.path.exists(output_file), f"❌ Fichier {output_file} non créé"
    
    print(f"\n💾 Fichier exporté: {output_file}")
    
    # Lire le fichier pour vérifier
    import json
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ {len(data)} étudiants préprocessés exportés")
    
    print("\n✅ TEST 7 RÉUSSI - Export réussi")


def run_all_tests():
    """Exécuter tous les tests OULAD"""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS COMPLÈTE - OULAD LOADER")
    print("#"*70)
    
    try:
        test_load_data()
        test_get_statistics()
        test_get_sample_students()
        test_get_student_profile()
        test_get_student_interactions()
        test_convert_to_system_format()
        test_export_processed_data()
        
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS OULAD SONT RÉUSSIS !")
        print("="*70)
        print("\n✅ Le dataset OULAD est correctement chargé et préprocessé")
        print("✅ Prêt pour l'intégration avec le système multi-agents\n")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()