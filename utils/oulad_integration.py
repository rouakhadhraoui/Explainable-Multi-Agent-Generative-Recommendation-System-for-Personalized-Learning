# utils/oulad_integration.py
"""
OULAD Integration - Intégration des données OULAD avec le système multi-agents

Ce module fait le pont entre les données OULAD et notre système :
- Charge les étudiants OULAD dans le Blackboard
- Convertit les données OULAD au format système
- Permet de tester le système avec de vraies données
"""

from typing import Dict, List, Optional
from memory.blackboard import Blackboard
from data.oulad_loader import OULADLoader


class OULADIntegration:
    """
    Intègre les données OULAD avec le système multi-agents
    """
    
    def __init__(self, blackboard: Blackboard, data_path: str = "data/raw"):
        """
        Initialise l'intégration OULAD
        
        Args:
            blackboard: Instance du Blackboard
            data_path: Chemin vers les données OULAD
        """
        self.blackboard = blackboard
        self.loader = OULADLoader(data_path=data_path)
        
        # Charger les données
        self.loader.load_all_data()
        
        print(f"✓ OULAD Integration initialisée")
    
    def load_student_to_blackboard(self, student_id: str) -> bool:
        """
        Charge un étudiant OULAD dans le Blackboard
        
        Args:
            student_id: ID de l'étudiant OULAD
        
        Returns:
            True si le chargement a réussi
        """
        # Convertir les données au format système
        student_data = self.loader.convert_to_system_format(student_id)
        
        if "error" in student_data:
            print(f"⚠️  {student_data['error']}")
            return False
        
        # Ajouter les interactions à l'historique du Blackboard
        for interaction in student_data['interactions']:
            self.blackboard.add_to_history(student_id, interaction)
        
        print(f"✓ Étudiant {student_id} chargé dans le Blackboard")
        print(f"  • {len(student_data['interactions'])} interactions ajoutées")
        
        return True
    
    def load_multiple_students(self, n: int = 10) -> List[str]:
        """
        Charge plusieurs étudiants dans le Blackboard
        
        Args:
            n: Nombre d'étudiants à charger
        
        Returns:
            Liste des IDs d'étudiants chargés
        """
        print(f"\n📥 Chargement de {n} étudiants OULAD dans le système...")
        
        # Récupérer un échantillon
        sample_students = self.loader.get_sample_students(n=n)
        
        loaded_students = []
        
        for i, student_id in enumerate(sample_students, 1):
            print(f"\n  [{i}/{len(sample_students)}] Chargement de l'étudiant {student_id}...")
            
            success = self.load_student_to_blackboard(student_id)
            
            if success:
                loaded_students.append(student_id)
        
        print(f"\n✅ {len(loaded_students)}/{n} étudiants chargés avec succès")
        
        return loaded_students
    
    def get_student_statistics(self, student_id: str) -> Dict:
        """
        Obtient des statistiques sur un étudiant OULAD
        
        Args:
            student_id: ID de l'étudiant
        
        Returns:
            Dictionnaire de statistiques
        """
        student_data = self.loader.convert_to_system_format(student_id)
        
        if "error" in student_data:
            return {}
        
        interactions = student_data['interactions']
        scores = [i['score'] for i in interactions if 'score' in i]
        
        stats = {
            "student_id": student_id,
            "total_interactions": len(interactions),
            "total_assessments": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "estimated_level": student_data['estimated_level'],
            "estimated_style": student_data['estimated_style']
        }
        
        return stats
    
    def compare_oulad_vs_profiling(self, student_id: str, 
                                   profiling_result: Dict) -> Dict:
        """
        Compare l'estimation OULAD avec le résultat du Profiling Agent
        
        Args:
            student_id: ID de l'étudiant
            profiling_result: Résultat du Profiling Agent
        
        Returns:
            Comparaison des résultats
        """
        oulad_data = self.loader.convert_to_system_format(student_id)
        
        if "error" in oulad_data:
            return {"error": "Student not found in OULAD"}
        
        comparison = {
            "student_id": student_id,
            "oulad_estimation": {
                "level": oulad_data['estimated_level'],
                "style": oulad_data['estimated_style']
            },
            "profiling_agent": {
                "level": profiling_result.get('level', 'N/A'),
                "style": profiling_result.get('learning_style', 'N/A')
            },
            "level_match": oulad_data['estimated_level'] == profiling_result.get('level'),
            "style_match": oulad_data['estimated_style'] == profiling_result.get('learning_style')
        }
        
        return comparison
    
    def get_dataset_statistics(self) -> Dict:
        """
        Obtient des statistiques globales sur le dataset OULAD
        
        Returns:
            Statistiques du dataset
        """
        return self.loader.get_statistics()
    
    def export_loaded_students(self, output_path: str = "data/processed/loaded_students.json"):
        """
        Exporte la liste des étudiants chargés dans le système
        
        Args:
            output_path: Chemin du fichier de sortie
        """
        import json
        import os
        
        # Récupérer tous les profils du Blackboard
        profiles = self.blackboard.read_section("profiles")
        
        export_data = {
            "total_students": len(profiles),
            "students": list(profiles.keys())
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ {len(profiles)} étudiants exportés vers {output_path}")
    
    def batch_analyze_students(self, orchestrator, n: int = 10) -> Dict:
        """
        Analyse un batch d'étudiants OULAD avec le système complet
        
        Args:
            orchestrator: Instance de l'Orchestrator
            n: Nombre d'étudiants à analyser
        
        Returns:
            Résultats agrégés des analyses
        """
        print(f"\n{'='*70}")
        print(f"🔬 ANALYSE EN BATCH DE {n} ÉTUDIANTS OULAD")
        print(f"{'='*70}")
        
        # Charger les étudiants
        students = self.load_multiple_students(n=n)
        
        results = {
            "total_students": len(students),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "students_results": []
        }
        
        # Analyser chaque étudiant
        for i, student_id in enumerate(students, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(students)}] Analyse de l'étudiant {student_id}")
            print(f"{'─'*70}")
            
            try:
                # Lancer l'analyse complète
                result = orchestrator.process_user_request(
                    student_id, 
                    request_type="full_analysis"
                )
                
                if result['overall_status'] == 'completed':
                    results['successful_analyses'] += 1
                    
                    # Récupérer les résultats
                    profile = self.blackboard.read("profiles", student_id)
                    
                    results['students_results'].append({
                        "student_id": student_id,
                        "status": "success",
                        "level": profile.get('level', 'N/A'),
                        "style": profile.get('learning_style', 'N/A')
                    })
                else:
                    results['failed_analyses'] += 1
                    results['students_results'].append({
                        "student_id": student_id,
                        "status": "failed"
                    })
                
            except Exception as e:
                print(f"❌ Erreur pour l'étudiant {student_id}: {e}")
                results['failed_analyses'] += 1
                results['students_results'].append({
                    "student_id": student_id,
                    "status": "error",
                    "error": str(e)
                })
        
        # Résumé
        print(f"\n{'='*70}")
        print(f"📊 RÉSUMÉ DE L'ANALYSE BATCH")
        print(f"{'='*70}")
        print(f"  • Total analysé   : {results['total_students']}")
        print(f"  • Succès          : {results['successful_analyses']}")
        print(f"  • Échecs          : {results['failed_analyses']}")
        print(f"  • Taux de succès  : {(results['successful_analyses']/results['total_students']*100):.1f}%")
        
        return results
    
    def __repr__(self) -> str:
        """Représentation textuelle"""
        stats = self.loader.get_statistics()
        return f"OULADIntegration(students={stats.get('total_students', 0)}, loaded={True})"