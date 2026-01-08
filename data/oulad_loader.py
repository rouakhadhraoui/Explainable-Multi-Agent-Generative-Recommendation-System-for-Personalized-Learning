# data/oulad_loader.py
"""
OULAD Data Loader - Chargement et preprocessing du dataset OULAD

OULAD (Open University Learning Analytics Dataset) contient :
- studentInfo.csv : informations démographiques des étudiants
- studentAssessment.csv : résultats aux évaluations
- studentRegistration.csv : inscriptions aux cours
- assessments.csv : métadonnées des évaluations
- courses.csv : informations sur les cours
- vle.csv : interactions avec la plateforme (Virtual Learning Environment)

Ce module charge et transforme ces données en format exploitable par nos agents.
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class OULADLoader:
    """
    Charge et préprocesse les données OULAD
    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialise le loader OULAD
        
        Args:
            data_path: Chemin vers le dossier contenant les fichiers CSV
        """
        self.data_path = data_path
        
        # Dictionnaire pour stocker les dataframes
        self.dataframes = {}
        
        print(f"✓ OULAD Loader initialisé")
        print(f"  Chemin des données: {data_path}")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Charge tous les fichiers CSV de OULAD
        
        Returns:
            Dictionnaire de DataFrames {nom_fichier: dataframe}
        """
        print(f"\n{'='*70}")
        print(f"📊 CHARGEMENT DES DONNÉES OULAD")
        print(f"{'='*70}")
        
        # Liste des fichiers à charger
        files = [
            "studentInfo.csv",
            "studentAssessment.csv", 
            "studentRegistration.csv",
            "assessments.csv",
            "courses.csv",
            "vle.csv"
        ]
        
        for filename in files:
            filepath = os.path.join(self.data_path, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️  Fichier manquant: {filename}")
                continue
            
            try:
                # Charger le CSV
                df = pd.read_csv(filepath)
                
                # Stocker dans le dictionnaire (sans l'extension .csv)
                name = filename.replace('.csv', '')
                self.dataframes[name] = df
                
                print(f"✓ {filename:30s} : {len(df):,} lignes, {len(df.columns)} colonnes")
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {filename}: {e}")
        
        print(f"\n✅ {len(self.dataframes)} fichiers chargés avec succès")
        
        return self.dataframes
    
    def get_student_profile(self, student_id: str) -> Optional[Dict]:
        """
        Récupère le profil complet d'un étudiant
        
        Args:
            student_id: ID de l'étudiant (code_module + code_presentation + id_student)
        
        Returns:
            Dictionnaire contenant le profil de l'étudiant
        """
        if 'studentInfo' not in self.dataframes:
            print("⚠️  Données studentInfo non chargées")
            return None
        
        # Rechercher l'étudiant
        student_data = self.dataframes['studentInfo'][
            self.dataframes['studentInfo']['id_student'] == int(student_id)
        ]
        
        if student_data.empty:
            return None
        
        # Prendre la première ligne (il peut y avoir plusieurs cours)
        student = student_data.iloc[0]
        
        profile = {
            "student_id": str(student['id_student']),
            "gender": student.get('gender', 'Unknown'),
            "region": student.get('region', 'Unknown'),
            "highest_education": student.get('highest_education', 'Unknown'),
            "age_band": student.get('age_band', 'Unknown'),
            "num_of_prev_attempts": int(student.get('num_of_prev_attempts', 0)),
            "studied_credits": int(student.get('studied_credits', 0)),
            "disability": student.get('disability', 'N')
        }
        
        return profile
    
    def get_student_interactions(self, student_id: str, 
                                code_module: str = None,
                                code_presentation: str = None) -> List[Dict]:
        """
        Récupère toutes les interactions d'un étudiant
        
        Args:
            student_id: ID de l'étudiant
            code_module: Code du module (optionnel)
            code_presentation: Code de présentation (optionnel)
        
        Returns:
            Liste d'interactions formatées pour le système
        """
        interactions = []
        
        # 1. Récupérer les résultats aux évaluations
        if 'studentAssessment' in self.dataframes:
            assessments = self.dataframes['studentAssessment'][
                self.dataframes['studentAssessment']['id_student'] == int(student_id)
            ]
            
            for _, assessment in assessments.iterrows():
                interactions.append({
                    "type": "quiz",
                    "resource_id": f"assessment_{assessment['id_assessment']}",
                    "score": float(assessment.get('score', 0)),
                    "date_submitted": assessment.get('date_submitted', None),
                    "is_banked": assessment.get('is_banked', 0)
                })
        
        # 2. Récupérer les interactions VLE (Virtual Learning Environment)
        if 'vle' in self.dataframes and 'studentVle' in self.dataframes:
            # Note: studentVle n'est pas toujours présent dans certaines versions
            # On simule des interactions view basées sur les assessments
            pass
        
        # Trier par date si disponible
        interactions.sort(key=lambda x: x.get('date_submitted', 0) if x.get('date_submitted') else 0)
        
        return interactions
    
    def get_learning_style_heuristic(self, student_id: str) -> str:
        """
        Détermine le style d'apprentissage basé sur les données OULAD
        (Heuristique simplifiée)
        
        Args:
            student_id: ID de l'étudiant
        
        Returns:
            Style d'apprentissage estimé
        """
        interactions = self.get_student_interactions(student_id)
        
        if not interactions:
            return "visual"  # Par défaut
        
        # Heuristique simple : 
        # - Beaucoup d'évaluations → kinesthetic (pratique)
        # - Peu d'évaluations → reading (théorie)
        num_assessments = len(interactions)
        
        if num_assessments > 10:
            return "kinesthetic"
        elif num_assessments > 5:
            return "visual"
        else:
            return "reading"
    
    def get_student_level(self, student_id: str) -> str:
        """
        Détermine le niveau de l'étudiant basé sur ses performances
        
        Args:
            student_id: ID de l'étudiant
        
        Returns:
            Niveau estimé (beginner, intermediate, advanced)
        """
        interactions = self.get_student_interactions(student_id)
        
        if not interactions:
            return "beginner"
        
        # Calculer le score moyen
        scores = [i['score'] for i in interactions if 'score' in i]
        
        if not scores:
            return "beginner"
        
        avg_score = sum(scores) / len(scores)
        
        # Classification simple
        if avg_score < 60:
            return "beginner"
        elif avg_score < 80:
            return "intermediate"
        else:
            return "advanced"
    
    def convert_to_system_format(self, student_id: str) -> Dict:
        """
        Convertit les données OULAD au format attendu par notre système
        
        Args:
            student_id: ID de l'étudiant
        
        Returns:
            Dictionnaire prêt à être utilisé par le système
        """
        profile = self.get_student_profile(student_id)
        
        if not profile:
            return {"error": f"Student {student_id} not found"}
        
        interactions = self.get_student_interactions(student_id)
        learning_style = self.get_learning_style_heuristic(student_id)
        level = self.get_student_level(student_id)
        
        return {
            "student_id": student_id,
            "profile": profile,
            "interactions": interactions,
            "estimated_style": learning_style,
            "estimated_level": level,
            "total_interactions": len(interactions)
        }
    
    def get_sample_students(self, n: int = 10) -> List[str]:
        """
        Récupère un échantillon d'IDs d'étudiants
        
        Args:
            n: Nombre d'étudiants à récupérer
        
        Returns:
            Liste d'IDs d'étudiants
        """
        if 'studentInfo' not in self.dataframes:
            return []
        
        # Prendre les n premiers étudiants uniques
        students = self.dataframes['studentInfo']['id_student'].unique()[:n]
        
        return [str(s) for s in students]
    
    def get_statistics(self) -> Dict:
        """
        Obtient des statistiques sur le dataset OULAD
        
        Returns:
            Dictionnaire de statistiques
        """
        stats = {
            "total_files_loaded": len(self.dataframes)
        }
        
        if 'studentInfo' in self.dataframes:
            stats['total_students'] = self.dataframes['studentInfo']['id_student'].nunique()
            stats['total_registrations'] = len(self.dataframes['studentInfo'])
        
        if 'studentAssessment' in self.dataframes:
            stats['total_assessments'] = len(self.dataframes['studentAssessment'])
            stats['avg_score'] = self.dataframes['studentAssessment']['score'].mean()
        
        if 'courses' in self.dataframes:
            stats['total_courses'] = len(self.dataframes['courses'])
        
        if 'assessments' in self.dataframes:
            stats['total_assessment_types'] = len(self.dataframes['assessments'])
        
        return stats
    
    def export_processed_data(self, output_path: str = "data/processed"):
        """
        Exporte les données préprocessées
        
        Args:
            output_path: Dossier de sortie
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Exporter un échantillon d'étudiants convertis
        sample_students = self.get_sample_students(100)
        
        processed_data = []
        for student_id in sample_students:
            data = self.convert_to_system_format(student_id)
            if "error" not in data:
                processed_data.append(data)
        
        # Sauvegarder en JSON
        import json
        output_file = os.path.join(output_path, "oulad_processed_students.json")
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"✓ {len(processed_data)} étudiants exportés vers {output_file}")
    
    def __repr__(self) -> str:
        """Représentation textuelle du loader"""
        return f"OULADLoader(files_loaded={len(self.dataframes)}, path='{self.data_path}')"