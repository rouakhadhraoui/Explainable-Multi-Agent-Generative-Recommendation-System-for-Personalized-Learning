# visualization/data_loader.py
"""
Data loader for visualization dashboard
Loads OULAD data and runs the multi-agent pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.blackboard import Blackboard
from orchestrator.orchestrator import Orchestrator
from data.oulad_loader import OULADLoader
from typing import Dict, List
import json


class DashboardDataLoader:
    """
    Loads and prepares data for the dashboard
    """
    
    def __init__(self, cache_file: str = "visualization/cached_results.json"):
        """
        Initialize the data loader
        
        Args:
            cache_file: Path to cache file for storing results
        """
        self.cache_file = cache_file
        self.blackboard = None
        self.orchestrator = None
        self.oulad_loader = None
        
    def initialize_system(self):
        """Initialize the multi-agent system"""
        print("🔄 Initializing multi-agent system...")
        self.blackboard = Blackboard()
        self.orchestrator = Orchestrator(self.blackboard)
        self.oulad_loader = OULADLoader()
        print("✅ System initialized")
        
    def load_sample_students(self, num_students: int = 5) -> List[str]:
        """
        Load sample students from OULAD
        
        Args:
            num_students: Number of students to load
            
        Returns:
            List of student IDs
        """
        if not self.oulad_loader:
            self.initialize_system()
            
        print(f"📚 Loading {num_students} sample students from OULAD...")
        
        # Load OULAD data
        data = self.oulad_loader.load_all_data()
        
        if data and 'studentInfo' in data:
            student_ids = data['studentInfo']['id_student'].unique()[:num_students].tolist()
            
            # Process each student through the pipeline
            for student_id in student_ids:
                student_id_str = f"oulad_{student_id}"
                
                # Get interactions
                interactions = self.oulad_loader.get_student_interactions(student_id)
                
                # Add to blackboard history
                for interaction in interactions:
                    self.blackboard.add_to_history(student_id_str, interaction)
                
                # Run pipeline
                try:
                    self.orchestrator.process_user_request(student_id_str, "full_analysis")
                    print(f"✅ Processed student {student_id}")
                except Exception as e:
                    print(f"⚠️  Error processing student {student_id}: {e}")
            
            return [f"oulad_{sid}" for sid in student_ids]
        
        return []
    
    def get_student_profile(self, student_id: str) -> Dict:
        """Get student profile from blackboard"""
        if not self.blackboard:
            self.initialize_system()
        return self.blackboard.read("profiles", student_id)
    
    def get_learning_path(self, student_id: str) -> Dict:
        """Get learning path from blackboard"""
        if not self.blackboard:
            self.initialize_system()
        return self.blackboard.read("learning_paths", student_id)
    
    def get_recommendations(self, student_id: str) -> Dict:
        """Get recommendations from blackboard"""
        if not self.blackboard:
            self.initialize_system()
        return self.blackboard.read("recommendations", student_id)
    
    def get_explanations(self, student_id: str) -> Dict:
        """Get XAI explanations from blackboard"""
        if not self.blackboard:
            self.initialize_system()
        return self.blackboard.read("explanations", student_id)
    
    def get_all_students(self) -> List[str]:
        """Get all processed student IDs"""
        if not self.blackboard:
            self.initialize_system()
        profiles = self.blackboard.read_section("profiles")
        return list(profiles.keys()) if profiles else []
