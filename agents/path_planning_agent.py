# agents/path_planning_agent.py
"""
Path Planning Agent - Planification de parcours d'apprentissage

Rôle :
- Analyser le profil utilisateur et définir des objectifs d'apprentissage
- Planifier un parcours optimal (séquence de ressources)
- Adapter le parcours selon le niveau et le style d'apprentissage
- Utiliser des heuristiques et du raisonnement LLM

Technologies : Graph Search, Heuristiques, LLM pour le raisonnement
"""

import ollama
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
import heapq
from collections import defaultdict

from memory.blackboard import Blackboard


class PathPlanningAgent:
    """
    Agent responsable de la planification des parcours d'apprentissage
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b"):
        """
        Initialise le Path Planning Agent
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Nom du modèle LLM local
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        
        # Base de connaissances : catalogue de ressources pédagogiques
        # Dans un vrai système, ceci viendrait d'une base de données
        self.resource_catalog = self._load_resource_catalog()
        
        # Q-Learning parameters for Reinforcement Learning
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q(state, action)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration rate
        
        print(f"✓ Path Planning Agent initialisé avec {len(self.resource_catalog)} ressources")
        print(f"✓ RL Q-Learning activé (α={self.learning_rate}, γ={self.discount_factor})")
    
    def _load_resource_catalog(self) -> List[Dict]:
        """
        Charge le catalogue de ressources disponibles
        
        Returns:
            Liste de ressources pédagogiques
        """
        # Catalogue fictif pour la démonstration
        # Dans un vrai système, ceci viendrait d'une base de données
        catalog = [
            # Python - Débutant
            {"id": "py_intro_video", "title": "Introduction to Python", "type": "video", 
             "level": "beginner", "topic": "python", "duration": 15, "prerequisites": []},
            {"id": "py_variables_video", "title": "Variables and Data Types", "type": "video",
             "level": "beginner", "topic": "python", "duration": 20, "prerequisites": ["py_intro_video"]},
            {"id": "py_basics_quiz", "title": "Python Basics Quiz", "type": "quiz",
             "level": "beginner", "topic": "python", "duration": 10, "prerequisites": ["py_variables_video"]},
            
            # Python - Intermédiaire
            {"id": "py_loops_exercise", "title": "Loops and Iterations", "type": "exercise",
             "level": "intermediate", "topic": "python", "duration": 30, "prerequisites": ["py_basics_quiz"]},
            {"id": "py_functions_course", "title": "Functions and Modules", "type": "course",
             "level": "intermediate", "topic": "python", "duration": 45, "prerequisites": ["py_loops_exercise"]},
            {"id": "py_lists_exercise", "title": "Lists and Data Structures", "type": "exercise",
             "level": "intermediate", "topic": "python", "duration": 35, "prerequisites": ["py_functions_course"]},
            {"id": "py_intermediate_quiz", "title": "Python Intermediate Quiz", "type": "quiz",
             "level": "intermediate", "topic": "python", "duration": 15, "prerequisites": ["py_lists_exercise"]},
            
            # Python - Avancé
            {"id": "py_oop_course", "title": "Object-Oriented Programming", "type": "course",
             "level": "advanced", "topic": "python", "duration": 60, "prerequisites": ["py_intermediate_quiz"]},
            {"id": "py_decorators_article", "title": "Decorators and Generators", "type": "article",
             "level": "advanced", "topic": "python", "duration": 40, "prerequisites": ["py_oop_course"]},
            {"id": "py_async_course", "title": "Asynchronous Programming", "type": "course",
             "level": "advanced", "topic": "python", "duration": 50, "prerequisites": ["py_decorators_article"]},
            {"id": "py_advanced_quiz", "title": "Python Advanced Quiz", "type": "quiz",
             "level": "advanced", "topic": "python", "duration": 20, "prerequisites": ["py_async_course"]},
            
            # Data Science
            {"id": "ds_intro_video", "title": "Introduction to Data Science", "type": "video",
             "level": "intermediate", "topic": "datascience", "duration": 25, "prerequisites": ["py_intermediate_quiz"]},
            {"id": "ds_pandas_course", "title": "Pandas for Data Analysis", "type": "course",
             "level": "intermediate", "topic": "datascience", "duration": 55, "prerequisites": ["ds_intro_video"]},
            {"id": "ds_visualization_exercise", "title": "Data Visualization", "type": "exercise",
             "level": "intermediate", "topic": "datascience", "duration": 40, "prerequisites": ["ds_pandas_course"]},
        ]
        
        return catalog
    
    def plan_learning_path(self, user_id: str, target_level: Optional[str] = None) -> Dict:
        """
        Planifie un parcours d'apprentissage personnalisé pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            target_level: Niveau cible optionnel ("intermediate", "advanced")
                         Si None, déterminé automatiquement
        
        Returns:
            Parcours d'apprentissage planifié (Dict)
        """
        print(f"\n{'='*70}")
        print(f"🗺️  PLANIFICATION DU PARCOURS D'APPRENTISSAGE")
        print(f"{'='*70}")
        print(f"User ID: {user_id}")
        
        # Étape 1 : Récupérer le profil utilisateur
        profile = self.blackboard.read("profiles", user_id)
        
        if not profile:
            print(f"⚠️  Profil non trouvé pour {user_id}. Impossible de planifier.")
            return {"error": "Profile not found"}
        
        print(f"✓ Profil récupéré:")
        print(f"  - Niveau actuel: {profile['level']}")
        print(f"  - Style: {profile['learning_style']}")
        print(f"  - Intérêts: {', '.join(profile['interests'])}")
        
        # Étape 2 : Déterminer le niveau cible
        if not target_level:
            target_level = self._determine_target_level(profile['level'])
        print(f"✓ Niveau cible: {target_level}")
        
        # Étape 3 : Identifier les ressources déjà complétées
        completed_resources = self._get_completed_resources(user_id)
        print(f"✓ Ressources déjà complétées: {len(completed_resources)}")
        
        # Étape 4 : Filtrer les ressources pertinentes
        relevant_resources = self._filter_resources(
            profile['interests'],
            profile['level'],
            target_level,
            profile['learning_style']
        )
        print(f"✓ Ressources pertinentes trouvées: {len(relevant_resources)}")
        
        # Étape 5 : Construire le parcours optimal
        learning_path = self._build_optimal_path(
            relevant_resources,
            completed_resources,
            profile['level'],
            target_level
        )
        print(f"✓ Parcours construit: {len(learning_path)} étapes")
        
        # Étape 6 : Générer une explication avec le LLM
        explanation = self._generate_path_explanation(profile, learning_path, target_level)
        
        # Créer le résultat final
        path_result = {
            "user_id": user_id,
            "current_level": profile['level'],
            "target_level": target_level,
            "learning_style": profile['learning_style'],
            "path": learning_path,
            "total_steps": len(learning_path),
            "estimated_duration_minutes": sum(step['duration'] for step in learning_path),
            "explanation": explanation,
            "created_at": datetime.now().isoformat()
        }
        
        # Sauvegarder dans le Blackboard
        self.blackboard.write("learning_paths", user_id, path_result)
        print(f"\n✅ Parcours sauvegardé dans le Blackboard")
        
        return path_result
    
    def _determine_target_level(self, current_level: str) -> str:
        """
        Détermine le niveau cible en fonction du niveau actuel
        
        Args:
            current_level: Niveau actuel de l'utilisateur
        
        Returns:
            Niveau cible recommandé
        """
        level_progression = {
            "beginner": "intermediate",
            "intermediate": "advanced",
            "advanced": "advanced"  # Reste avancé
        }
        
        return level_progression.get(current_level, "intermediate")
    
    def _get_completed_resources(self, user_id: str) -> List[str]:
        """
        Récupère les ressources déjà complétées par l'utilisateur
        
        Args:
            user_id: ID de l'utilisateur
        
        Returns:
            Liste des IDs de ressources complétées
        """
        history = self.blackboard.get_user_history(user_id)
        
        # Extraire les ressources avec score > 70% ou durée > 50% du temps recommandé
        completed = set()
        
        for interaction in history:
            resource_id = interaction.get("resource_id", "")
            
            # Si c'est un quiz avec bon score
            if interaction.get("type") == "quiz" and interaction.get("score", 0) >= 70:
                completed.add(resource_id)
            
            # Si c'est une ressource visionnée suffisamment longtemps
            elif interaction.get("type") in ["view", "exercise"] and interaction.get("duration", 0) > 60:
                completed.add(resource_id)
        
        return list(completed)
    
    def _filter_resources(self, interests: List[str], current_level: str, 
                         target_level: str, learning_style: str) -> List[Dict]:
        """
        Filtre les ressources pertinentes selon le profil
        
        Args:
            interests: Intérêts de l'utilisateur
            current_level: Niveau actuel
            target_level: Niveau cible
            learning_style: Style d'apprentissage
        
        Returns:
            Liste de ressources filtrées
        """
        filtered = []
        
        # Définir les niveaux acceptables
        level_order = ["beginner", "intermediate", "advanced"]
        current_idx = level_order.index(current_level)
        target_idx = level_order.index(target_level)
        acceptable_levels = level_order[current_idx:target_idx + 1]
        
        # Mapper le style d'apprentissage aux types de ressources préférés
        style_preferences = {
            "visual": ["video", "diagram"],
            "kinesthetic": ["exercise", "quiz", "project"],
            "reading": ["course", "article", "doc"],
            "auditory": ["audio", "podcast"]
        }
        preferred_types = style_preferences.get(learning_style, ["course", "video"])
        
        for resource in self.resource_catalog:
            # Filtrer par niveau
            if resource['level'] not in acceptable_levels:
                continue
            
            # Filtrer par intérêt/topic
            if not any(interest in resource['topic'] for interest in interests):
                continue
            
            # Bonus pour les ressources qui matchent le style d'apprentissage
            resource_copy = resource.copy()
            if resource['type'] in preferred_types:
                resource_copy['priority'] = 1  # Haute priorité
            else:
                resource_copy['priority'] = 2  # Priorité normale
            
            filtered.append(resource_copy)
        
        # Trier par priorité et niveau
        filtered.sort(key=lambda x: (x['priority'], level_order.index(x['level'])))
        
        return filtered
    
    def _build_optimal_path(self, resources: List[Dict], completed: List[str],
                       current_level: str, target_level: str) -> List[Dict]:
        """
        Construit le parcours optimal en utilisant A* graph search et Q-Learning RL
        
        Args:
            resources: Ressources disponibles
            completed: Ressources déjà complétées
            current_level: Niveau actuel
            target_level: Niveau cible
        
        Returns:
            Parcours optimal
        """
        # Utiliser A* pour trouver le chemin optimal
        path = self._astar_path_search(resources, completed, current_level, target_level)
        
        # Affiner avec Q-Learning (RL)
        if len(path) > 0:
            path = self._apply_rl_optimization(path, resources)
        
        return path
    
    def _astar_path_search(self, resources: List[Dict], completed: List[str],
                          current_level: str, target_level: str) -> List[Dict]:
        """
        Utilise A* pour chercher le parcours optimal dans le graphe de ressources
        
        Args:
            resources: Ressources disponibles
            completed: Ressources déjà complétées
            current_level: Niveau actuel
            target_level: Niveau cible
        
        Returns:
            Parcours trouvé par A*
        """
        # État initial : ressources complétées
        initial_state = frozenset(completed)
        
        # Priority queue: (f_score, g_score, state, path)
        heap = [(0, 0, initial_state, [])]
        visited = set()
        
        level_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
        target_level_idx = level_map.get(target_level, 2)
        
        max_steps = 10
        
        while heap:
            f_score, g_score, state, path = heapq.heappop(heap)
            
            # Limiter la profondeur
            if len(path) >= max_steps:
                return path
            
            # Éviter les états visités
            if state in visited:
                continue
            visited.add(state)
            
            # Heuristique : si on a atteint suffisamment de ressources du niveau cible
            if self._goal_reached(path, target_level_idx):
                return path
            
            # Trouver les successeurs (prochaines ressources disponibles)
            for resource in resources:
                if resource['id'] in state:  # Déjà fait
                    continue
                
                # Vérifier les prérequis
                if not all(prereq in state for prereq in resource['prerequisites']):
                    continue
                
                # Créer le nouvel état
                new_state = frozenset(state | {resource['id']})
                new_path = path + [{
                    "step": len(path) + 1,
                    "resource_id": resource['id'],
                    "title": resource['title'],
                    "type": resource['type'],
                    "level": resource['level'],
                    "duration": resource['duration'],
                    "prerequisites": resource['prerequisites']
                }]
                
                # Coût g : durée cumulée
                new_g_score = g_score + resource['duration']
                
                # Heuristique h : distance au niveau cible
                resource_level_idx = level_map.get(resource['level'], 0)
                h_score = abs(target_level_idx - resource_level_idx) * 30
                
                # f = g + h
                new_f_score = new_g_score + h_score
                
                heapq.heappush(heap, (new_f_score, new_g_score, new_state, new_path))
        
        # Retour au fallback si A* ne trouve rien
        return self._fallback_greedy_path(resources, completed, max_steps)
    
    def _goal_reached(self, path: List[Dict], target_level_idx: int) -> bool:
        """
        Vérifie si l'objectif est atteint (suffisamment de ressources du niveau cible)
        """
        if len(path) < 5:
            return False
        
        level_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
        target_count = sum(1 for step in path if level_map.get(step['level'], 0) >= target_level_idx)
        
        return target_count >= 3
    
    def _fallback_greedy_path(self, resources: List[Dict], completed: List[str], max_steps: int) -> List[Dict]:
        """
        Fallback : construction greedy si A* échoue
        """
        path = []
        available = set(completed)
        
        while len(path) < max_steps:
            next_resources = [
                r for r in resources
                if r['id'] not in available
                and all(prereq in available for prereq in r['prerequisites'])
            ]
            
            if not next_resources:
                break
            
            # Prendre la première ressource avec priorité
            next_resource = next_resources[0]
            
            path.append({
                "step": len(path) + 1,
                "resource_id": next_resource['id'],
                "title": next_resource['title'],
                "type": next_resource['type'],
                "level": next_resource['level'],
                "duration": next_resource['duration'],
                "prerequisites": next_resource['prerequisites']
            })
            
            available.add(next_resource['id'])
        
        return path
    
    def _apply_rl_optimization(self, path: List[Dict], resources: List[Dict]) -> List[Dict]:
        """
        Applique Q-Learning pour optimiser le parcours
        
        Args:
            path: Parcours initial trouvé par A*
            resources: Toutes les ressources disponibles
        
        Returns:
            Parcours optimisé
        """
        # Créer un mapping des ressources
        resource_map = {r['id']: r for r in resources}
        
        # Pour chaque étape, vérifier si on peut améliorer avec Q-Learning
        for i in range(len(path)):
            state = tuple(step['resource_id'] for step in path[:i])
            current_action = path[i]['resource_id']
            
            # Explorer parfois (epsilon-greedy)
            if np.random.random() < self.epsilon and i < len(path) - 1:
                # Trouver des alternatives
                alternatives = self._find_alternative_resources(path[i], resources, set(s['resource_id'] for s in path[:i]))
                
                if alternatives:
                    # Choisir la meilleure selon Q-table
                    best_alt = max(alternatives, key=lambda alt: self.q_table[state][alt['id']])
                    
                    # Mettre à jour Q-value
                    reward = self._calculate_reward(best_alt)
                    next_state = state + (best_alt['id'],)
                    
                    # Q-Learning update
                    old_q = self.q_table[state][current_action]
                    max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
                    new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
                    self.q_table[state][current_action] = new_q
        
        return path
    
    def _find_alternative_resources(self, current_step: Dict, resources: List[Dict], completed_ids: set) -> List[Dict]:
        """
        Trouve des ressources alternatives au même niveau
        """
        alternatives = []
        for res in resources:
            if (res['level'] == current_step['level'] 
                and res['id'] not in completed_ids
                and res['id'] != current_step['resource_id']):
                alternatives.append(res)
        return alternatives
    
    def _calculate_reward(self, resource: Dict) -> float:
        """
        Calcule la récompense pour une ressource (utilisé par RL)
        """
        # Récompense basée sur priorité et durée
        priority_reward = resource.get('priority', 1) * 10
        duration_penalty = -resource['duration'] * 0.1  # Pénalité pour durée longue
        
        return priority_reward + duration_penalty
    
    def _generate_path_explanation(self, profile: Dict, path: List[Dict], 
                                   target_level: str) -> str:
        """
        Génère une explication textuelle du parcours avec le LLM
        
        Args:
            profile: Profil utilisateur
            path: Parcours planifié
            target_level: Niveau cible
        
        Returns:
            Explication textuelle
        """
        # Construire le prompt
        path_summary = "\n".join([
            f"  Step {step['step']}: {step['title']} ({step['type']}, {step['duration']}min)"
            for step in path[:5]  # Limiter à 5 pour le prompt
        ])
        
        prompt = f"""You are an educational advisor. Explain this learning path briefly (2-3 sentences).

User Profile:
- Current Level: {profile['level']}
- Learning Style: {profile['learning_style']}
- Interests: {', '.join(profile['interests'])}

Planned Learning Path (first 5 steps):
{path_summary}

Target Level: {target_level}
Total Steps: {len(path)}

Provide a clear, motivating explanation of why this path is suitable."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            explanation = response["message"]["content"].strip()
            return explanation
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return f"This {len(path)}-step learning path is designed to progress from {profile['level']} to {target_level} level, adapted to your {profile['learning_style']} learning style."
    
    def update_path_progress(self, user_id: str, completed_step: int) -> Dict:
        """
        Met à jour la progression dans le parcours
        
        Args:
            user_id: ID de l'utilisateur
            completed_step: Numéro de l'étape complétée
        
        Returns:
            État mis à jour du parcours
        """
        path = self.blackboard.read("learning_paths", user_id)
        
        if not path:
            return {"error": "No learning path found"}
        
        # Marquer l'étape comme complétée
        for step in path['path']:
            if step['step'] == completed_step:
                step['completed'] = True
                step['completed_at'] = datetime.now().isoformat()
        
        # Calculer la progression
        completed_steps = sum(1 for step in path['path'] if step.get('completed', False))
        path['progress_percentage'] = (completed_steps / len(path['path'])) * 100
        
        # Sauvegarder
        self.blackboard.write("learning_paths", user_id, path)
        
        print(f"✓ Progression mise à jour: {path['progress_percentage']:.1f}%")
        
        return path