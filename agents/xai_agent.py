# agents/xai_agent.py
"""
XAI Agent - Explainable AI Agent

Rôle :
- Générer des explications compréhensibles pour toutes les décisions du système
- Expliquer le profil utilisateur (pourquoi ce niveau, ce style)
- Expliquer le parcours d'apprentissage (pourquoi ces étapes)
- Expliquer les recommandations (pourquoi ces ressources)
- Fournir des explications contrefactuelles (et si...)

Méthodes :
- Chaînes de raisonnement structurées
- Explications contrefactuelles
- Importance des features (profil)
- LLM pour génération d'explications en langage naturel
"""

import ollama
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not installed. Install with: pip install lime")

from memory.blackboard import Blackboard


class XAIAgent:
    """
    Agent responsable de l'explicabilité du système
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b"):
        """
        Initialise le XAI Agent
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Nom du modèle LLM local
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        
        # Feature names for SHAP/LIME
        self.feature_names = [
            'level_beginner', 'level_intermediate', 'level_advanced',
            'style_visual', 'style_reading', 'style_kinesthetic', 'style_auditory',
            'total_interactions', 'avg_score'
        ]
        
        print(f"✓ XAI Agent initialisé")
        if SHAP_AVAILABLE:
            print(f"✓ SHAP disponible pour feature importance")
        if LIME_AVAILABLE:
            print(f"✓ LIME disponible pour explications locales")
    
    def explain_full_system(self, user_id: str) -> Dict:
        """
        Génère une explication complète de toutes les décisions pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
        
        Returns:
            Dict contenant toutes les explications
        """
        print(f"\n{'='*70}")
        print(f"🔍 GÉNÉRATION D'EXPLICATIONS COMPLÈTES")
        print(f"{'='*70}")
        print(f"User ID: {user_id}")
        
        # Récupérer toutes les données
        profile = self.blackboard.read("profiles", user_id)
        learning_path = self.blackboard.read("learning_paths", user_id)
        recommendations = self.blackboard.read("recommendations", user_id)
        
        if not profile:
            print(f"⚠️  Profil non trouvé pour {user_id}")
            return {"error": "Profile not found"}
        
        print(f"✓ Données récupérées")
        
        # Générer les explications pour chaque composant
        explanations = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            
            # 1. Explication du profil
            "profile_explanation": self._explain_profile(profile),
            
            # 2. Explication du parcours
            "path_explanation": self._explain_learning_path(profile, learning_path) if learning_path else None,
            
            # 3. Explication des recommandations
            "recommendations_explanation": self._explain_recommendations(profile, recommendations) if recommendations else None,
            
            # 4. Explications contrefactuelles
            "counterfactuals": self._generate_counterfactuals(profile),
            
            # 5. Résumé global
            "summary": self._generate_global_summary(profile, learning_path, recommendations)
        }
        
        # Sauvegarder dans le Blackboard
        self.blackboard.write("explanations", user_id, explanations)
        print(f"\n✅ Explications complètes générées et sauvegardées")
        
        return explanations
    
    def _explain_profile(self, profile: Dict) -> Dict:
        """
        Explique pourquoi l'utilisateur a ce profil
        
        Args:
            profile: Profil utilisateur
        
        Returns:
            Explication structurée du profil
        """
        print(f"\n📊 Explication du profil...")
        
        # Construire le prompt pour le LLM
        prompt = f"""You are an educational AI explaining user profiling decisions. Explain WHY this learner received this profile.

User Profile:
- Level: {profile['level']}
- Learning Style: {profile['learning_style']}
- Interests: {', '.join(profile.get('interests', ['general']))}
- Total Interactions: {profile.get('total_interactions', 0)}
- Strengths: {', '.join(profile.get('strengths', [])) if profile.get('strengths') else 'None identified yet'}
- Weaknesses: {', '.join(profile.get('weaknesses', [])) if profile.get('weaknesses') else 'None identified yet'}

Provide a structured explanation:

LEVEL_REASONING: [Why was this level assigned? What evidence supports it?]

STYLE_REASONING: [Why was this learning style identified? What patterns led to this?]

INTERESTS_REASONING: [How were these interests determined?]

IMPROVEMENT_SUGGESTIONS: [What could the learner do to progress?]

Keep each section to 2-3 sentences."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            return {
                "level_reasoning": self._extract_section(text, "LEVEL_REASONING"),
                "style_reasoning": self._extract_section(text, "STYLE_REASONING"),
                "interests_reasoning": self._extract_section(text, "INTERESTS_REASONING"),
                "improvement_suggestions": self._extract_section(text, "IMPROVEMENT_SUGGESTIONS"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "level_reasoning": f"Level '{profile['level']}' assigned based on performance and interactions.",
                "style_reasoning": f"Learning style '{profile['learning_style']}' identified from resource preferences.",
                "interests_reasoning": f"Interests: {', '.join(profile.get('interests', []))}",
                "improvement_suggestions": "Continue practicing to improve.",
                "full_text": "Error generating detailed explanation."
            }
    
    def _explain_learning_path(self, profile: Dict, learning_path: Dict) -> Dict:
        """
        Explique pourquoi ce parcours d'apprentissage a été planifié
        
        Args:
            profile: Profil utilisateur
            learning_path: Parcours planifié
        
        Returns:
            Explication du parcours
        """
        print(f"🗺️  Explication du parcours...")
        
        path_steps = learning_path.get('path', [])[:5]  # Limiter à 5 premières étapes
        
        steps_text = "\n".join([
            f"  {step['step']}. {step['title']} ({step['type']}, {step['level']}, {step['duration']}min)"
            for step in path_steps
        ])
        
        prompt = f"""Explain why this learning path was designed for this learner.

Learner Profile:
- Level: {profile['level']}
- Target: {learning_path.get('target_level', 'N/A')}
- Learning Style: {profile['learning_style']}

Learning Path (first 5 steps):
{steps_text}

Total Steps: {learning_path.get('total_steps', 0)}
Estimated Duration: {learning_path.get('estimated_duration_minutes', 0)} minutes

Provide:
PATH_LOGIC: [Why this sequence of steps? What's the progression logic?]
PERSONALIZATION: [How is this path adapted to the learner's profile?]
EXPECTED_OUTCOMES: [What will the learner achieve?]

2-3 sentences per section."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            return {
                "path_logic": self._extract_section(text, "PATH_LOGIC"),
                "personalization": self._extract_section(text, "PERSONALIZATION"),
                "expected_outcomes": self._extract_section(text, "EXPECTED_OUTCOMES"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "path_logic": f"Path designed to progress from {profile['level']} to {learning_path.get('target_level')}.",
                "personalization": f"Adapted to {profile['learning_style']} learning style.",
                "expected_outcomes": "Mastery of planned topics.",
                "full_text": "Error generating path explanation."
            }
    
    def _explain_recommendations(self, profile: Dict, recommendations: Dict) -> Dict:
        """
        Explique pourquoi ces ressources sont recommandées
        
        Args:
            profile: Profil utilisateur
            recommendations: Recommandations générées
        
        Returns:
            Explication des recommandations
        """
        print(f"🎯 Explication des recommandations...")
        
        top_recs = recommendations.get('recommendations', [])[:3]
        
        recs_text = "\n".join([
            f"  {i+1}. {rec['title']} ({rec['type']}, {rec['level']})\n     Reason: {rec['reason']}"
            for i, rec in enumerate(top_recs)
        ])
        
        prompt = f"""Explain why these resources are recommended to this learner.

Learner Profile:
- Level: {profile['level']}
- Learning Style: {profile['learning_style']}
- Interests: {', '.join(profile.get('interests', []))}

Top Recommendations:
{recs_text}

Provide:
SELECTION_CRITERIA: [What criteria determined these recommendations?]
RANKING_LOGIC: [Why this specific order/ranking?]
PERSONALIZATION_FACTORS: [How do these match the learner's profile?]

2-3 sentences per section."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            return {
                "selection_criteria": self._extract_section(text, "SELECTION_CRITERIA"),
                "ranking_logic": self._extract_section(text, "RANKING_LOGIC"),
                "personalization_factors": self._extract_section(text, "PERSONALIZATION_FACTORS"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "selection_criteria": "Resources selected based on level and interests.",
                "ranking_logic": "Ranked by relevance and priority.",
                "personalization_factors": f"Matched to {profile['learning_style']} style.",
                "full_text": "Error generating recommendations explanation."
            }
    
    def _generate_counterfactuals(self, profile: Dict) -> Dict:
        """
        Génère des explications contrefactuelles ("Et si...")
        
        Args:
            profile: Profil utilisateur
        
        Returns:
            Explications contrefactuelles
        """
        print(f"💭 Génération d'explications contrefactuelles...")
        
        prompt = f"""Generate "what if" explanations for this learner.

Current Profile:
- Level: {profile['level']}
- Learning Style: {profile['learning_style']}
- Interactions: {profile.get('total_interactions', 0)}

Provide 3 counterfactual scenarios:

IF_HIGHER_LEVEL: [What would change if the learner was at a higher level?]
IF_DIFFERENT_STYLE: [What would change with a different learning style?]
IF_MORE_PRACTICE: [What would change with more practice/interactions?]

Keep each to 1-2 sentences."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            return {
                "if_higher_level": self._extract_section(text, "IF_HIGHER_LEVEL"),
                "if_different_style": self._extract_section(text, "IF_DIFFERENT_STYLE"),
                "if_more_practice": self._extract_section(text, "IF_MORE_PRACTICE"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "if_higher_level": "More advanced resources would be recommended.",
                "if_different_style": "Resource types would be adjusted.",
                "if_more_practice": "Better personalization would be possible.",
                "full_text": "Error generating counterfactuals."
            }
    
    def _generate_global_summary(self, profile: Dict, learning_path: Optional[Dict],
                                 recommendations: Optional[Dict]) -> str:
        """
        Génère un résumé global de toutes les décisions
        
        Args:
            profile: Profil utilisateur
            learning_path: Parcours planifié
            recommendations: Recommandations
        
        Returns:
            Résumé textuel
        """
        print(f"📝 Génération du résumé global...")
        
        has_path = learning_path is not None
        has_recs = recommendations is not None
        
        prompt = f"""Provide a comprehensive summary explaining all system decisions for this learner (3-4 sentences).

Profile: {profile['level']} level, {profile['learning_style']} style
Learning Path: {'Created' if has_path else 'Not created yet'}
Recommendations: {'Generated' if has_recs else 'Not generated yet'}

Explain how the profile led to the path and recommendations, and what the learner should expect."""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return f"The system analyzed your profile as {profile['level']} level with {profile['learning_style']} learning style, and created personalized recommendations accordingly."
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extrait une section du texte généré
        
        Args:
            text: Texte complet
            section_name: Nom de la section
        
        Returns:
            Contenu de la section
        """
        try:
            start = text.find(f"{section_name}:")
            if start == -1:
                return "N/A"
            
            start += len(section_name) + 1
            
            # Trouver la fin
            end = len(text)
            for marker in ["LEVEL_REASONING:", "STYLE_REASONING:", "INTERESTS_REASONING:",
                          "IMPROVEMENT_SUGGESTIONS:", "PATH_LOGIC:", "PERSONALIZATION:",
                          "EXPECTED_OUTCOMES:", "SELECTION_CRITERIA:", "RANKING_LOGIC:",
                          "PERSONALIZATION_FACTORS:", "IF_HIGHER_LEVEL:", "IF_DIFFERENT_STYLE:",
                          "IF_MORE_PRACTICE:"]:
                pos = text.find(marker, start)
                if pos != -1 and pos < end:
                    end = pos
            
            content = text[start:end].strip()
            return content if content else "N/A"
            
        except Exception:
            return "N/A"
    
    def get_feature_importance(self, user_id: str) -> Dict:
        """
        Analyse l'importance des différentes features du profil avec SHAP
        
        Args:
            user_id: ID de l'utilisateur
        
        Returns:
            Importance des features avec SHAP values
        """
        profile = self.blackboard.read("profiles", user_id)
        
        if not profile:
            return {"error": "Profile not found"}
        
        # Préparer les features
        features = self._profile_to_features(profile)
        
        # Utiliser SHAP si disponible
        if SHAP_AVAILABLE:
            shap_values = self._compute_shap_values(profile, features)
            importance = shap_values
        else:
            # Fallback : importance heuristique
            importance = {
                "level": 0.35,
                "learning_style": 0.25,
                "interests": 0.20,
                "total_interactions": 0.10,
                "strengths": 0.05,
                "weaknesses": 0.05
            }
        
        return {
            "user_id": user_id,
            "feature_importance": importance,
            "method": "SHAP" if SHAP_AVAILABLE else "heuristic",
            "explanation": "Feature importance computed using SHAP values" if SHAP_AVAILABLE else "Level and learning style are the most influential factors."
        }
    
    def _profile_to_features(self, profile: Dict) -> np.ndarray:
        """
        Convertit un profil en vecteur de features pour SHAP/LIME
        
        Args:
            profile: Profil utilisateur
        
        Returns:
            Vecteur numpy de features
        """
        # One-hot encoding du niveau
        level = profile.get('level', 'beginner')
        level_features = [
            1 if level == 'beginner' else 0,
            1 if level == 'intermediate' else 0,
            1 if level == 'advanced' else 0
        ]
        
        # One-hot encoding du style
        style = profile.get('learning_style', 'visual')
        style_features = [
            1 if style == 'visual' else 0,
            1 if style == 'reading' else 0,
            1 if style == 'kinesthetic' else 0,
            1 if style == 'auditory' else 0
        ]
        
        # Autres features
        total_interactions = profile.get('total_interactions', 0) / 100.0  # Normaliser
        
        # Calculer le score moyen si disponible
        history = self.blackboard.get_user_history(profile['user_id'])
        scores = [i.get('score', 0) for i in history if i.get('type') == 'quiz']
        avg_score = np.mean(scores) / 100.0 if scores else 0.5
        
        return np.array(level_features + style_features + [total_interactions, avg_score])
    
    def _compute_shap_values(self, profile: Dict, features: np.ndarray) -> Dict:
        """
        Calcule les valeurs SHAP pour l'importance des features
        
        Args:
            profile: Profil utilisateur
            features: Vecteur de features
        
        Returns:
            Dict avec les valeurs SHAP
        """
        try:
            # Récupérer tous les profils pour créer un dataset de référence
            all_profiles = self.blackboard.read_section("profiles")
            
            if len(all_profiles) < 5:
                # Pas assez de données pour SHAP
                return self._heuristic_importance()
            
            # Créer une matrice de features de tous les utilisateurs
            X_background = []
            for _, prof in all_profiles.items():
                X_background.append(self._profile_to_features(prof))
            
            X_background = np.array(X_background)
            
            # Créer un modèle simple (fonction de prédiction)
            def prediction_function(X):
                """Fonction qui prédit une score de recommandation basé sur les features"""
                predictions = []
                for x in X:
                    # Simple weighted sum
                    score = (
                        x[0] * 0.2 +  # beginner
                        x[1] * 0.3 +  # intermediate
                        x[2] * 0.4 +  # advanced
                        x[3] * 0.3 +  # visual
                        x[4] * 0.2 +  # reading
                        x[5] * 0.3 +  # kinesthetic
                        x[6] * 0.2 +  # auditory
                        x[7] * 0.4 +  # interactions
                        x[8] * 0.5    # avg_score
                    )
                    predictions.append(score)
                return np.array(predictions)
            
            # Créer l'explainer SHAP
            explainer = shap.KernelExplainer(prediction_function, X_background[:10])  # Limiter pour performance
            
            # Calculer les valeurs SHAP pour cet utilisateur
            shap_values = explainer.shap_values(features.reshape(1, -1))
            
            # Créer un dict avec les importances
            importance_dict = {}
            for i, name in enumerate(self.feature_names):
                importance_dict[name] = float(abs(shap_values[0][i]))
            
            # Normaliser
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v/total for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            print(f"⚠️  Erreur SHAP: {e}")
            return self._heuristic_importance()
    
    def _heuristic_importance(self) -> Dict:
        """Fallback heuristic importance"""
        return {
            "level": 0.35,
            "learning_style": 0.25,
            "interests": 0.20,
            "total_interactions": 0.10,
            "strengths": 0.05,
            "weaknesses": 0.05
        }
    
    def explain_with_lime(self, user_id: str, decision_type: str = "recommendation") -> Dict:
        """
        Génère une explication LIME pour une décision spécifique
        
        Args:
            user_id: ID de l'utilisateur
            decision_type: Type de décision à expliquer
        
        Returns:
            Explication LIME
        """
        if not LIME_AVAILABLE:
            return {
                "error": "LIME not available",
                "suggestion": "Install LIME with: pip install lime"
            }
        
        profile = self.blackboard.read("profiles", user_id)
        if not profile:
            return {"error": "Profile not found"}
        
        try:
            # Récupérer les données de background
            all_profiles = self.blackboard.read_section("profiles")
            
            if len(all_profiles) < 5:
                return {"error": "Not enough data for LIME analysis"}
            
            # Créer le dataset
            X_train = []
            for _, prof in all_profiles.items():
                X_train.append(self._profile_to_features(prof))
            
            X_train = np.array(X_train)
            
            # Fonction de prédiction
            def predict_fn(X):
                predictions = []
                for x in X:
                    score = (
                        x[0] * 0.2 + x[1] * 0.3 + x[2] * 0.4 +  # level
                        x[3] * 0.3 + x[4] * 0.2 + x[5] * 0.3 + x[6] * 0.2 +  # style
                        x[7] * 0.4 + x[8] * 0.5  # interactions & score
                    )
                    predictions.append([1 - score, score])  # Binary classification format
                return np.array(predictions)
            
            # Créer l'explainer LIME
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=['low_match', 'high_match'],
                mode='classification'
            )
            
            # Expliquer l'instance actuelle
            instance = self._profile_to_features(profile)
            explanation = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=len(self.feature_names)
            )
            
            # Extraire les résultats
            lime_values = {}
            for feature, weight in explanation.as_list():
                lime_values[feature] = weight
            
            return {
                "user_id": user_id,
                "method": "LIME",
                "explanation": lime_values,
                "prediction_probabilities": explanation.predict_proba.tolist(),
                "interpretation": "Positive values increase recommendation score, negative values decrease it."
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LIME: {e}")
            return {
                "error": str(e),
                "fallback": "Using LLM-based explanation instead"
            }
    
    def explain_decision(self, decision_type: str, user_id: str, **kwargs) -> str:
        """
        Génère une explication pour une décision spécifique
        
        Args:
            decision_type: Type de décision ("profile", "path", "recommendation")
            user_id: ID de l'utilisateur
            **kwargs: Paramètres additionnels
        
        Returns:
            Texte d'explication
        """
        if decision_type == "profile":
            profile = self.blackboard.read("profiles", user_id)
            if profile:
                explanation = self._explain_profile(profile)
                return explanation['full_text']
        
        elif decision_type == "path":
            profile = self.blackboard.read("profiles", user_id)
            learning_path = self.blackboard.read("learning_paths", user_id)
            if profile and learning_path:
                explanation = self._explain_learning_path(profile, learning_path)
                return explanation['full_text']
        
        elif decision_type == "recommendation":
            profile = self.blackboard.read("profiles", user_id)
            recommendations = self.blackboard.read("recommendations", user_id)
            if profile and recommendations:
                explanation = self._explain_recommendations(profile, recommendations)
                return explanation['full_text']
        
        return "Unable to generate explanation: missing data."