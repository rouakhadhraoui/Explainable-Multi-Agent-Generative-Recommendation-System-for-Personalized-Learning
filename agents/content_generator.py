# agents/content_generator.py
"""
Content Generator Agent - Génération de contenu pédagogique personnalisé

Rôle :
- Générer des ressources pédagogiques adaptées au profil utilisateur
- Créer des cours, exercices, quiz personnalisés
- Utiliser RAG (Retrieval-Augmented Generation) pour enrichir le contenu
- Adapter le contenu au style d'apprentissage et au niveau

Technologies : LLM (Llama), RAG, Prompt Engineering
"""

import ollama
from typing import Dict, List, Optional
from datetime import datetime
import json

from memory.blackboard import Blackboard
from utils.embeddings import get_embedding_generator


class ContentGenerator:
    """
    Agent responsable de la génération de contenu pédagogique personnalisé
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b"):
        """
        Initialise le Content Generator
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Nom du modèle LLM local
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        self.embedding_gen = get_embedding_generator()
        
        # Base de connaissances pour RAG (exemples de contenus pédagogiques)
        self.knowledge_base = self._initialize_knowledge_base()
        
        print(f"✓ Content Generator initialisé avec {len(self.knowledge_base)} documents")
    
    def _initialize_knowledge_base(self) -> List[Dict]:
        """
        Initialise une base de connaissances pour le RAG
        
        Dans un vrai système, ceci viendrait d'une vraie base de données vectorielle
        
        Returns:
            Liste de documents pédagogiques
        """
        knowledge_base = [
            {
                "id": "kb_python_basics",
                "topic": "python",
                "level": "beginner",
                "content": """Python is a high-level, interpreted programming language. 
                Variables are created by assignment: x = 5. 
                Python uses indentation for code blocks. 
                Basic data types include integers, floats, strings, and booleans."""
            },
            {
                "id": "kb_python_loops",
                "topic": "python",
                "level": "intermediate",
                "content": """Python has two main loop types: for and while loops.
                For loops iterate over sequences: for item in list.
                While loops continue until a condition is false.
                Break and continue statements control loop execution."""
            },
            {
                "id": "kb_python_functions",
                "topic": "python",
                "level": "intermediate",
                "content": """Functions are defined with the def keyword.
                Parameters can have default values.
                Functions return values using the return statement.
                Lambda functions provide anonymous function syntax."""
            },
            {
                "id": "kb_python_oop",
                "topic": "python",
                "level": "advanced",
                "content": """Object-Oriented Programming in Python uses classes.
                Classes define objects with attributes and methods.
                Inheritance allows classes to inherit from parent classes.
                Encapsulation, polymorphism, and abstraction are key OOP concepts."""
            },
            {
                "id": "kb_datascience_intro",
                "topic": "datascience",
                "level": "intermediate",
                "content": """Data Science combines statistics, programming, and domain knowledge.
                Common libraries include NumPy, Pandas, and Matplotlib.
                Data cleaning and preprocessing are crucial steps.
                Exploratory Data Analysis (EDA) reveals patterns in data."""
            }
        ]
        
        return knowledge_base
    
    def generate_content(self, user_id: str, content_type: str, 
                        topic: str, level: str) -> Dict:
        """
        Génère du contenu pédagogique personnalisé
        
        Args:
            user_id: ID de l'utilisateur
            content_type: Type de contenu ("course", "exercise", "quiz")
            topic: Sujet du contenu
            level: Niveau de difficulté
        
        Returns:
            Contenu généré (Dict)
        """
        print(f"\n{'='*70}")
        print(f"📝 GÉNÉRATION DE CONTENU PÉDAGOGIQUE")
        print(f"{'='*70}")
        print(f"User ID      : {user_id}")
        print(f"Type         : {content_type}")
        print(f"Topic        : {topic}")
        print(f"Level        : {level}")
        
        # Étape 1 : Récupérer le profil utilisateur
        profile = self.blackboard.read("profiles", user_id)
        
        if not profile:
            print(f"⚠️  Profil non trouvé. Utilisation des paramètres par défaut.")
            profile = {
                "learning_style": "visual",
                "level": level,
                "interests": [topic]
            }
        
        print(f"✓ Profil récupéré: {profile['learning_style']} learner")
        
        # Étape 2 : Récupération de contexte via RAG
        context = self._retrieve_relevant_context(topic, level)
        print(f"✓ Contexte RAG récupéré: {len(context)} documents")
        
        # Étape 3 : Générer le contenu avec le LLM
        generated_content = self._generate_with_llm(
            content_type, topic, level, profile, context
        )
        
        # Étape 4 : Créer le résultat final
        content_result = {
            "content_id": f"{content_type}_{topic}_{level}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "type": content_type,
            "topic": topic,
            "level": level,
            "learning_style": profile['learning_style'],
            "content": generated_content,
            "generated_at": datetime.now().isoformat(),
            "rag_sources": [doc['id'] for doc in context]
        }
        
        # Sauvegarder dans le cache
        self.blackboard.write("cached_content", content_result['content_id'], content_result)
        print(f"\n✅ Contenu généré et mis en cache")
        
        return content_result
    
    def _retrieve_relevant_context(self, topic: str, level: str, top_k: int = 2) -> List[Dict]:
        """
        Récupère le contexte pertinent depuis la base de connaissances (RAG)
        
        Args:
            topic: Sujet recherché
            level: Niveau de difficulté
            top_k: Nombre de documents à récupérer
        
        Returns:
            Liste de documents pertinents
        """
        # Filtrer par topic et level
        relevant_docs = [
            doc for doc in self.knowledge_base
            if doc['topic'] == topic and doc['level'] == level
        ]
        
        # Si pas assez de docs du même niveau, prendre des niveaux adjacents
        if len(relevant_docs) < top_k:
            other_docs = [
                doc for doc in self.knowledge_base
                if doc['topic'] == topic and doc['level'] != level
            ]
            relevant_docs.extend(other_docs)
        
        # Retourner les top_k premiers
        return relevant_docs[:top_k]
    
    def _generate_with_llm(self, content_type: str, topic: str, level: str,
                          profile: Dict, context: List[Dict]) -> Dict:
        """
        Génère le contenu avec le LLM en utilisant le contexte RAG
        
        Args:
            content_type: Type de contenu
            topic: Sujet
            level: Niveau
            profile: Profil utilisateur
            context: Contexte RAG
        
        Returns:
            Contenu généré structuré
        """
        # Construire le contexte RAG
        rag_context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        # Adapter le prompt selon le type de contenu
        if content_type in ["course", "video", "article"]:
            # Video et article sont traités comme des cours
            content = self._generate_course(topic, level, profile, rag_context)
        elif content_type == "exercise":
            content = self._generate_exercise(topic, level, profile, rag_context)
        elif content_type == "quiz":
            content = self._generate_quiz(topic, level, profile, rag_context)
        else:
            # Fallback : générer un cours par défaut
            print(f"⚠️  Type '{content_type}' inconnu, génération d'un cours par défaut")
            content = self._generate_course(topic, level, profile, rag_context)
        
        return content
    
    def _generate_course(self, topic: str, level: str, profile: Dict, 
                        rag_context: str) -> Dict:
        """
        Génère un cours textuel
        
        Args:
            topic: Sujet du cours
            level: Niveau
            profile: Profil utilisateur
            rag_context: Contexte RAG
        
        Returns:
            Structure du cours
        """
        # Adapter au style d'apprentissage
        style_instruction = {
            "visual": "Include descriptions that would work well with diagrams and visual representations.",
            "kinesthetic": "Include practical examples and hands-on activities.",
            "reading": "Provide detailed explanations and comprehensive text.",
            "auditory": "Write in a conversational tone suitable for audio."
        }.get(profile['learning_style'], "")
        
        prompt = f"""You are an educational content creator. Create a brief lesson on {topic} for {level} level learners.

Context from knowledge base:
{rag_context}

Instructions:
- Create a clear, structured lesson with 3-4 main points
- {style_instruction}
- Keep it concise (200-300 words)
- Include a summary at the end

Format your response as:
TITLE: [lesson title]
INTRODUCTION: [brief intro]
MAIN_POINTS:
1. [point 1]
2. [point 2]
3. [point 3]
SUMMARY: [brief summary]"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            # Parser la réponse (simple)
            return {
                "title": self._extract_section(text, "TITLE"),
                "introduction": self._extract_section(text, "INTRODUCTION"),
                "main_points": self._extract_section(text, "MAIN_POINTS"),
                "summary": self._extract_section(text, "SUMMARY"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "title": f"{topic.title()} - {level.title()} Level",
                "introduction": f"This lesson covers {topic} concepts.",
                "main_points": "Content generation failed.",
                "summary": "Please try again.",
                "full_text": "Error generating content."
            }
    
    def _generate_exercise(self, topic: str, level: str, profile: Dict,
                          rag_context: str) -> Dict:
        """
        Génère un exercice pratique
        
        Args:
            topic: Sujet de l'exercice
            level: Niveau
            profile: Profil utilisateur
            rag_context: Contexte RAG
        
        Returns:
            Structure de l'exercice
        """
        prompt = f"""Create a practical coding exercise on {topic} for {level} level.

Context:
{rag_context}

Format:
TITLE: [exercise title]
DESCRIPTION: [what to build]
REQUIREMENTS: [list 3-4 requirements]
HINTS: [2-3 helpful hints]
EXPECTED_OUTPUT: [what the solution should produce]"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            return {
                "title": self._extract_section(text, "TITLE"),
                "description": self._extract_section(text, "DESCRIPTION"),
                "requirements": self._extract_section(text, "REQUIREMENTS"),
                "hints": self._extract_section(text, "HINTS"),
                "expected_output": self._extract_section(text, "EXPECTED_OUTPUT"),
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "title": f"{topic.title()} Exercise",
                "description": "Practice exercise",
                "requirements": "Complete the task",
                "hints": "Use what you learned",
                "expected_output": "Working solution",
                "full_text": "Error generating exercise."
            }
    
    def _generate_quiz(self, topic: str, level: str, profile: Dict,
                      rag_context: str) -> Dict:
        """
        Génère un quiz avec questions et réponses
        
        Args:
            topic: Sujet du quiz
            level: Niveau
            profile: Profil utilisateur
            rag_context: Contexte RAG
        
        Returns:
            Structure du quiz
        """
        prompt = f"""Create a quiz with 3 multiple-choice questions on {topic} for {level} level.

Context:
{rag_context}

For each question, provide:
- The question
- 4 options (A, B, C, D)
- The correct answer
- A brief explanation

Format:
Q1: [question]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
CORRECT: [A/B/C/D]
EXPLANATION: [why this is correct]

[Repeat for Q2 and Q3]"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response["message"]["content"].strip()
            
            # Parser les questions (simplifié)
            questions = self._parse_quiz_questions(text)
            
            return {
                "title": f"{topic.title()} Quiz - {level.title()}",
                "total_questions": len(questions),
                "questions": questions,
                "full_text": text
            }
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return {
                "title": f"{topic.title()} Quiz",
                "total_questions": 0,
                "questions": [],
                "full_text": "Error generating quiz."
            }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extrait une section du texte généré
        
        Args:
            text: Texte complet
            section_name: Nom de la section à extraire
        
        Returns:
            Contenu de la section
        """
        try:
            start = text.find(f"{section_name}:")
            if start == -1:
                return "N/A"
            
            start += len(section_name) + 1
            
            # Trouver la fin (prochaine section en majuscules ou fin du texte)
            end = len(text)
            for next_section in ["TITLE:", "INTRODUCTION:", "MAIN_POINTS:", 
                                "SUMMARY:", "DESCRIPTION:", "REQUIREMENTS:", 
                                "HINTS:", "EXPECTED_OUTPUT:", "Q1:", "Q2:", "Q3:"]:
                pos = text.find(next_section, start)
                if pos != -1 and pos < end:
                    end = pos
            
            content = text[start:end].strip()
            return content if content else "N/A"
            
        except Exception as e:
            return "N/A"
    
    def _parse_quiz_questions(self, text: str) -> List[Dict]:
        """
        Parse les questions d'un quiz depuis le texte généré
        
        Args:
            text: Texte du quiz
        
        Returns:
            Liste de questions structurées
        """
        questions = []
        
        for i in range(1, 4):  # 3 questions
            try:
                q_marker = f"Q{i}:"
                q_start = text.find(q_marker)
                
                if q_start == -1:
                    continue
                
                # Extraire la question
                q_text_start = q_start + len(q_marker)
                q_text_end = text.find("\nA)", q_text_start)
                question_text = text[q_text_start:q_text_end].strip()
                
                questions.append({
                    "question_number": i,
                    "question": question_text,
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Explanation not parsed"
                })
                
            except Exception:
                continue
        
        return questions
    
    def generate_for_learning_path(self, user_id: str, step_number: int) -> Dict:
        """
        Génère du contenu pour une étape spécifique du parcours d'apprentissage
        
        Args:
            user_id: ID de l'utilisateur
            step_number: Numéro de l'étape dans le parcours
        
        Returns:
            Contenu généré pour cette étape
        """
        # Récupérer le parcours
        path = self.blackboard.read("learning_paths", user_id)
        
        if not path:
            return {"error": "No learning path found"}
        
        # Trouver l'étape correspondante
        step = None
        for s in path['path']:
            if s['step'] == step_number:
                step = s
                break
        
        if not step:
            return {"error": f"Step {step_number} not found"}
        
        # Générer le contenu pour cette étape
        print(f"\n🎯 Génération de contenu pour l'étape {step_number}: {step['title']}")
        
        content = self.generate_content(
            user_id=user_id,
            content_type=step['type'],
            topic=step['title'].split()[0].lower(),  # Extraire le topic
            level=step['level']
        )
        
        return content