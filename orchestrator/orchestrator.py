# orchestrator/orchestrator.py
"""
Orchestrator - Coordinateur central du système multi-agents avec LangGraph

Rôle :
- Coordonner l'exécution séquentielle des agents via LangGraph
- Gérer le flux de données entre agents via le Blackboard
- Implémenter le pipeline cognitif complet
- Gérer les erreurs et les cas limites

Pipeline : User → Profiling → Path Planning → Content Generation → Recommendation → XAI

Technologies : LangGraph pour l'orchestration multi-agents
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from enum import Enum
import operator

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️  LangGraph import error: {e}")
    print("⚠️  Falling back to custom orchestration (LangGraph features disabled)")

from memory.blackboard import Blackboard
from agents.profiling_agent import ProfilingAgent
from agents.path_planning_agent import PathPlanningAgent
from agents.content_generator import ContentGenerator
from agents.recommendation_agent import RecommendationAgent
from agents.xai_agent import XAIAgent


class AgentStatus(Enum):
    """États possibles d'un agent"""
    PENDING = "pending"      # En attente d'exécution
    RUNNING = "running"      # En cours d'exécution
    COMPLETED = "completed"  # Terminé avec succès
    FAILED = "failed"        # Échec
    SKIPPED = "skipped"      # Ignoré


class AgentState(TypedDict):
    """
    État partagé entre les agents dans le graph LangGraph
    """
    user_id: str
    request_type: str
    profile: Optional[Dict]
    learning_path: Optional[Dict]
    generated_content: Optional[Dict]
    recommendations: Optional[Dict]
    explanations: Optional[Dict]
    errors: Annotated[List[str], operator.add]
    agent_results: Dict[str, Any]
    current_step: str


class Orchestrator:
    """
    Orchestrateur principal du système multi-agents
    """
    
    def __init__(self, blackboard: Blackboard, llm_model: str = "llama3.2:3b", use_langgraph: bool = True):
        """
        Initialise l'orchestrateur avec LangGraph
        
        Args:
            blackboard: Instance du Shared Memory
            llm_model: Modèle LLM à utiliser
            use_langgraph: Utiliser LangGraph si disponible (True par défaut)
        """
        self.blackboard = blackboard
        self.llm_model = llm_model
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        # Initialiser les agents disponibles
        self.agents = {}
        self._initialize_agents()
        
        # Pipeline d'exécution (ordre des agents)
        self.pipeline = [
            "profiling",
            "path_planning",
            "content_generator",
            "recommendation",
            "xai"
        ]
        
        # Historique des exécutions
        self.execution_history = []
        
        # Initialiser le graph LangGraph si disponible
        if self.use_langgraph:
            self.workflow = self._build_langgraph_workflow()
            print(f"✓ Orchestrator initialisé avec LangGraph")
        else:
            self.workflow = None
            print(f"✓ Orchestrator initialisé en mode custom (sans LangGraph)")
        
        print(f"  {len(self.agents)} agent(s) disponibles")
        print(f"  Pipeline: {' → '.join(self.pipeline)}")
    
    def _initialize_agents(self):
        """
        Initialise tous les agents disponibles
        """
        # Agent de profilage
        self.agents["profiling"] = {
            "instance": ProfilingAgent(self.blackboard, self.llm_model),
            "status": AgentStatus.PENDING,
            "description": "Analyse le profil utilisateur"
        }
        
        # Agent de planification
        self.agents["path_planning"] = {
            "instance": PathPlanningAgent(self.blackboard, self.llm_model),
            "status": AgentStatus.PENDING,
            "description": "Planifie le parcours d'apprentissage"
        }
        
        # Agent de génération de contenu
        self.agents["content_generator"] = {
            "instance": ContentGenerator(self.blackboard, self.llm_model),
            "status": AgentStatus.PENDING,
            "description": "Génère du contenu pédagogique personnalisé"
        }
        
        # Agent de recommandation
        self.agents["recommendation"] = {
            "instance": RecommendationAgent(self.blackboard, self.llm_model),
            "status": AgentStatus.PENDING,
            "description": "Recommande les meilleures ressources"
        }
        
        # Agent XAI
        self.agents["xai"] = {
            "instance": XAIAgent(self.blackboard, self.llm_model),
            "status": AgentStatus.PENDING,
            "description": "Génère des explications pour toutes les décisions"
        }
    
    def process_user_request(self, user_id: str, request_type: str = "full_analysis") -> Dict:
        """
        Traite une requête utilisateur complète
        
        Args:
            user_id: Identifiant de l'utilisateur
            request_type: Type de requête
                - "full_analysis" : Analyse complète + recommandations
                - "profile_only" : Uniquement le profilage
                - "recommendations" : Uniquement nouvelles recommandations
        
        Returns:
            Résultat complet de l'exécution du pipeline
        """
        print("\n" + "="*80)
        print(f"🚀 ORCHESTRATOR - TRAITEMENT REQUÊTE UTILISATEUR")
        print("="*80)
        print(f"User ID       : {user_id}")
        print(f"Request Type  : {request_type}")
        print(f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Créer un contexte d'exécution
        execution_context = {
            "user_id": user_id,
            "request_type": request_type,
            "started_at": datetime.now().isoformat(),
            "agents_results": {},
            "overall_status": "running"
        }
        
        try:
            # Exécuter le pipeline selon le type de requête
            if request_type == "full_analysis":
                execution_context = self._execute_full_pipeline(user_id, execution_context)
            
            elif request_type == "profile_only":
                execution_context = self._execute_profiling_only(user_id, execution_context)
            
            elif request_type == "recommendations":
                # TODO : À implémenter quand l'agent Recommendation sera créé
                print("⚠️  Type 'recommendations' pas encore implémenté")
                execution_context["overall_status"] = "skipped"
            
            else:
                raise ValueError(f"Type de requête inconnu: {request_type}")
            
            # Marquer comme terminé
            execution_context["completed_at"] = datetime.now().isoformat()
            execution_context["overall_status"] = "completed"
            
        except Exception as e:
            print(f"\n❌ ERREUR DANS L'ORCHESTRATION: {e}")
            execution_context["overall_status"] = "failed"
            execution_context["error"] = str(e)
        
        # Sauvegarder dans l'historique
        self.execution_history.append(execution_context)
        
        # Afficher le résumé
        self._print_execution_summary(execution_context)
        
        return execution_context
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """
        Construit le workflow LangGraph pour l'orchestration des agents
        
        Returns:
            StateGraph configuré
        """
        # Créer le graph
        workflow = StateGraph(AgentState)
        
        # Ajouter les nœuds (chaque agent)
        workflow.add_node("profiling", self._profiling_node)
        workflow.add_node("path_planning", self._path_planning_node)
        workflow.add_node("content_generator", self._content_generator_node)
        workflow.add_node("recommendation", self._recommendation_node)
        workflow.add_node("xai", self._xai_node)
        
        # Définir les edges (flux)
        workflow.set_entry_point("profiling")
        workflow.add_edge("profiling", "path_planning")
        workflow.add_edge("path_planning", "content_generator")
        workflow.add_edge("content_generator", "recommendation")
        workflow.add_edge("recommendation", "xai")
        workflow.add_edge("xai", END)
        
        # Compiler le graph
        return workflow.compile()
    
    # Fonctions de nœud pour chaque agent
    def _profiling_node(self, state: AgentState) -> AgentState:
        """Nœud LangGraph pour le Profiling Agent"""
        print(f"\n{'─'*80}")
        print(f"⚙️  Exécution: PROFILING AGENT")
        print(f"{'─'*80}")
        
        try:
            self.agents["profiling"]["status"] = AgentStatus.RUNNING
            result = self.agents["profiling"]["instance"].analyze_user(state["user_id"])
            
            state["profile"] = result
            state["agent_results"]["profiling"] = {"status": "success", "result": result}
            state["current_step"] = "profiling_complete"
            
            self.agents["profiling"]["status"] = AgentStatus.COMPLETED
            print(f"✅ Profiling Agent terminé")
            
        except Exception as e:
            print(f"❌ Erreur Profiling Agent: {e}")
            state["errors"].append(f"Profiling: {str(e)}")
            state["agent_results"]["profiling"] = {"status": "failed", "error": str(e)}
            self.agents["profiling"]["status"] = AgentStatus.FAILED
        
        return state
    
    def _path_planning_node(self, state: AgentState) -> AgentState:
        """Nœud LangGraph pour le Path Planning Agent"""
        print(f"\n{'─'*80}")
        print(f"⚙️  Exécution: PATH PLANNING AGENT")
        print(f"{'─'*80}")
        
        try:
            self.agents["path_planning"]["status"] = AgentStatus.RUNNING
            result = self.agents["path_planning"]["instance"].plan_learning_path(state["user_id"])
            
            state["learning_path"] = result
            state["agent_results"]["path_planning"] = {"status": "success", "result": result}
            state["current_step"] = "path_planning_complete"
            
            self.agents["path_planning"]["status"] = AgentStatus.COMPLETED
            print(f"✅ Path Planning Agent terminé")
            
        except Exception as e:
            print(f"❌ Erreur Path Planning Agent: {e}")
            state["errors"].append(f"Path Planning: {str(e)}")
            state["agent_results"]["path_planning"] = {"status": "failed", "error": str(e)}
            self.agents["path_planning"]["status"] = AgentStatus.FAILED
        
        return state
    
    def _content_generator_node(self, state: AgentState) -> AgentState:
        """Nœud LangGraph pour le Content Generator"""
        print(f"\n{'─'*80}")
        print(f"⚙️  Exécution: CONTENT GENERATOR")
        print(f"{'─'*80}")
        
        try:
            self.agents["content_generator"]["status"] = AgentStatus.RUNNING
            
            # Générer du contenu pour la première étape du parcours
            if state.get("learning_path") and len(state["learning_path"].get('path', [])) > 0:
                result = self.agents["content_generator"]["instance"].generate_for_learning_path(
                    state["user_id"], step_number=1
                )
            else:
                result = {"skipped": "No learning path available"}
            
            state["generated_content"] = result
            state["agent_results"]["content_generator"] = {"status": "success", "result": result}
            state["current_step"] = "content_generation_complete"
            
            self.agents["content_generator"]["status"] = AgentStatus.COMPLETED
            print(f"✅ Content Generator terminé")
            
        except Exception as e:
            print(f"❌ Erreur Content Generator: {e}")
            state["errors"].append(f"Content Generator: {str(e)}")
            state["agent_results"]["content_generator"] = {"status": "failed", "error": str(e)}
            self.agents["content_generator"]["status"] = AgentStatus.FAILED
        
        return state
    
    def _recommendation_node(self, state: AgentState) -> AgentState:
        """Nœud LangGraph pour le Recommendation Agent"""
        print(f"\n{'─'*80}")
        print(f"⚙️  Exécution: RECOMMENDATION AGENT")
        print(f"{'─'*80}")
        
        try:
            self.agents["recommendation"]["status"] = AgentStatus.RUNNING
            result = self.agents["recommendation"]["instance"].generate_recommendations(
                state["user_id"], top_k=5
            )
            
            state["recommendations"] = result
            state["agent_results"]["recommendation"] = {"status": "success", "result": result}
            state["current_step"] = "recommendation_complete"
            
            self.agents["recommendation"]["status"] = AgentStatus.COMPLETED
            print(f"✅ Recommendation Agent terminé")
            
        except Exception as e:
            print(f"❌ Erreur Recommendation Agent: {e}")
            state["errors"].append(f"Recommendation: {str(e)}")
            state["agent_results"]["recommendation"] = {"status": "failed", "error": str(e)}
            self.agents["recommendation"]["status"] = AgentStatus.FAILED
        
        return state
    
    def _xai_node(self, state: AgentState) -> AgentState:
        """Nœud LangGraph pour le XAI Agent"""
        print(f"\n{'─'*80}")
        print(f"⚙️  Exécution: XAI AGENT")
        print(f"{'─'*80}")
        
        try:
            self.agents["xai"]["status"] = AgentStatus.RUNNING
            result = self.agents["xai"]["instance"].explain_full_system(state["user_id"])
            
            state["explanations"] = result
            state["agent_results"]["xai"] = {"status": "success", "result": result}
            state["current_step"] = "xai_complete"
            
            self.agents["xai"]["status"] = AgentStatus.COMPLETED
            print(f"✅ XAI Agent terminé")
            
        except Exception as e:
            print(f"❌ Erreur XAI Agent: {e}")
            state["errors"].append(f"XAI: {str(e)}")
            state["agent_results"]["xai"] = {"status": "failed", "error": str(e)}
            self.agents["xai"]["status"] = AgentStatus.FAILED
        
        return state
    
    def _execute_full_pipeline(self, user_id: str, context: Dict) -> Dict:
        """
        Exécute le pipeline complet
        
        Args:
            user_id: ID utilisateur
            context: Contexte d'exécution
        
        Returns:
            Contexte mis à jour avec les résultats
        """
        print("\n📋 Exécution du pipeline complet...")
        
        for agent_name in self.pipeline:
            print(f"\n{'─'*80}")
            print(f"⚙️  Exécution de l'agent: {agent_name.upper()}")
            print(f"{'─'*80}")
            
            # Marquer l'agent comme en cours
            self.agents[agent_name]["status"] = AgentStatus.RUNNING
            
            try:
                # Exécuter l'agent
                result = self._execute_agent(agent_name, user_id)
                
                # Sauvegarder le résultat
                context["agents_results"][agent_name] = {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Marquer comme terminé
                self.agents[agent_name]["status"] = AgentStatus.COMPLETED
                print(f"✅ Agent {agent_name} terminé avec succès")
                
            except Exception as e:
                print(f"❌ Erreur dans l'agent {agent_name}: {e}")
                context["agents_results"][agent_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.agents[agent_name]["status"] = AgentStatus.FAILED
                
                # Décider si on continue ou on arrête
                # Pour l'instant, on arrête en cas d'erreur
                raise
        
        return context
    
    def _execute_profiling_only(self, user_id: str, context: Dict) -> Dict:
        """
        Exécute uniquement l'agent de profilage
        
        Args:
            user_id: ID utilisateur
            context: Contexte d'exécution
        
        Returns:
            Contexte mis à jour
        """
        print("\n📋 Exécution du profilage uniquement...")
        
        # ✅ CORRECTION : Marquer l'agent comme en cours
        self.agents["profiling"]["status"] = AgentStatus.RUNNING
        
        try:
            result = self._execute_agent("profiling", user_id)
            context["agents_results"]["profiling"] = {
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            # ✅ CORRECTION : Marquer comme terminé avec succès
            self.agents["profiling"]["status"] = AgentStatus.COMPLETED
            
        except Exception as e:
            context["agents_results"]["profiling"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            # ✅ CORRECTION : Marquer comme échoué
            self.agents["profiling"]["status"] = AgentStatus.FAILED
            
            raise
        
        return context
    
    def _execute_agent(self, agent_name: str, user_id: str) -> Any:
        """
        Exécute un agent spécifique
        
        Args:
            agent_name: Nom de l'agent à exécuter
            user_id: ID utilisateur
        
        Returns:
            Résultat de l'agent
        """
        agent = self.agents.get(agent_name)
        
        if not agent:
            raise ValueError(f"Agent '{agent_name}' non trouvé")
        
        agent_instance = agent["instance"]
        
        # Exécuter selon le type d'agent
        if agent_name == "profiling":
            return agent_instance.analyze_user(user_id)
        
        elif agent_name == "path_planning":
            return agent_instance.plan_learning_path(user_id)
        
        elif agent_name == "content_generator":
            # Générer du contenu pour la première étape du parcours
            path = self.blackboard.read("learning_paths", user_id)
            if path and len(path['path']) > 0:
                return agent_instance.generate_for_learning_path(user_id, step_number=1)
            else:
                return {"skipped": "No learning path available"}
        
        elif agent_name == "recommendation":
            return agent_instance.generate_recommendations(user_id, top_k=5)
        
        elif agent_name == "xai":
            return agent_instance.explain_full_system(user_id)
        
        else:
            raise ValueError(f"Type d'agent inconnu: {agent_name}")
    
    def _print_execution_summary(self, context: Dict):
        """
        Affiche un résumé de l'exécution
        
        Args:
            context: Contexte d'exécution
        """
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DE L'EXÉCUTION")
        print("="*80)
        
        print(f"User ID         : {context['user_id']}")
        print(f"Request Type    : {context['request_type']}")
        print(f"Overall Status  : {context['overall_status'].upper()}")
        
        if "completed_at" in context:
            start = datetime.fromisoformat(context['started_at'])
            end = datetime.fromisoformat(context['completed_at'])
            duration = (end - start).total_seconds()
            print(f"Duration        : {duration:.2f} seconds")
        
        print(f"\nAgents exécutés : {len(context['agents_results'])}")
        for agent_name, result in context['agents_results'].items():
            status_emoji = "✅" if result['status'] == "success" else "❌"
            print(f"  {status_emoji} {agent_name:20s} : {result['status']}")
        
        print("="*80)
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """
        Obtient le statut d'un agent
        
        Args:
            agent_name: Nom de l'agent
        
        Returns:
            Statut de l'agent ou None si non trouvé
        """
        agent = self.agents.get(agent_name)
        return agent["status"] if agent else None
    
    def get_execution_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Récupère l'historique des exécutions
        
        Args:
            user_id: Filtrer par utilisateur (optionnel)
        
        Returns:
            Liste des exécutions
        """
        if user_id:
            return [ex for ex in self.execution_history if ex["user_id"] == user_id]
        return self.execution_history
    
    def reset_agents(self):
        """
        Réinitialise tous les agents à l'état PENDING
        """
        for agent_name in self.agents:
            self.agents[agent_name]["status"] = AgentStatus.PENDING
        print("✓ Tous les agents réinitialisés")
    
    def get_pipeline_info(self) -> Dict:
        """
        Obtient des informations sur le pipeline
        
        Returns:
            Dict avec les infos du pipeline
        """
        return {
            "pipeline": self.pipeline,
            "agents_available": list(self.agents.keys()),
            "agents_status": {
                name: agent["status"].value 
                for name, agent in self.agents.items()
            },
            "total_executions": len(self.execution_history),
            "orchestration_method": "LangGraph" if self.use_langgraph else "Custom",
            "langgraph_available": LANGGRAPH_AVAILABLE
        }
    
    def __repr__(self) -> str:
        """
        Représentation textuelle de l'orchestrateur
        """
        return (f"Orchestrator(agents={len(self.agents)}, "
                f"pipeline_steps={len(self.pipeline)}, "
                f"executions={len(self.execution_history)})")