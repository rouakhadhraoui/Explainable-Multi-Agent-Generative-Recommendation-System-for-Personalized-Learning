# memory/blackboard.py
"""
Shared Memory (Blackboard) - Mémoire partagée entre tous les agents

Ce module implémente un système de mémoire centralisée où :
- Chaque agent peut LIRE et ÉCRIRE des données
- L'historique des interactions est stocké
- Les profils utilisateurs sont sauvegardés
- Les contenus générés sont mis en cache

Architecture : Dictionnaire Python avec sections organisées
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json


class Blackboard:
    """
    Tableau noir partagé - Mémoire centrale du système multi-agents
    """
    
    def __init__(self):
        """
        Initialise le blackboard avec des sections vides
        """
        # Structure de données principale
        self.data = {
            # Profils des utilisateurs (user_id -> profil)
            "profiles": {},
            
            # Historique des interactions (user_id -> liste d'interactions)
            "history": {},
            
            # Contenus générés en cache (content_id -> contenu)
            "cached_content": {},
            
            # Parcours d'apprentissage planifiés (user_id -> path)
            "learning_paths": {},
            
            # Recommandations générées (user_id -> liste de recommandations)
            "recommendations": {},
            
            # Explications XAI (recommendation_id -> explication)
            "explanations": {},
            
            # Métadonnées système
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    # ========== MÉTHODES GÉNÉRIQUES ==========
    
    def write(self, section: str, key: str, value: Any) -> bool:
        """
        Écrire une donnée dans une section du blackboard
        
        Args:
            section: Nom de la section (ex: "profiles", "history")
            key: Clé unique (ex: user_id, content_id)
            value: Valeur à stocker (dict, list, str, etc.)
        
        Returns:
            True si l'écriture a réussi
        """
        try:
            # Vérifier que la section existe
            if section not in self.data:
                print(f"⚠️  Section '{section}' n'existe pas. Création automatique.")
                self.data[section] = {}
            
            # Écrire la valeur
            self.data[section][key] = value
            
            # Mettre à jour le timestamp
            self.data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            print(f"✓ Écriture dans [{section}][{key}] réussie")
            return True
            
        except Exception as e:
            print(f"✗ Erreur d'écriture dans [{section}][{key}]: {e}")
            return False
    
    def read(self, section: str, key: str) -> Optional[Any]:
        """
        Lire une donnée depuis une section du blackboard
        
        Args:
            section: Nom de la section
            key: Clé à lire
        
        Returns:
            La valeur stockée, ou None si inexistante
        """
        try:
            # Vérifier que la section existe
            if section not in self.data:
                print(f"⚠️  Section '{section}' n'existe pas")
                return None
            
            # Lire la valeur
            value = self.data[section].get(key, None)
            
            if value is None:
                print(f"⚠️  Clé '{key}' introuvable dans [{section}]")
            else:
                print(f"✓ Lecture de [{section}][{key}] réussie")
            
            return value
            
        except Exception as e:
            print(f"✗ Erreur de lecture dans [{section}][{key}]: {e}")
            return None
    
    def read_section(self, section: str) -> Dict:
        """
        Lire toute une section du blackboard
        
        Args:
            section: Nom de la section
        
        Returns:
            Dictionnaire complet de la section
        """
        if section in self.data:
            return self.data[section]
        else:
            print(f"⚠️  Section '{section}' n'existe pas")
            return {}
    
    def delete(self, section: str, key: str) -> bool:
        """
        Supprimer une entrée du blackboard
        
        Args:
            section: Nom de la section
            key: Clé à supprimer
        
        Returns:
            True si la suppression a réussi
        """
        try:
            if section in self.data and key in self.data[section]:
                del self.data[section][key]
                self.data["metadata"]["last_updated"] = datetime.now().isoformat()
                print(f"✓ Suppression de [{section}][{key}] réussie")
                return True
            else:
                print(f"⚠️  [{section}][{key}] introuvable")
                return False
        except Exception as e:
            print(f"✗ Erreur de suppression: {e}")
            return False
    
    # ========== MÉTHODES SPÉCIFIQUES ==========
    
    def add_to_history(self, user_id: str, interaction: Dict) -> bool:
        """
        Ajouter une interaction à l'historique d'un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            interaction: Dict contenant les détails de l'interaction
                        (ex: {"type": "click", "resource_id": "123", "timestamp": "..."})
        
        Returns:
            True si ajout réussi
        """
        try:
            # Initialiser l'historique si inexistant
            if user_id not in self.data["history"]:
                self.data["history"][user_id] = []
            
            # Ajouter le timestamp si absent
            if "timestamp" not in interaction:
                interaction["timestamp"] = datetime.now().isoformat()
            
            # Ajouter l'interaction
            self.data["history"][user_id].append(interaction)
            
            print(f"✓ Interaction ajoutée pour user {user_id}")
            return True
            
        except Exception as e:
            print(f"✗ Erreur ajout historique: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Récupérer l'historique d'un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre max d'interactions à retourner (None = tout)
        
        Returns:
            Liste des interactions (les plus récentes en premier si limit)
        """
        history = self.data["history"].get(user_id, [])
        
        if limit:
            return history[-limit:]  # Dernières N interactions
        return history
    
    # ========== MÉTHODES UTILITAIRES ==========
    
    def clear_all(self):
        """
        Vider complètement le blackboard (utile pour les tests)
        """
        self.__init__()
        print("⚠️  Blackboard réinitialisé")
    
    def get_stats(self) -> Dict:
        """
        Obtenir des statistiques sur le contenu du blackboard
        
        Returns:
            Dict avec le nombre d'éléments par section
        """
        stats = {}
        for section, content in self.data.items():
            if isinstance(content, dict):
                stats[section] = len(content)
            elif isinstance(content, list):
                stats[section] = len(content)
            else:
                stats[section] = "N/A"
        
        return stats
    
    def export_to_json(self, filepath: str) -> bool:
        """
        Exporter le blackboard vers un fichier JSON
        
        Args:
            filepath: Chemin du fichier de sortie
        
        Returns:
            True si export réussi
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print(f"✓ Blackboard exporté vers {filepath}")
            return True
        except Exception as e:
            print(f"✗ Erreur export: {e}")
            return False
    
    def __repr__(self) -> str:
        """
        Représentation textuelle du blackboard
        """
        stats = self.get_stats()
        return f"Blackboard(sections={list(self.data.keys())}, stats={stats})"