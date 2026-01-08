# evaluation/visualize_results.py
"""
Visualisation des Résultats d'Évaluation

Génère des graphiques et tableaux pour publication scientifique :
- Graphiques comparatifs des métriques
- Tableaux de résultats
- Diagrammes pour le papier de recherche
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns


class ResultsVisualizer:
    """
    Crée des visualisations des résultats d'évaluation
    """
    
    def __init__(self, results_file: str):
        """
        Initialise le visualiseur
        
        Args:
            results_file: Chemin vers le fichier JSON des résultats
        """
        self.results_file = results_file
        
        # Charger les résultats
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Configurer le style des graphiques
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
        
        # Dossier de sortie
        self.output_dir = os.path.join(os.path.dirname(results_file), "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Visualiseur initialisé")
        print(f"  Résultats: {results_file}")
        print(f"  Figures: {self.output_dir}")
    
    def plot_recommendation_metrics(self):
        """
        Graphique des métriques de recommandation
        """
        print("\n📊 Création du graphique: Métriques de Recommandation...")
        
        rec_results = self.results.get('recommendations', {})
        
        if not rec_results:
            print("  ⚠️  Pas de données de recommandation")
            return
        
        # Extraire les métriques
        metrics_data = {
            'NDCG@5': rec_results.get('NDCG@5', 0),
            'NDCG@10': rec_results.get('NDCG@10', 0),
            'MRR': rec_results.get('MRR', 0),
            'Recall@5': rec_results.get('Recall@5', 0),
            'Recall@10': rec_results.get('Recall@10', 0),
            'Precision@5': rec_results.get('Precision@5', 0),
        }
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        bars = ax.bar(metrics, values, color=['#3498db', '#3498db', '#e74c3c', 
                                               '#2ecc71', '#2ecc71', '#9b59b6'])
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Métriques de Recommandation', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'recommendation_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Sauvegardé: {filename}")
    
    def plot_generation_metrics(self):
        """
        Graphique des métriques de génération
        """
        print("\n📊 Création du graphique: Métriques de Génération...")
        
        gen_results = self.results.get('generation', {})
        
        if not gen_results:
            print("  ⚠️  Pas de données de génération")
            return
        
        # Extraire les métriques ROUGE
        rouge_data = {
            'ROUGE-1': gen_results.get('ROUGE-1_f1', 0),
            'ROUGE-2': gen_results.get('ROUGE-2_f1', 0),
            'ROUGE-L': gen_results.get('ROUGE-L_f1', 0),
        }
        
        # BERTScore et autres
        other_data = {
            'BERTScore': gen_results.get('BERTScore', 0),
            'Coherence': gen_results.get('coherence', 0),
            'Lexical Diversity': gen_results.get('lexical_diversity', 0),
        }
        
        # Créer un graphique avec deux subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1 : ROUGE
        metrics1 = list(rouge_data.keys())
        values1 = list(rouge_data.values())
        
        bars1 = ax1.bar(metrics1, values1, color=['#e74c3c', '#e67e22', '#f39c12'])
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax1.set_title('ROUGE Metrics', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2 : Autres métriques
        metrics2 = list(other_data.keys())
        values2 = list(other_data.values())
        
        bars2 = ax2.bar(metrics2, values2, color=['#3498db', '#9b59b6', '#2ecc71'])
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Other Quality Metrics', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'generation_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Sauvegardé: {filename}")
    
    def plot_xai_metrics(self):
        """
        Graphique des métriques XAI
        """
        print("\n📊 Création du graphique: Métriques XAI...")
        
        xai_results = self.results.get('xai', {})
        
        if not xai_results:
            print("  ⚠️  Pas de données XAI")
            return
        
        # Extraire les métriques principales
        metrics_data = {
            'Faithfulness': xai_results.get('faithfulness', 0),
            'Plausibility': xai_results.get('plausibility', 0),
            'Trust Score': xai_results.get('trust_score', 0),
            'Consistency': xai_results.get('consistency', 0),
            'Contrastive\nQuality': xai_results.get('contrastive_quality', 0),
        }
        
        # Créer le graphique radar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        # Fermer le polygone
        values += values[:1]
        
        # Angles pour chaque métrique
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        # Tracer
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', label='XAI Metrics')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Configurer
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.set_title('XAI Metrics (Explicabilité)', fontsize=14, fontweight='bold', pad=30)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'xai_metrics_radar.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Sauvegardé: {filename}")
    
    def plot_overall_comparison(self):
        """
        Graphique de comparaison globale
        """
        print("\n📊 Création du graphique: Comparaison Globale...")
        
        # Sélectionner des métriques clés
        key_metrics = {
            'Recommendations\n(NDCG@10)': self.results.get('recommendations', {}).get('NDCG@10', 0),
            'Recommendations\n(MRR)': self.results.get('recommendations', {}).get('MRR', 0),
            'Generation\n(ROUGE-1)': self.results.get('generation', {}).get('ROUGE-1_f1', 0),
            'Generation\n(BERTScore)': self.results.get('generation', {}).get('BERTScore', 0),
            'XAI\n(Faithfulness)': self.results.get('xai', {}).get('faithfulness', 0),
            'XAI\n(Trust Score)': self.results.get('xai', {}).get('trust_score', 0),
        }
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        metrics = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        # Couleurs par catégorie
        colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Vue d\'Ensemble des Performances du Système', fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Recommendations'),
            Patch(facecolor='#e74c3c', label='Generation'),
            Patch(facecolor='#2ecc71', label='XAI')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'overall_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Sauvegardé: {filename}")
    
    def generate_latex_table(self):
        """
        Génère un tableau LaTeX pour la publication
        """
        print("\n📄 Génération du tableau LaTeX...")
        
        latex = r"""\begin{table}[h]
\centering
\caption{Résultats d'Évaluation du Système Multi-Agents}
\label{tab:results}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Catégorie} & \textbf{Métrique} & \textbf{Score} \\
\hline
\hline
"""
        
        # Recommandations
        rec = self.results.get('recommendations', {})
        if rec:
            latex += r"\multirow{4}{*}{Recommandations} "
            latex += f"& NDCG@10 & {rec.get('NDCG@10', 0):.4f} \\\\\n"
            latex += f"& MRR & {rec.get('MRR', 0):.4f} \\\\\n"
            latex += f"& Recall@10 & {rec.get('Recall@10', 0):.4f} \\\\\n"
            latex += f"& Precision@10 & {rec.get('Precision@10', 0):.4f} \\\\\n"
            latex += r"\hline" + "\n"
        
        # Génération
        gen = self.results.get('generation', {})
        if gen:
            latex += r"\multirow{3}{*}{Génération} "
            latex += f"& ROUGE-1 F1 & {gen.get('ROUGE-1_f1', 0):.4f} \\\\\n"
            latex += f"& ROUGE-L F1 & {gen.get('ROUGE-L_f1', 0):.4f} \\\\\n"
            latex += f"& BERTScore & {gen.get('BERTScore', 0):.4f} \\\\\n"
            latex += r"\hline" + "\n"
        
        # XAI
        xai = self.results.get('xai', {})
        if xai:
            latex += r"\multirow{4}{*}{XAI} "
            latex += f"& Faithfulness & {xai.get('faithfulness', 0):.4f} \\\\\n"
            latex += f"& Plausibility & {xai.get('plausibility', 0):.4f} \\\\\n"
            latex += f"& Trust Score & {xai.get('trust_score', 0):.4f} \\\\\n"
            latex += f"& Consistency & {xai.get('consistency', 0):.4f} \\\\\n"
            latex += r"\hline" + "\n"
        
        latex += r"""\end{tabular}
\end{table}
"""
        
        # Sauvegarder
        filename = os.path.join(self.output_dir, 'results_table.tex')
        with open(filename, 'w') as f:
            f.write(latex)
        
        print(f"  ✓ Sauvegardé: {filename}")
    
    def generate_all_visualizations(self):
        """
        Génère toutes les visualisations
        """
        print(f"\n{'='*70}")
        print(f"📊 GÉNÉRATION DE TOUTES LES VISUALISATIONS")
        print(f"{'='*70}")
        
        self.plot_recommendation_metrics()
        self.plot_generation_metrics()
        self.plot_xai_metrics()
        self.plot_overall_comparison()
        self.generate_latex_table()
        
        print(f"\n{'='*70}")
        print(f"✅ TOUTES LES VISUALISATIONS CRÉÉES !")
        print(f"{'='*70}")
        print(f"\n📁 Dossier de sortie: {self.output_dir}")
        print(f"\n📊 Fichiers générés:")
        print(f"  • recommendation_metrics.png")
        print(f"  • generation_metrics.png")
        print(f"  • xai_metrics_radar.png")
        print(f"  • overall_comparison.png")
        print(f"  • results_table.tex (pour LaTeX)\n")


def visualize_latest_results():
    """
    Visualise les résultats les plus récents
    """
    results_dir = "evaluation/results"
    
    # Trouver le fichier le plus récent
    files = [f for f in os.listdir(results_dir) if f.startswith('evaluation_results_') and f.endswith('.json')]
    
    if not files:
        print("❌ Aucun fichier de résultats trouvé")
        return
    
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    filepath = os.path.join(results_dir, latest_file)
    
    print(f"📊 Visualisation de: {latest_file}")
    
    visualizer = ResultsVisualizer(filepath)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    visualize_latest_results()