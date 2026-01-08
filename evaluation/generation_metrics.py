# evaluation/generation_metrics.py
"""
Métriques d'évaluation pour la génération de contenu

Implémente les métriques standard :
- BERTScore (similarité sémantique)
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU (optionnel)
- Métriques de lisibilité

Ces métriques évaluent la qualité du contenu généré par le LLM
en le comparant à des références de qualité.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


class GenerationMetrics:
    """
    Calcule les métriques d'évaluation pour la génération de contenu
    """
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize le texte en mots
        
        Args:
            text: Texte à tokenizer
        
        Returns:
            Liste de tokens (mots en minuscules)
        """
        # Nettoyer et tokenizer
        text = text.lower()
        # Garder les lettres, chiffres et espaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        return tokens
    
    @staticmethod
    def get_ngrams(tokens: List[str], n: int) -> List[Tuple]:
        """
        Extrait les n-grams d'une liste de tokens
        
        Args:
            tokens: Liste de tokens
            n: Taille des n-grams
        
        Returns:
            Liste de n-grams (tuples)
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    @staticmethod
    def rouge_n(generated: str, reference: str, n: int = 1) -> Dict[str, float]:
        """
        Calcule le ROUGE-N (overlap de n-grams)
        
        ROUGE mesure le chevauchement entre le texte généré et la référence.
        ROUGE-1 : unigrammes, ROUGE-2 : bigrammes, etc.
        
        Args:
            generated: Texte généré
            reference: Texte de référence
            n: Taille des n-grams (1, 2, etc.)
        
        Returns:
            Dict avec precision, recall, f1
        """
        gen_tokens = GenerationMetrics.tokenize(generated)
        ref_tokens = GenerationMetrics.tokenize(reference)
        
        if not gen_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Extraire les n-grams
        gen_ngrams = GenerationMetrics.get_ngrams(gen_tokens, n)
        ref_ngrams = GenerationMetrics.get_ngrams(ref_tokens, n)
        
        if not gen_ngrams or not ref_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Compter les n-grams
        gen_counter = Counter(gen_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        # Calculer le chevauchement
        overlap = sum((gen_counter & ref_counter).values())
        
        # Precision : overlap / nombre de n-grams générés
        precision = overlap / len(gen_ngrams) if gen_ngrams else 0.0
        
        # Recall : overlap / nombre de n-grams de référence
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        
        # F1-score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def rouge_l(generated: str, reference: str) -> Dict[str, float]:
        """
        Calcule le ROUGE-L (Longest Common Subsequence)
        
        ROUGE-L mesure la plus longue sous-séquence commune.
        
        Args:
            generated: Texte généré
            reference: Texte de référence
        
        Returns:
            Dict avec precision, recall, f1
        """
        gen_tokens = GenerationMetrics.tokenize(generated)
        ref_tokens = GenerationMetrics.tokenize(reference)
        
        if not gen_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculer LCS (Longest Common Subsequence)
        m, n = len(gen_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Precision, Recall, F1
        precision = lcs_length / len(gen_tokens) if gen_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def bertscore_simple(generated: str, reference: str, 
                        embedding_model=None) -> float:
        """
        Calcule une version simplifiée de BERTScore
        
        BERTScore utilise les embeddings pour mesurer la similarité sémantique.
        Cette version utilise la similarité cosinus moyenne des embeddings.
        
        Args:
            generated: Texte généré
            reference: Texte de référence
            embedding_model: Modèle d'embeddings (optionnel)
        
        Returns:
            Score de similarité entre 0 et 1
        """
        try:
            from utils.embeddings import get_embedding_generator
            
            if embedding_model is None:
                embedding_model = get_embedding_generator()
            
            # Générer les embeddings
            gen_emb = embedding_model.encode(generated)
            ref_emb = embedding_model.encode(reference)
            
            # Calculer la similarité cosinus
            similarity = embedding_model.cosine_similarity(gen_emb, ref_emb)
            
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️  Erreur BERTScore: {e}")
            return 0.0
    
    @staticmethod
    def readability_score(text: str) -> Dict[str, float]:
        """
        Calcule des métriques de lisibilité
        
        Args:
            text: Texte à analyser
        
        Returns:
            Dict avec différentes métriques de lisibilité
        """
        tokens = GenerationMetrics.tokenize(text)
        
        if not tokens:
            return {
                'avg_word_length': 0.0,
                'total_words': 0,
                'unique_words': 0,
                'lexical_diversity': 0.0
            }
        
        # Nombre total de mots
        total_words = len(tokens)
        
        # Mots uniques
        unique_words = len(set(tokens))
        
        # Longueur moyenne des mots
        avg_word_length = np.mean([len(word) for word in tokens])
        
        # Diversité lexicale (Type-Token Ratio)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        return {
            'avg_word_length': float(avg_word_length),
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': float(lexical_diversity)
        }
    
    @staticmethod
    def coherence_score(text: str) -> float:
        """
        Mesure la cohérence du texte (heuristique simple)
        
        Vérifie la présence de connecteurs logiques et la structure
        
        Args:
            text: Texte à analyser
        
        Returns:
            Score de cohérence entre 0 et 1
        """
        text_lower = text.lower()
        
        # Connecteurs logiques
        connectors = [
            'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'thus', 'hence',
            'first', 'second', 'finally', 'in conclusion',
            'for example', 'for instance', 'such as'
        ]
        
        # Compter les connecteurs présents
        connectors_found = sum(1 for conn in connectors if conn in text_lower)
        
        # Score basé sur le nombre de connecteurs (normalisé)
        score = min(connectors_found / 5.0, 1.0)  # Max 5 connecteurs
        
        return score
    
    @classmethod
    def evaluate_all(cls, generated: str, reference: str,
                    embedding_model=None) -> Dict[str, float]:
        """
        Évalue toutes les métriques de génération
        
        Args:
            generated: Texte généré
            reference: Texte de référence
            embedding_model: Modèle d'embeddings (optionnel)
        
        Returns:
            Dict contenant toutes les métriques
        """
        results = {}
        
        # ROUGE-1
        rouge1 = cls.rouge_n(generated, reference, n=1)
        results['ROUGE-1_precision'] = rouge1['precision']
        results['ROUGE-1_recall'] = rouge1['recall']
        results['ROUGE-1_f1'] = rouge1['f1']
        
        # ROUGE-2
        rouge2 = cls.rouge_n(generated, reference, n=2)
        results['ROUGE-2_precision'] = rouge2['precision']
        results['ROUGE-2_recall'] = rouge2['recall']
        results['ROUGE-2_f1'] = rouge2['f1']
        
        # ROUGE-L
        rougeL = cls.rouge_l(generated, reference)
        results['ROUGE-L_precision'] = rougeL['precision']
        results['ROUGE-L_recall'] = rougeL['recall']
        results['ROUGE-L_f1'] = rougeL['f1']
        
        # BERTScore
        results['BERTScore'] = cls.bertscore_simple(generated, reference, embedding_model)
        
        # Lisibilité
        readability = cls.readability_score(generated)
        results['avg_word_length'] = readability['avg_word_length']
        results['lexical_diversity'] = readability['lexical_diversity']
        
        # Cohérence
        results['coherence'] = cls.coherence_score(generated)
        
        return results
    
    @classmethod
    def evaluate_batch(cls, batch_data: List[Tuple[str, str]],
                      embedding_model=None) -> Dict[str, float]:
        """
        Évalue un batch de textes générés
        
        Args:
            batch_data: Liste de tuples (generated, reference)
            embedding_model: Modèle d'embeddings (optionnel)
        
        Returns:
            Dict des métriques moyennes
        """
        all_results = []
        
        for generated, reference in batch_data:
            results = cls.evaluate_all(generated, reference, embedding_model)
            all_results.append(results)
        
        # Calculer les moyennes
        avg_results = {}
        
        if all_results:
            for metric in all_results[0].keys():
                values = [r[metric] for r in all_results]
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results


def print_generation_metrics_report(metrics: Dict[str, float], 
                                   title: str = "Métriques de Génération"):
    """
    Affiche un rapport formaté des métriques de génération
    
    Args:
        metrics: Dictionnaire des métriques
        title: Titre du rapport
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # ROUGE
    rouge_metrics = {k: v for k, v in metrics.items() if 'ROUGE' in k and '_std' not in k}
    if rouge_metrics:
        print(f"\n📝 ROUGE (Recall-Oriented Understudy for Gisting Evaluation):")
        for metric, value in sorted(rouge_metrics.items()):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:25s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:25s} : {value:.4f}")
    
    # BERTScore
    if 'BERTScore' in metrics:
        print(f"\n🤖 BERTScore (Similarité Sémantique):")
        std_key = 'BERTScore_std'
        if std_key in metrics:
            print(f"  BERTScore              : {metrics['BERTScore']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  BERTScore              : {metrics['BERTScore']:.4f}")
    
    # Lisibilité
    readability = {k: v for k, v in metrics.items() 
                   if k in ['avg_word_length', 'lexical_diversity'] and '_std' not in k}
    if readability:
        print(f"\n📖 Lisibilité:")
        for metric, value in readability.items():
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"  {metric:25s} : {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {metric:25s} : {value:.4f}")
    
    # Cohérence
    if 'coherence' in metrics:
        print(f"\n🔗 Cohérence:")
        std_key = 'coherence_std'
        if std_key in metrics:
            print(f"  Coherence              : {metrics['coherence']:.4f} ± {metrics[std_key]:.4f}")
        else:
            print(f"  Coherence              : {metrics['coherence']:.4f}")
    
    print(f"\n{'='*70}\n")