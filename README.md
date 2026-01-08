# Explainable Multi-Agent Generative Recommendation System for Personalized Learning

A modular multi-agent system that provides personalized learning recommendations with explainable AI, using LLMs, graph search, reinforcement learning, and hybrid filtering.

## Project Overview

This system analyzes learner behavior, plans optimal learning paths, generates personalized content, and provides recommendations with transparent explanations. It's built on a **Blackboard architecture** where specialized agents coordinate to deliver a complete learning experience.

## Architecture

## Agents

### 1. **Profiling Agent**
- **Role**: Analyze user behavior and build learner profiles
- **Technologies**: 
  - Embeddings (`sentence-transformers`)
  - K-Means clustering
  - LLM profiling (Ollama)
- **Outputs**: Learning style, proficiency level, interests, strengths/weaknesses

### 2. **Path Planning Agent**
- **Role**: Plan optimal learning paths
- **Technologies**:
  - A* graph search
  - Q-Learning reinforcement learning
  - Heuristics (level progression, style preferences)
- **Outputs**: Ordered sequence of learning resources

### 3. **Content Generator**
- **Role**: Generate personalized educational content
- **Technologies**:
  - RAG (Retrieval-Augmented Generation)
  - LLM generation (Ollama)
- **Outputs**: Lessons, exercises, quizzes (course/video/article formats)

### 4. **Recommendation Agent**
- **Role**: Recommend best next learning resources
- **Technologies**:
  - Hybrid filtering (path-based + content-based + collaborative)
  - LLM ranking
- **Outputs**: Top-K ranked recommendations with explanations

### 5. **XAI Agent**
- **Role**: Explain all system decisions
- **Technologies**:
  - SHAP (feature importance)
  - LIME (local explanations)
  - Counterfactual reasoning
  - LLM-based reasoning chains
- **Outputs**: Profile explanations, path rationale, recommendation justifications

## Dataset

- **Primary**: OULAD (Open University Learning Analytics Dataset)
  - `data/raw/*.csv`: studentInfo, studentAssessment, courses, VLE interactions
  - Loader: `data/oulad_loader.py`
- **Demo**: Synthetic users (Alice, Bob, Charlie) in `main.py`

## Evaluation Metrics

- **Recommendation Quality**: NDCG@K, MRR, Recall@K, Precision@K, MAP
- **Content Quality**: ROUGE-N, ROUGE-L, BERTScore-like similarity
- **Explainability**: Faithfulness, Plausibility, Completeness, Trust Score

## Installation

### Prerequisites
- Python 3.9 - 3.13 (NOT 3.14+ due to Pydantic compatibility)
- Conda or virtualenv
- Ollama with `llama3.2:3b` model installed
### Setup

**Clone the repository**

git clone https://github.com/<your-username>/multiagent-learning-xai.git
cd multiagent-learning-xai


multiagent_learning/
├── agents/                      # Intelligent agents
│   ├── profiling_agent.py
│   ├── path_planning_agent.py
│   ├── content_generator.py
│   ├── recommendation_agent.py
│   └── xai_agent.py
├── memory/
│   └── blackboard.py           # Shared memory
├── orchestrator/
│   └── orchestrator.py         # Agent coordination
├── data/
│   ├── raw/                    # OULAD CSV files
│   ├── oulad_loader.py         # Dataset loader
│   └── blackboard_export.json  # Demo output
├── evaluation/                 # Metrics & evaluation
│   ├── recommendation_metrics.py
│   ├── generation_metrics.py
│   ├── xai_metrics.py
│   └── system_evaluation.py
├── tests/                      # 73 automated tests
├── utils/
│   ├── embeddings.py           # Sentence transformers
│   └── oulad_integration.py
├── main.py                     # Demo script
└── requirements.txt

conda create -n multiagent-learning python=3.11 -y
conda activate multiagent-learning
cd explainable-multiagent-learning

