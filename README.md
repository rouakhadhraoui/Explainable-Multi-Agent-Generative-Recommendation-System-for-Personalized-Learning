# Explainable Multi-Agent Generative Recommendation System for Personalized Learning

## Overview

This system **analyzes learner behavior**, **plans optimal learning paths**, **generates personalized content**, and provides **explainable recommendations** using a sophisticated multi-agent architecture.

### Key Features

- **AI-Powered Profiling**: Embeddings + K-Means clustering + LLM analysis
- **Intelligent Path Planning**: A* graph search + Q-Learning optimization
- **Content Generation**: RAG-based personalized lessons, exercises, and quizzes
- **Hybrid Recommendations**: Path-based + Content-based + Collaborative filtering
- **Full Explainability**: SHAP, LIME, Counterfactuals, and LLM reasoning
- **Real Data Integration**: OULAD (Open University Learning Analytics Dataset)
- **Production Ready**: 73 automated tests, modular design, deployable dashboard

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│              (LangGraph + Custom Pipeline)              │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │    BLACKBOARD           │
        │  (Shared Memory)        │
        └────────────┬────────────┘
                     │
    ┌────────────────┼────────────────┬─────────────┬──────────┐
    │                │                │             │          │
┌───▼───┐      ┌─────▼─────┐   ┌─────▼──────┐ ┌───▼────┐ ┌──▼───┐
│Profile│      │   Path    │   │  Content   │ │  Rec   │ │ XAI  │
│ Agent │──────│  Planning │───│ Generator  │─│ Agent  │─│Agent │
└───────┘      └───────────┘   └────────────┘ └────────┘ └──────┘
```

### Pipeline Flow

1. **Profiling** → Analyze user behavior & learning style
2. **Path Planning** → Build optimal learning sequence
3. **Content Generation** → Create personalized resources
4. **Recommendation** → Rank and filter best options
5. **XAI** → Explain all decisions transparently

---

## Agents

### 1. Profiling Agent

**Role**: Build comprehensive learner profiles from interaction history

**Technologies**:
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Clustering**: K-Means for learning style inference
- **LLM Profiling**: Ollama (llama3.2:3b)

**Outputs**: Learning style, proficiency level, interests, strengths, weaknesses, AI-generated summary

---

### 2. Path Planning Agent

**Role**: Plan optimal learning paths using graph search and reinforcement learning

**Technologies**:
- **Graph Search**: Custom A* algorithm over resource dependency graph
- **Reinforcement Learning**: Q-Learning with epsilon-greedy exploration
- **Heuristics**: Level progression, style preferences, prerequisite handling

**Outputs**: Ordered sequence of learning resources with durations and prerequisites

---

### 3. Content Generator

**Role**: Generate personalized educational content adapted to learner profile

**Technologies**:
- **RAG**: Retrieval-Augmented Generation from knowledge base
- **LLM**: Ollama for text generation
- **Multi-format**: Courses, exercises, quizzes, videos, articles

**Outputs**: Contextual lessons, interactive exercises, adaptive quizzes

---

### 4. Recommendation Agent

**Role**: Recommend next-best learning resources with intelligent ranking

**Technologies**:
- **Hybrid Filtering**: Path-based + Content-based + Collaborative
- **LLM Ranking**: AI-powered priority scoring
- **Top-K Selection**: Guarantees exact number of recommendations

**Outputs**: Ranked resource list with priority scores and justifications

---

### 5. XAI Agent

**Role**: Provide transparent explanations for all system decisions

**Technologies**:
- **SHAP**: Feature importance for profile classification
- **LIME**: Local interpretable model-agnostic explanations
- **Counterfactuals**: "What-if" scenario generation
- **LLM Reasoning**: Natural language explanation chains

**Outputs**: Profile rationale, path logic, recommendation justifications, trust scores

---

## Dataset & Evaluation

### OULAD Dataset

**Primary Data Source**: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)

- **Files**: studentInfo.csv, studentAssessment.csv, courses.csv, vle.csv, studentVle.csv
- **Students**: 32,593 learners
- **Courses**: 7 modules
- **Interactions**: 10+ million VLE events

**Loader**: [data/oulad_loader.py](data/oulad_loader.py)

### Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Recommendation Quality** | NDCG@K, MRR, Recall@K, Precision@K, MAP, HitRate |
| **Content Quality** | ROUGE-N, ROUGE-L, BERTScore-like similarity, Readability |
| **Explainability** | Faithfulness, Plausibility, Completeness, Trust Score, Consistency |

**Evaluation Suite**: [evaluation/system_evaluation.py](evaluation/system_evaluation.py)

---

## Installation

### Prerequisites

- Python 3.9 - 3.13 (**NOT 3.14+** due to Pydantic compatibility)
- Conda or virtualenv
- [Ollama](https://ollama.ai) with llama3.2:3b model

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/rouakhadhraoui/Explainable-Multi-Agent-Generative-Recommendation-System-for-Personalized-Learning.git
cd Explainable-Multi-Agent-Generative-Recommendation-System-for-Personalized-Learning

# 2. Create virtual environment
conda create -n multiagent_learning python=3.11 -y
conda activate multiagent_learning

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama (if not already installed)
# Visit: https://ollama.ai/download
ollama pull llama3.2:3b

# 5. Run the demo
python main.py
```

---

## Usage

### Run Demo with Synthetic Users

```bash
python main.py
```

Demonstrates the full pipeline on 3 synthetic users (Alice, Bob, Charlie) with different learning profiles.

**Output**: Profile analysis, learning paths, generated content, recommendations, XAI explanations

---

### Run Tests

```bash
# All tests (73 tests, ~12 minutes)
pytest -q

# Specific agent
pytest tests/test_profiling_agent.py -v
pytest tests/test_path_planning_agent.py -v

# OULAD integration
pytest tests/test_oulad_integration.py -v
```

**Expected**: 73 tests passed

---

### Run System Evaluation

```bash
python evaluation/system_evaluation.py
```

Generates comprehensive metrics report with NDCG, MRR, ROUGE, BERTScore, Faithfulness, etc.

---

## Interactive Dashboard

### Live Demo

**Streamlit Dashboard**: [https://5q3gprp9db2bdx2c6jmgcl.streamlit.app/]

### Run Locally

```bash
streamlit run visualization/app.py
```



### Features

- **Learner Profiles**: Interactive radar charts, AI summaries
- **Learning Paths**: Network graph visualization
- **Recommendations**: Priority bars, detailed cards
- **XAI Insights**: SHAP waterfall charts, LIME explanations, counterfactuals

---

## Testing

### Test Coverage

**73 automated tests** covering:

- Unit tests for each agent
- Integration tests with OULAD
- Metrics validation
- End-to-end pipeline tests
- Edge cases and error handling

### Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| Profiling Agent | 8 | Passing |
| Path Planning Agent | 7 | Passing |
| Content Generator | 9 | Passing |
| Recommendation Agent | 10 | Passing |
| XAI Agent | 8 | Passing |
| Orchestrator | 6 | Passing |
| OULAD Integration | 5 | Passing |
| Metrics | 20 | Passing |

---

## Technologies

### Core Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11 |
| **LLM** | Ollama (llama3.2:3b) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **ML** | scikit-learn (K-Means, preprocessing) |
| **Graph Search** | Custom A* implementation |
| **RL** | Q-Learning with epsilon-greedy |
| **XAI** | SHAP, LIME |
| **Orchestration** | LangGraph (with custom fallback) |
| **Data** | Pandas (OULAD processing) |
| **Visualization** | Streamlit, Plotly |
| **Testing** | pytest |

### Agent Technologies

```python
Profiling:     Embeddings + K-Means + LLM
Path Planning: A* + Q-Learning + Heuristics
Content Gen:   RAG + LLM
Recommendation: Hybrid Filtering + LLM Ranking
XAI:           SHAP + LIME + Counterfactuals + LLM Reasoning
```

---

## Acknowledgments

- **OULAD Dataset**: Open University Learning Analytics Dataset
- **Ollama**: Local LLM infrastructure
- **Streamlit**: Dashboard framework
- **Hugging Face**: sentence-transformers models
