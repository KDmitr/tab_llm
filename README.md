<div align="center">

# 🧬 Synthetic Tabular Data via LLMs

**An LLM-based tabular data generator as an alternative to specialized models,**
**with strategic prompt example selection and a comprehensive evaluation pipeline**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat)](#-license)

<br>

*Can a general-purpose LLM generate synthetic tabular data*
*on par with — or better than — specialized generative models?*
*And does it matter which real rows we put in the prompt?*

<br>

[Key Ideas](#-key-ideas) · [Strategies](#-selection-strategies-s1s6) · [Architecture](#%EF%B8%8F-architecture) · [Quick Start](#-quick-start) · [Evaluation](#-evaluation-suite) · [Datasets](#-datasets)

</div>

<br>

---

## 📖 Table of Contents

- [Key Ideas](#-key-ideas)
- [Selection Strategies (S1–S6)](#-selection-strategies-s1s6)
- [Architecture](#%EF%B8%8F-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Evaluation Suite](#-evaluation-suite)
- [Tail Distribution Analysis](#-tail-distribution-analysis)
- [Evolutionary Data Curation](#-evolutionary-data-curation)
- [Datasets](#-datasets)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [License](#-license)

---

## 💡 Key Ideas

This project implements an **LLM-based synthetic tabular data generator** (`generator_syn.py`) and pits it against specialized generative models (CTGAN, DPGAN, DDPM, TabDDPM, TVAE, and others). The repository includes both generated synthetic datasets and precomputed evaluation metrics, enabling direct comparison.

The generator supports two modes — **few-shot** (real rows as in-context examples) and **zero-shot** (column schema only) — and works with any OpenAI-compatible LLM endpoint.

> **Core question:** In the few-shot setting, *which* real rows should we include in the prompt? This project proposes **six principled selection strategies** and shows that the choice of examples significantly impacts generation quality and privacy.

The repository provides:

- **An LLM-based generator** with domain-aware prompting and automatic JSON parsing with retry logic
- **Six example selection strategies** for choosing which real rows go into the prompt
- **A full evaluation pipeline** built on top of [synthcity](https://github.com/vanderschaarlab/synthcity) with additional custom metrics
- **A three-layer privacy audit** via Membership Inference Attacks
- **Evolutionary curation** via genetic algorithms for post-hoc dataset optimization
- **Precomputed results** — generated data and metrics across multiple models, strategies, and datasets

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 🎯 Selection Strategies (S1–S6)

Six strategies for selecting which real rows to include as few-shot examples in the LLM prompt:

| # | Strategy | Selection Criterion | Intuition |
|:-:|:---------|:---------------------|:----------|
| S1 | **Random** | Uniform random sample | Reproducible baseline |
| S2 | **Center** | Most typical examples | Dense, representative rows |
| S3 | **Tail** | Most statistically isolated examples | Rare, outlier rows |
| S4 | **MaxCoverage** | Greedy farthest-point sampling | Maximum feature-space spread |
| S5 | **PrivacyAware** | Pareto ranking (utility vs. privacy) | Balanced tradeoff via α parameter |
| S6 | **Stratified** | Clustering → one representative per cluster | Covers all data subgroups |

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REAL DATASET                             │
│                    (train / test split)                          │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│    PER-EXAMPLE SCORING   │  ← scoring.py
│    (distance-based)      │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│  STRATEGY SELECTOR       │
│  S1–S6 (selection.py)    │
└──────────────┬───────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM GENERATOR (generator_syn.py)             │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │  Few-Shot   │    │  Zero-Shot   │    │  JSON Parsing &   │   │
│  │  (real rows │    │  (schema     │    │  Auto-Retry       │   │
│  │  in prompt) │    │   only)      │    │  (3 attempts)     │   │
│  └─────────────┘    └──────────────┘    └───────────────────┘   │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION PIPELINE                        │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ synthcity       │  │ Privacy Audit│  │ Custom Metrics    │   │
│  │ (sanity, stats, │  │ 3-Layer MIA  │  │ (PRDC, MMD,       │   │
│  │  perf, privacy) │  │              │  │  coverage, NN-Id) │   │
│  └─────────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 📁 Project Structure

```
├── llm/                            # Core generation & evaluation pipeline
│   ├── generator_syn.py            # GEN class: few-shot & zero-shot synthesis
│   ├── selection.py                # Six prompt example selection strategies
│   ├── scoring.py                  # Distance-based per-example scoring
│   ├── add_metrics.py              # Custom quality metrics (PRDC, MMD, etc.)
│   ├── AS_MIA_with_layers.py       # Three-layer Membership Inference Attack
│   └── synthetic_combination_analysis.py  # Convex decomposition analysis
│
├── evol_curation/                  # Genetic algorithm data curation
│   ├── gen.py                      # GA engine (GAConfig, GAResult)
│   ├── fitness.py                  # ML fitness: XGBoost + LogReg AUC
│   ├── individ.py                  # Individual (row subset) representation
│   ├── crossover.py                # Exchange crossover operator
│   ├── mutation.py                 # Replacement mutation operator
│   └── selection.py                # Tournament selection
│
├── tail_extension/                 # Tail distribution analysis
│   ├── tail.py                     # Mahalanobis, energy dist, KDE JS-div
│   └── Matrix.py                   # MI / Wasserstein / JS dependency matrices
│
├── data/
│   ├── train/                      # Training splits (23 datasets)
│   └── test/                       # Held-out test splits
│
└── metric/                         # Precomputed experimental results
    ├── llm_strategy/               # Per-strategy results (S1–S6)
    ├── llm_hint/                   # With domain-knowledge hints
    ├── llm_zero_shot/              # Zero-shot baseline
    ├── gen_ratio/                  # Generation ratio ablation (1.0×–2.5×)
    ├── curation/                   # Evolutionary curation outputs
    ├── batch/                      # Batch-level metrics
    └── tail/                       # Tail quality evaluation
```

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/synthetic-tabular-llm.git
cd synthetic-tabular-llm

pip install pandas numpy scikit-learn scipy xgboost matplotlib tqdm rich synthcity
```

> **Note:** You need an OpenAI-compatible LLM endpoint (see [Tech Stack](#%EF%B8%8F-tech-stack) for tested models).

### Generate Synthetic Data

```python
from llm.scoring import compute_scores
from llm.selection import select_examples
from llm.generator_syn import GEN

# Score training examples and select 50 using MaxCoverage strategy
scores = compute_scores(df_train, num_cols=num_cols, cat_cols=cat_cols)
indices = select_examples(D=scores.D, scores=scores, n=50, strategy="S4_MaxCoverage")

# Generate
gen = GEN(
    gen_client=client,                # OpenAI-compatible client
    gen_model_nm="qwen2.5-coder:14b",
    real_data=df_train.iloc[indices],
    cols=feature_columns,
    batch_size=10,
    gen_ratio=1.0,
    dataset_hint="adult",             # optional domain constraints
)
df_syn = gen.run(name="X_syn_adult")
```

### Evaluate Quality

```python
from llm.add_metrics import evaluate_generation

metrics = evaluate_generation(
    df_syn=df_syn,
    df_real_test=df_test,
    num_cols=num_cols,
    cat_cols=cat_cols,
)
print(metrics.to_dict())
```

> The full evaluation pipeline also uses **synthcity** metrics — see [Evaluation Suite](#-evaluation-suite) for details.

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 📊 Evaluation Suite

Evaluation combines the **[synthcity](https://github.com/vanderschaarlab/synthcity)** library with custom metrics, covering four dimensions:

| Category | Metrics (synthcity) | Custom Additions |
|:---------|:-------------------|:-----------------|
| **Sanity** | Data mismatch, common rows proportion, nearest neighbor distance, close/distant values probability | — |
| **Statistical** | Jensen–Shannon divergence, KL divergence, KS test, Wasserstein distance, Alpha Precision & Coverage | PRDC (precision, recall, density, coverage), MMD, NN identifiability, distributional coverage |
| **Performance** | Train-on-synthetic / test-on-real with Linear Model, MLP, XGBoost; feature rank distance | — |
| **Privacy** | δ-presence, k-anonymization, k-map, l-diversity, identifiability score | Three-layer MIA audit |
| **Detection** | XGBoost / GMM / Linear detector (real vs. synthetic) | — |

### Privacy Audit (MIA)

A **three-layer Membership Inference Attack** models three adversary strength levels — from an omniscient oracle (upper bound) to a blind external attacker (realistic threat). The gap between layers quantifies how much privacy protection the generation pipeline provides.

### Structural Analysis

Convex decomposition of synthetic rows reveals whether the LLM interpolates between real examples, extrapolates beyond them, or memorizes individual rows.

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 📈 Tail Distribution Analysis

The `tail_extension/` module evaluates how well synthetic data captures the **tails of the real distribution** — a known weak point for generative models. Includes Mahalanobis distance profiling, energy distance, KDE-based Jensen–Shannon divergence, and pairwise dependency matrix comparisons.

---

## 🧪 Evolutionary Data Curation

A **genetic algorithm** optimizes synthetic dataset composition after generation. Individuals represent subsets of generated rows; fitness is evaluated via downstream ML performance (XGBoost + Logistic Regression AUC on a held-out test set). Supports tournament selection, exchange crossover, and replacement mutation.

```python
from evol_curation.gen import GeneticAlgorithm, GAConfig

config = GAConfig(n_generations=50, crossover_prob=0.7, mutation_prob=0.02)
ga = GeneticAlgorithm(config=config, ...)
results = ga.run(test_data=df_test)
```

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 📋 Datasets

Evaluated on **23 tabular benchmarks** spanning classification and regression, including:

| Dataset | Domain | Dataset | Domain |
|:--------|:-------|:--------|:-------|
| Adult | Census income | Seattle Housing | Real estate |
| Phoneme | Speech signals | 562 CPU Small | Hardware performance |
| Magic04 | Gamma-ray telescope | Credit-G | Credit scoring |
| Iris | Flower species | Page Blocks | Document layout |
| Wine | Chemical analysis | Breast Cancer | Medical diagnosis |
| Diabetes | Health indicators | Pendigits | Handwriting recognition |
| Drug Consumption | Pharmacology | Online Shoppers | E-commerce behavior |

…and others (Yeast, Nursery, HELOC, Travel, VK Data, BodyFat, Sea Level, US Location).

<p align="right">(<a href="#-table-of-contents">↑ back to top</a>)</p>

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.9+ |
| **LLM Models** | Qwen 2.5 Coder 14B (local, Ollama / vLLM), LLaMA-3 70B, GPT-4o, Qwen3 32B, Qwen3 235B-A22B (API) |
| **Baselines** | CTGAN, DPGAN, DDPM, TabDDPM, TVAE, and others |
| **Evaluation** | [synthcity](https://github.com/vanderschaarlab/synthcity) + custom metrics |
| **ML** | scikit-learn, XGBoost |
| **Numerics** | NumPy, SciPy, pandas |
| **Visualization** | Matplotlib |
| **Utilities** | tqdm, rich |

---

## 📄 License

This project was developed as part of an academic thesis (ВКР).
Please contact the author for licensing and citation details.

---

<div align="center">

**⭐ If you find this work useful, please consider giving it a star! ⭐**

</div>
