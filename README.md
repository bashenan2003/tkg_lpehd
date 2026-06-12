# TKG-LPEHD

**LLM-Driven Dynamic Fusion of Graph and Rule-Based Methods for Temporal Knowledge Graph Reasoning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

TKG-LPEHD is a modular framework for temporal knowledge graph reasoning (TKGR) that deeply integrates rule-based reasoning, graph-based neural reasoning, and large language models. It employs LLM-guided three-dimensional relational path sampling, adaptive rule evolution with temporal validity assessment, and dynamic weight fusion to achieve state-of-the-art reasoning accuracy with strong interpretability. To deploy and run this framework, you need to create a data_files folder in the root directory and add datasets that meet the format requirements in the folder. Besides, you need to download SBERT semantic embeddings yourself.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)
- [License](#license)

---

## Architecture Overview

The framework operates as a five-phase pipeline:

```
Phase 1 (4.1): LLM-Guided 3D Relational Path Sampling
                ├─ Semantic Relevance (Sentence-BERT cosine)
                ├─ Co-occurrence Frequency
                └─ Temporal Correlation

Phase 2 (4.2): LLM-Assisted Validity Assessment
                └─ Dynamic rule validity period inference

Phase 3 (4.3): Rule Prioritization
                └─ Confidence × Validity Coverage ranking

Phase 4 (4.4): Adaptive Rule Evolution
                ├─ Real-time priority update
                └─ Low-priority rule pruning

Phase 5 (4.5): Dynamic Weight Fusion Reasoning
                ├─ Rule-based candidate scoring
                ├─ Graph-based (GNN) candidate scoring
                └─ LLM bidirectional path verification
```

---

## Project Structure

```
tkg_lpehd/
├── main.py                      # Pipeline entry point
├── config.py                    # Global configuration & hyperparameters
├── llm_client.py                # LLM API client abstraction
├── ablation_study.py            # Ablation experiment runner
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── dataset.py               # TKGDataSet: CSV/TTL loading, vocabularies, splits
│   ├── datahandler.py           # Data preprocessing utilities
│   └── neo4j_store.py           # Neo4j graph database connector
│
├── preprocessing/
│   ├── temporal_normalizer.py   # Timestamp normalization
│   └── quadruple_formatter.py   # Quad-to-LLM-prompt formatting
│
├── path_sampling/
│   ├── llm_guided_sampler.py    # Phase 1: 3D relational path sampling
│   └── validity_assessment.py   # Phase 2: LLM temporal validity assessment
│
├── rules/
│   ├── rule_mining.py           # Temporal rule induction from paths
│   ├── rule_evolution.py        # Phase 4: Adaptive rule evolution & pruning
│   └── sparql_matcher.py        # SPARQL temporal event matching engine
│
├── reasoning/
│   ├── rule_reasoning.py        # Phase 5a: Rule-based candidate scoring
│   ├── graph_reasoning.py       # Phase 5b: GNN embedding-based scoring
│   ├── dynamic_fusion.py        # Phase 5c: Dynamic weight fusion
│   └── llm_verification.py      # Phase 5: LLM bidirectional path verification
│
├── llm_providers/               # Pluggable LLM backends
│   ├── base.py                  # Abstract provider interface
│   ├── deepseek.py              # DeepSeek API (default)
│   ├── gpt4.py                  # GPT-4
│   ├── gpt35_turbo.py           # GPT-3.5 Turbo
│   ├── llama2_7b.py             # Llama2 7B (local)
│   ├── stochastic.py            # Stochastic decoding variant
│   ├── deterministic.py         # Deterministic decoding variant
│   └── prompt_paraphrasing.py   # Prompt paraphrasing for stability tests
│
├── evaluation/
│   └── metrics.py               # MRR, Hit@1, Hit@3, Hit@10 evaluator
│
├── models/                      # Cached GNN model checkpoints
├── data_files/                  # Dataset directory (see Datasets section)
├── output/                      # Evaluation scripts and diagrams
│   ├── eval_case_rules.py       # Case study rule evaluation (v1)
│   ├── eval_case_rules_v2.py    # Case study rule evaluation (v2, standalone)
│   └── generate_architecture_diagram.py
│
└── utils/
    └── __init__.py
```

---

## Hardware & Software Requirements

| Category | Minimum | Recommended (Paper Experiments) |
|----------|---------|--------------------------------|
| **CPU** | Intel Core i7-10700K, 8 cores | Intel Xeon 8375C, 16 cores |
| **RAM** | 32 GB | 64 GB |
| **GPU** | NVIDIA RTX 3090 (24 GB VRAM) | NVIDIA A100 (80 GB VRAM) |
| **OS** | Linux / Windows 10+ | Ubuntu 20.04+ |
| **Python** | 3.10 | 3.10 |
| **CUDA** | 11.7+ | 11.7+ |
| **Storage** | 5 GB free | 20 GB free (for full datasets + model checkpoints) |

**CPU-only mode** is supported but increases latency by 2.5--3×. GPU acceleration is strongly recommended for large-scale TKG experiments.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/tkg-lpehd.git
cd tkg-lpehd
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# or
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0 | GNN backbone, tensor operations |
| `numpy` | 2.4.4 | Numerical computation |
| `scipy` | 1.17.1 | Sparse matrix operations |
| `sentence-transformers` | 5.5.1 | SBERT semantic embeddings |
| `transformers` | 5.9.0 | HuggingFace LLM inference |
| `rdflib` | 7.6.0 | RDF graph / SPARQL query engine |
| `neo4j` | 6.2.0 | Neo4j graph database driver (optional) |
| `networkx` | 3.6.1 | Graph algorithms |
| `scikit-learn` | 1.8.0 | Evaluation utilities |
| `requests` | 2.34.2 | LLM API calls |

### 4. Configure LLM API (Required)

Edit `config.py` and set your API credentials:

```python
LLM_PROVIDER = "deepseek"                     # or "gpt35_turbo" / "llama2_7b"
LLM_API_KEY = "sk-your-api-key-here"          # Your API key
LLM_API_BASE = "https://api.deepseek.com/v1"  # API endpoint
```

**Without a real LLM API key**, set simulation mode:

```bash
export TKG_SIMULATE=1    # Linux / macOS
set TKG_SIMULATE=1       # Windows
```

In simulation mode, LLM validation steps return deterministic scores based on cosine similarity and temporal distance (sufficient for testing the pipeline, but metrics will differ from the paper).

---

## Datasets

### Dataset Format

All datasets must be CSV files with four columns:

| subject | relation | object | timestamp |
|---------|----------|--------|-----------|
| Barack_Obama | Make_a_visit | South_Korea | 2014-12-09 |
| Patient_1258 | Submit_appointment | Hospital_A | 2025-03-11 |

Timestamps must be in `YYYY-MM-DD` format.

### Dataset Directory Structure

```
data_files/
├── icews14/
│   ├── icews14.csv         # Quadruple data
│   ├── entities.txt        # Entity list (one per line)
│   └── relations.txt       # Relation list (one per line)
├── icews18/
│   ├── icews18.csv
│   ├── entities.txt
│   └── relations.txt
├── mhaes/
│   ├── mhaes.csv
│   ├── entities.txt
│   └── relations.txt
├── icews0515/
│   └── icews0515.csv
└── yago/
    ├── yago.csv
    └── yago.ttl
```

### Dataset Descriptions

| Dataset | #Quads | #Entities | #Relations | Time Span | Domain | Availability |
|---------|--------|-----------|------------|-----------|--------|-------------|
| **ICEWS14** | 92,991 | 7,128 | 230 | 2014 | Global political events | [Public](https://dataverse.harvard.edu/dataverse/icews) |
| **ICEWS18** | 468,558 | 23,033 | 256 | 2018 | Global political events | [Public](https://dataverse.harvard.edu/dataverse/icews) |
| **ICEWS05-15** | 479,329 | 10,488 | 251 | 2005–2015 | Global political events | [Public](https://dataverse.harvard.edu/dataverse/icews) |
| **YAGO** | — | — | — | Multi-year | General knowledge | [Public](https://yago-knowledge.org/) |
| **MHAES** | 582,636 | 26,249 | 328 | Up to 2025 | Chinese medical events | Private (partial academic release) |

**Note on MHAES**: This is a confidential Chinese medical epidemic surveillance dataset. Only a subset is available for academic research. Contact the corresponding author for access inquiries.

---

## Quick Start

### Minimal Run (Simulated LLM, ICEWS14, No GPU Required)

```bash
# Run the full pipeline with simulated LLM (no API key needed)
TKG_SIMULATE=1 python main.py --dataset ICEWS14

# Skip GNN training if you lack GPU (uses random embeddings)
TKG_SIMULATE=1 python main.py --dataset ICEWS14 --skip-training
```

### Standard Run (Real LLM, ICEWS14)

```bash
python main.py --dataset ICEWS14
```

Expected output:
```
[Phase 0] Loading ICEWS14 dataset...
[数据集] ICEWS14: 总四元组=92991, 实体=7128, 关系=230, 时间戳=365
[Phase 1] LLM-guided 3D path sampling...
[Phase 2] LLM-assisted validity assessment...
[Phase 3] Rule prioritization...
[Phase 4] Adaptive rule evolution...
[Phase 5] Dynamic weight fusion reasoning...
============================================================
```

### Case Study Evaluation Only

```bash
cd output
python eval_case_rules_v2.py
```

This runs the standalone case-study rule evaluation against ICEWS14 and MHAES without touching the main codebase.

---

## Advanced Usage

### Command-Line Flags

| Flag | Description |
|------|-------------|
| `--dataset <name>` | Dataset: `ICEWS14`, `ICEWS18`, `ICEWS0515`, `YAGO`, `MHAES` (default: `ICEWS14`) |
| `--skip-training` | Skip GNN training; use cached model or random embeddings |
| `--neo4j` | Enable Neo4j persistent graph storage |
| `--case-study-only` | Run Phase 5 case study query without full pipeline |

### Environment Variables

| Variable | Effect |
|----------|--------|
| `TKG_SIMULATE=1` | Use simulated LLM (no API calls, deterministic) |
| `TKG_REAL_LLM=1` | Force real LLM API calls |
| `TKG_ABLATION=<mode>` | Override ablation mode: `w/o G`, `w/o T`, `w/o E`, or empty for full model |
| `TKG_PROMPT_STYLE=<style>` | Prompt template: `improved` (default) or `original` |
| `TKG_LLM_PROVIDER=<provider>` | LLM backend: `deepseek`, `gpt35_turbo`, `llama2_7b`, `deterministic`, `prompt_paraphrasing` |

### Running Ablation Studies

```bash
# Without LLM Guidance Module (Phase 1 path validation)
TKG_ABLATION="w/o G" python main.py --dataset ICEWS14

# Without Temporal Validity Module (Phase 2)
TKG_ABLATION="w/o T" python main.py --dataset ICEWS14

# Without Candidate Entity Prompt Module (Phase 5 LLM verification)
TKG_ABLATION="w/o E" python main.py --dataset ICEWS14

# Full model (default)
python main.py --dataset ICEWS14
```

### Running LLM Stability Tests

```bash
# Switch LLM provider at runtime
TKG_LLM_PROVIDER="llama2_7b" python main.py --dataset ICEWS14
TKG_LLM_PROVIDER="gpt35_turbo" python main.py --dataset ICEWS14
TKG_LLM_PROVIDER="deterministic" python main.py --dataset ICEWS14
TKG_LLM_PROVIDER="prompt_paraphrasing" python main.py --dataset ICEWS14
```

### Running from Parent Directory as Module

```bash
cd D:\MyProjects
python -m tkg_lpehd.main --dataset ICEWS14
```

### Using Neo4j Graph Database (Optional)

```bash
# Requires running Neo4j instance (default: bolt://localhost:7687)
python main.py --dataset ICEWS14 --neo4j
```

---

## Reproducing Paper Results

### Figure 4: Time Complexity Plot

```bash
# Run on different dataset scales (use subset sampling)
python main.py --dataset ICEWS14    # ~93K quads
python main.py --dataset ICEWS18    # ~469K quads
```

### Figure 5: Temporal Validity Across Time Spans

Results are generated automatically during the standard pipeline run (Section 5.5). Check the output from Phase 2 validity assessment and Section 5.5 logging.

### Case Study Rule Evaluation (Section 4.6.2)

```bash
cd output
python eval_case_rules_v2.py
```

This standalone script computes Hit@1, Hit@10, MRR, and Conditional Accuracy for all three case-study rules against the actual ICEWS14 and MHAES datasets.

---

## Configuration Reference

Key hyperparameters in `config.py` (Section 5.7 validated optimal values):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LAMBDA_DECAY` | 0.1 | Temporal decay coefficient (Eq. 6, 12, 14) |
| `W_SEMANTIC` | 0.5 | Semantic relevance weight in 3D scoring |
| `W_FREQUENCY` | 0.3 | Co-occurrence frequency weight |
| `W_TEMPORAL` | 0.2 | Temporal correlation weight |
| `CONFIDENCE_THRESHOLD` | 0.6 | Minimum rule confidence c_p |
| `PRIORITY_THRESHOLD` | 0.6 | Rule priority threshold P_th |
| `ALPHA_FIXED` | 0.5 | Base weight for rule- vs graph-based reasoning |
| `EMBEDDING_DIM` | 128 | Entity/relation embedding dimension |
| `GNN_LAYERS` | 2 | Graph convolution layers |
| `GNN_LR` | 0.001 | GNN learning rate (Adam) |
| `GNN_EPOCHS` | 20 | GNN training epochs |
| `LLM_TEMPERATURE` | 0.1 | LLM decoding temperature |
| `MAX_PATH_LENGTH` | 3 | Maximum relational path length |

---

## Citation

If you use TKG-LPEHD in your research, please cite:

```bibtex
@article{tkglpehd2025,
  title   = {Enhanced Temporal Knowledge Graph Reasoning through TKG-LPEHD:
             LLM-Driven Dynamic Fusion of Graph and Rule-Based Methods},
  author  = {<Authors>},
  journal = {Data \& Knowledge Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Dataset Note**: ICEWS14, ICEWS18, and ICEWS05-15 are publicly available from the Harvard Dataverse. MHAES is a private dataset --- contact the corresponding author for access. Dataset usage must comply with the original data provider terms.
