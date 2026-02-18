# Standard Benchmark — Open-Source Search API Evaluation Framework

A comprehensive, open-source framework for evaluating and comparing search APIs — Linkup, Exa, Tavily, and Perplexity — using configurable datasets and three independent scoring systems.

---

## Features

- **Multi-system comparison** — benchmark up to 4 search APIs side-by-side
- **3 independent scoring systems** — Ragas (LLM-based), HuggingFace Judges, and automated metrics (BERTScore)
- **Flexible dataset loading** — CSV, JSONL, or built-in samples
- **Full CLI pipeline** — load datasets, run searches, evaluate responses, analyze results
- **Interactive dashboard** — Streamlit-powered UI for exploring results and adjusting weights
- **Evaluation caching** — granular per-metric caching to minimize API costs
- **Configurable weights** — adjust scoring weights via dashboard or config

---

## Quick Start

### 1. Install

```bash
cd standard-benchmark
uv sync
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — at minimum you need OPENAI_API_KEY and one search API key
```

### 3. Load a dataset

```bash
# Built-in sample (30 queries, instant)
uv run standard-benchmark dataset sample --output data/queries.jsonl

# Or from CSV
uv run standard-benchmark dataset load \
  --source csv --path queries.csv --text-column question \
  --output data/queries.jsonl
```

### 4. Run searches

```bash
uv run standard-benchmark search run \
  -i data/queries.jsonl \
  -o data/responses.json \
  --systems linkup,exa,tavily,perplexity
```

### 5. Evaluate responses

```bash
uv run standard-benchmark evaluate run \
  -q data/queries.jsonl \
  -r data/responses.json \
  -o data/evaluations.json \
  --use-ragas
```

### 6. View results

```bash
# Summary statistics
uv run standard-benchmark analyze stats --evaluations data/evaluations.json

# Interactive dashboard
uv run standard-benchmark analyze dashboard
```

---

## Dataset Format

Queries are stored as JSONL, one JSON object per line:

```json
{"id": "q_0001", "text": "What were Apple's total revenues in Q4 2024?", "metadata": {}}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | auto-generated | Unique query identifier |
| `text` | string | yes | The search query |
| `metadata` | object | no | Arbitrary metadata |

---

## Loading Datasets

### Built-in sample (30 queries)

```bash
uv run standard-benchmark dataset sample --output data/queries.jsonl
```

### From CSV

```bash
uv run standard-benchmark dataset load --source csv \
  --path queries.csv --text-column question \
  --output data/queries.jsonl
```

### From JSONL

```bash
uv run standard-benchmark dataset load --source jsonl \
  --path queries.jsonl --output data/queries.jsonl
```

---

## Pipeline Overview

```
Dataset → Search → Evaluate → Analyze
```

| Stage | Command | Description |
|-------|---------|-------------|
| **Dataset** | `dataset load` / `dataset sample` | Load queries from CSV, JSONL, or samples |
| **Search** | `search run` | Execute queries across Linkup, Exa, Tavily, Perplexity |
| **Evaluate** | `evaluate run` | Score responses using Ragas, HuggingFace Judges, BERTScore |
| **Analyze** | `analyze stats` / `analyze dashboard` | View statistics, launch interactive dashboard |

---

## Evaluation System

### Three Independent Scoring Systems

All scores are normalized to a **1-10 scale**.

#### 1. Ragas Score (LLM-based)

| Metric | Weight | Description |
|--------|--------|-------------|
| Faithfulness | 0.45 | Factual accuracy validated against sources |
| Completeness | 0.45 | Coverage of query aspects |
| Source Quality | 0.10 | Quality of retrieved sources |

#### 2. HuggingFace Score (LLM Judges)

| Metric | Weight | Description |
|--------|--------|-------------|
| Correctness Classifier | 0.45 | Binary correctness (PollMultihopCorrectness) |
| Correctness Grader | 0.45 | Granular correctness (PrometheusAbsoluteCoarseCorrectness) |
| Response Quality | 0.10 | Overall quality (MTBenchChatBotResponseQuality) |

#### 3. Automated Score (Metrics-based)

| Metric | Weight | Description |
|--------|--------|-------------|
| BERTScore Relevance | 0.55 | Semantic similarity: query ↔ response |
| BERTScore Faithfulness | 0.45 | Semantic similarity: response ↔ sources |

### Score Normalization

| Metric | Original Scale | Normalized To |
|--------|---------------|---------------|
| Faithfulness | 0-1 | 1-10 |
| Completeness | 1-5 | 1-10 |
| Source Quality | 1-5 | 1-10 |
| HF Classifier | True/False | 0 or 10 (binary) |
| HF Grader | 1-5 | 1-10 |
| HF Quality | 1-10 | 1-10 (native) |
| BERTScore | 0-1 | 1-10 |

### Evaluation Caching

Each metric is cached individually (7-day TTL). Adding new judges later won't invalidate existing cached metrics — only missing ones trigger API calls.

---

## Configuration

### API Keys (`.env`)

```bash
cp .env.example .env
```

Required keys depend on which search systems you want to benchmark:

```bash
LINKUP_API_KEY=...       # For Linkup searches
EXA_API_KEY=...          # For Exa searches
TAVILY_API_KEY=...       # For Tavily searches
PERPLEXITY_API_KEY=...   # For Perplexity searches
OPENAI_API_KEY=...       # For LLM-based evaluation (Ragas/HuggingFace judges)
```

### Evaluation Weights (`config/evaluation_config.yaml`)

Customize scoring weights for each evaluation system.

---

## Project Structure

```
standard-benchmark/
├── src/standard_benchmark/
│   ├── cli/                  # CLI commands (dataset, search, evaluate, analyze)
│   ├── config/               # Settings, evaluation config
│   ├── data/
│   │   ├── loaders/          # Dataset loaders (CSV, JSONL, sample)
│   │   └── models.py         # Pydantic data models
│   ├── search/               # Search API clients (Linkup, Exa, Tavily, Perplexity)
│   ├── evaluation/           # Evaluation pipeline (Ragas, HuggingFace, automated)
│   ├── visualization/        # Streamlit dashboard & charts
│   ├── storage/              # Data persistence & caching
│   └── utils/                # Embeddings, helpers, logging
├── config/                   # Configuration files
├── data/                     # Data directory (queries ship here; outputs written at runtime)
├── pyproject.toml
├── .env.example
├── README.md
└── LICENSE
```

---

## Troubleshooting

**Embedding model download on first run:**
```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**Clear caches:**
```bash
uv run standard-benchmark search clear-cache
uv run standard-benchmark evaluate clear-cache
```

---

## License

MIT — see [LICENSE](./LICENSE).
