# HateScan 🔍

> Advanced multilabel hate speech detection for YouTube — built for political advisors and journalists

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-blue)](https://mlflow.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)

---

## What is HateScan

HateScan is a web application that allows authenticated users to enter a YouTube video URL, extract its comments, and automatically classify them across **4 simultaneous labels**: `IsAbusive`, `IsHatespeech`, `IsRacist`, `IsProvocative`.

It is designed as a support tool for **political advisory teams and journalists** who receive massive comment volumes and cannot afford to miss a single hate message.

Most teams solve a binary question: *is this comment toxic — yes or no?*
HateScan answers four questions simultaneously, per comment. That is not just a technical difference — it is a product difference. A politician does not need to know a comment "is toxic". They need to know if it is racist, if it is a coordinated provocation, if it requires legal escalation.

**What HateScan does:**
- Scrapes comments from any YouTube video via the YouTube Data API v3
- Classifies each comment across 4 labels simultaneously (multilabel, not multiclass)
- Stores search history per authenticated user in Supabase
- Generates a filterable executive dashboard with confidence-based prioritization
- Supports longitudinal analysis — compare toxicity of a channel before and after media events

**What HateScan does NOT do:**
- Delete or report comments on YouTube (no YouTube write API access — actions are performed by the user directly on YouTube)
- Support live streaming videos

---

## The Business Decision: Why Recall, not F1

The most important design decision in HateScan is not a model choice — it is a product policy.

When building a harmful content detection system, there are two types of errors:

| Error | Consequence | Cost |
|---|---|---|
| **False negative** — hate message not detected | Goes unmoderated. Reputational, legal, personal safety risk. | **Unbounded** |
| **False positive** — harmless comment flagged | User reviews it in 5 seconds and dismisses it. | **Marginal** |

This asymmetry changes everything. The classification threshold is not a neutral hyperparameter — it is a business policy lever.

That is why HateScan does not use the default threshold of 0.50, and why the primary KPI is **Recall per label, not aggregate F1**.

### Threshold Calibration

The threshold was calibrated on the validation set across four values (0.30, 0.35, 0.40, 0.45):

| Threshold | F1-macro | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.41 | 0.27 | ~0.98 |
| 0.35 | 0.48 | 0.33 | ~0.92 |
| 0.40 | 0.52 | 0.39 | ~0.87 |
| **0.45** | **0.553** | **0.426** | **0.807** |

**Threshold 0.45 selected** — it maximizes F1-macro while preserving Recall ≥ 0.80. Above 0.45, the precision gain no longer justifies the coverage loss.

**In practice:** of every 10 real hate messages in a video, the system automatically detects 8. The remaining 2 are not lost — they remain visible in the dashboard, sorted by probability, for manual review.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  FRONTEND  Streamlit + Google OAuth 2.0                          │
│            Filterable table · Longitudinal history · Charts      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────────────────┐
│  API  FastAPI · Pydantic validation at boundary                  │
└──────┬───────────────────┬──────────────────────────────────────-┘
       │                   │
┌──────▼──────┐   ┌────────▼──────────────────────────────────────┐
│  SCRAPING   │   │  NLP PIPELINE                                  │
│  YouTube    │   │  spaCy lemmatization                           │
│  Data API   │   │  → RoBERTa tokenizer (max_length=128)          │
│  v3         │   │  → HateScanTransformer logits [n,4]            │
└─────────────┘   │  → sigmoid per label → threshold 0.45         │
                  └───────────────────────────────────────────────-┘
┌──────────────────────────────────────────────────────────────────┐
│  PERSISTENCE  Supabase PostgreSQL                                │
│  Atomic INSERT: searches → comments (FK) or full rollback        │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  INFRA  Docker · Docker Compose · GitHub Actions (lint + tests)  │
│         MLflow experiment tracking · uv package manager          │
└──────────────────────────────────────────────────────────────────┘
```

The transformer (~3 GB) is loaded once per session via `@st.cache_resource` — the UI does not block on each interaction.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | uv |
| API | FastAPI + Pydantic |
| Frontend | Streamlit |
| Auth | Google OAuth 2.0 |
| NLP Classic (baseline) | scikit-learn · XGBoost · TF-IDF |
| NLP Transformer (production) | Hugging Face Transformers · RoBERTa |
| Experiment tracking | MLflow 2.12 |
| Hyperparameter tuning | Optuna |
| Database | Supabase (PostgreSQL) |
| Scraping | YouTube Data API v3 |
| Infra | Docker · Docker Compose · GitHub Actions |

---

## Model: transformer_roberta_4labels

### Why `cardiffnlp/twitter-roberta-base-hate`

The base model is pre-trained specifically on Twitter hate speech. The linguistic register of YouTube comments — abbreviations, emojis, informal language — is very similar to Twitter. This gives domain advantage over a generic RoBERTa without additional training cost.

### Architecture

A linear classification head over the `[CLS]` token produces logits of shape `(n, 4)`, with independent sigmoid per label. **Multilabel, not multiclass** — a comment can be simultaneously abusive and racist.

### Handling Critical Class Imbalance

`IsRacist` has only 125 positives out of 1,000 samples. With standard Binary Cross Entropy, the model learns to always predict negative — 87.5% accuracy, zero Recall. Useless for our use case.

**Solution: FocalLoss with γ=2.0 and inverse class-frequency weights.**

The factor `(1-p)²` reduces the gradient of well-classified negatives and forces the model to learn from rare positives. SMOTE was explicitly discarded — interpolation in token ID space generates semantically incoherent sequences.

### Final Metrics (test set, threshold 0.45)

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.838** |
| F1-macro | 0.553 |
| Recall | 0.807 |
| Precision | 0.426 |
| F1-micro | 0.575 |
| Overfitting gap (train/test) | **0.044 ✅** |

**On ROC-AUC 0.838:** given a real hate comment and a harmless one at random, the model ranks them correctly 83.8% of the time — regardless of where the threshold is set. A random classifier would be at 0.50. With 1,000 samples and 4 imbalanced labels, 0.838 demonstrates the model has learned real patterns.

### Per-label metrics (test set)

| Label | F1 | Precision | Recall |
|---|---|---|---|
| IsAbusive | 0.675 | 0.550 | 0.880 |
| IsHatespeech | 0.557 | 0.450 | 0.740 |
| IsRacist | 0.519 | 0.390 | 0.780 |
| IsProvocative | 0.462 | 0.320 | 0.830 |

### Hyperparameters

2 epochs · lr 2e-5 · warmup 10% · batch 16. Epoch 3 memorizes with 700 examples — early stopping at epoch 2.

---

## Scientific Validation Protocol

### Split Strategy

1,000 samples → Train 70% · Val 15% · Test 15%. Stratified by label — positive proportions preserved across all three sets. Without stratification, the holdout could be accidentally easier or harder with such imbalanced classes.

**The threshold of 0.45 was calibrated on Validation. Test was never touched until final evaluation. Zero data contamination.**

### Comparison with Classic Models

| Model | Task | F1 test | Gap | Notes |
|---|---|---|---|---|
| **transformer_roberta_4labels** | **4 labels** | **0.553** | **0.044 ✅** | Production model |
| baseline_lr | Binary (IsToxic) | 0.749 | 0.131 ✅ | |
| random_forest | Binary (IsToxic) | 0.755 | 0.319 ⚠️ | |
| xgboost | Binary (IsToxic) | 0.748 | 0.262 ⚠️ | |

**Comparing F1 0.553 (4 labels) with F1 0.749 (1 binary label) is comparing fundamentally different problems.** The transformer solves a much more complex hypothesis space with imbalanced minority classes.

---

## Dataset

**youtoxic_english_1000.csv** — 1,000 YouTube comments with 12 boolean labels.

| Label | Positives | Status |
|---|---|---|
| IsToxic | 462 | ✅ Binary baseline target |
| IsAbusive | ~350 | ✅ Production label |
| IsProvocative | ~200 | ✅ Production label |
| IsHatespeech | 138 | ✅ Production label |
| IsRacist | 125 | ✅ Production label |
| IsThreat | ~80 | ⚠️ Not included (scope) |
| IsObscene | ~70 | ⚠️ Not included (scope) |
| IsNationalist | 8 | ❌ Insufficient data |
| IsSexist | 1 | ❌ Insufficient data |
| IsHomophobic | 0 | ❌ Insufficient data |
| IsRadicalism | 0 | ❌ Insufficient data |

---

## Repository Governance: Large Artifacts

The transformer weights occupy ~3 GB — incompatible with standard Git. A model is an inference binary artifact, not source code; versioning it alongside code couples release cycles with retraining cycles.

**Solution: strict path contract.**

```
Git repository                    External storage
─────────────────                 ────────────────────────────────
src/hatescan/models/              models/artifacts/
  neural.py (path contract)         transformer_roberta_4labels/
  predictor.py (parametrizable)       model.safetensors (~3 GB)
training/                              config.json
  train_transformer.py               tokenizer_config.json
  trainer.py
  run_training_transformer.py
```

`predictor.py` loads the model from a parametrizable path — never hardcoded. The repository is fully cloneable and runnable without the weights; the model is provisioned separately.

**Automatic validations in Trainer:**
- `MAX_GAP_TRANSFORMER = 0.15` → observed gap: 0.044 ✅
- `MIN_F1_THRESHOLD = 0.40` → F1 test: 0.553 ✅

If any validation fails → the run is marked as FAILED in MLflow.

---

## API Output Format

```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "model_used": "transformer_roberta_4labels",
  "total_comments": 150,
  "toxic_count": 23,
  "non_toxic_count": 127,
  "comments": [
    {
      "comment_id": "abc123",
      "text_original": "you are so stupid and pathetic",
      "is_toxic": true,
      "confidence": 0.94,
      "categories": {
        "is_hatespeech": true,
        "is_racist": false,
        "is_threat": null,
        "is_obscene": null
      }
    }
  ]
}
```

---

## Database Schema

```
searches
├── search_id     UUID · primary key
├── user_id       Google OAuth sub
├── user_session  user email
├── video_url
├── video_id
├── video_title
├── created_at
├── num_comments
└── model_used

comments
├── comment_id    primary key
├── search_id     FK → searches
├── text_original
├── text_processed
├── is_toxic      boolean
├── confidence    float (0–1)
├── is_hatespeech boolean | null
├── is_racist     boolean | null
├── is_threat     boolean | null  (reserved for future model)
└── is_obscene    boolean | null  (reserved for future model)
```

Confidence prioritization in the dashboard:
- `confidence > 0.85` → immediate escalation
- `0.45 – 0.85` → standard review queue

---

## Project Structure

```
P9_NLP_EQUIPO1/
├── src/
│   └── hatescan/
│       ├── preprocessing/       # Text cleaning, tokenization, spaCy lemmatization
│       ├── scraping/            # YouTube Data API v3 scraper
│       ├── training/
│       │   ├── baseline.py      # LR + XGBoost + RF pipeline
│       │   ├── enhanced.py      # Pipeline + extra features (length, caps, punctuation)
│       │   ├── hyperparameter_tuning.py  # Optuna tuning
│       │   ├── train_transformer.py      # RoBERTa fine-tuning with FocalLoss
│       │   ├── trainer.py               # MLflow wrapper with automatic validations
│       │   ├── register_models.py       # Register classic models in MLflow
│       │   └── register_all_transformers.py  # Register transformer models in MLflow
│       ├── models/
│       │   └── predictor.py     # HateScanPredictor — unified inference interface
│       └── database/            # Supabase adapter (atomic INSERT with FK integrity)
├── api/
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Pydantic schemas
├── app/
│   └── streamlit_app.py         # Streamlit frontend with @st.cache_resource
├── models/
│   └── artifacts/               # Model weights — NOT committed to git
│       └── transformer_roberta_4labels/  # Production model
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 03_transformer_vs_ensemble.ipynb
│   └── colab/                   # Google Colab training notebooks
├── tests/
│   └── unit/
│       ├── test_mlflow.py              # 17 unit tests
│       └── test_integration_mlflow.py  # 18 integration tests
├── data/raw/youtoxic_english_1000.csv
├── run_training_pipeline.py     # Entry point: classic models
├── run_training_transformer.py  # Entry point: transformer
├── test_predictor.py            # Test inference without Streamlit/FastAPI
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- YouTube Data API v3 key
- Supabase project
- Google OAuth 2.0 credentials
- GPU recommended for training (Google Colab T4 works)

### Installation

```bash
git clone https://github.com/your-org/P9_NLP_EQUIPO1.git
cd P9_NLP_EQUIPO1

uv sync
python -m spacy download en_core_web_sm
cp .env.example .env
# Edit .env with your keys
```

### Environment Variables

```bash
# YouTube
YOUTUBE_API_KEY=your_key_here
MAX_COMMENTS_PER_SEARCH=20

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Model
MODEL_PATH=models/artifacts/transformer_roberta_4labels
```

### Provision the Model

The transformer weights are not in the repository. Download them from the shared Drive folder and place them at:

```
models/artifacts/transformer_roberta_4labels/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── metrics.json
```

### Train from Scratch (optional)

```bash
# Classic models (CPU)
python run_training_pipeline.py

# Transformer (requires GPU — use Google Colab)
python run_training_transformer.py

# Register pre-trained transformers in MLflow
python -m src.hatescan.training.register_all_transformers
```

### View MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

### Run Tests

```bash
pytest tests/unit/ -v
# Expected: 35 tests passing
```

### Run with Docker

```bash
docker compose up
# API:      http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Test Inference

```bash
python test_predictor.py
python test_predictor.py --save-db        # persist to Supabase
python test_predictor.py --threshold 0.40  # custom threshold
```

---

## Roadmap for the future implementation

- **Fine-tuning on Jigsaw** (160k Wikipedia comments) → expected F1-macro 0.65–0.72 without architecture changes
- **Public API** with real-time alerts
- **Multilingual support** — XLM-RoBERTa variant for non-English comments

---

## Key Differentiators

- ✅ **4 simultaneous multilabel classification**
- ✅ **FocalLoss without SMOTE** — semantically coherent imbalance handling
- ✅ Overfitting gap **0.044** — most generalizable model
- ✅ **ROC-AUC 0.838** on blind holdout
- ✅ **Complete end-to-end productive pipeline** — URL to dashboard

---

## Team

| Role | Name | Responsibilities |
|---|---|---|
| Data Engineer | Isabel Rodriguez | Docker · YouTube scraping · FastAPI · Supabase · CI/CD |
| Data Analyst | Juan Manuel Iriondo | EDA · Metrics · Dashboard · Streamlit frontend |
| Data Scientist #1 | Joaquín Lazaro | NLP pipeline · Threshold calibration · Validation protocol |
| Data Scientist #2 | Iris Amorim | MLflow · Experiment tracking · Transformer fine-tuning · Business framing |

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- [cardiffnlp/twitter-roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)
- [youtoxic dataset — Kaggle](https://www.kaggle.com/datasets/miklgr500/youtoxic)

---

*P9 NLP · Team 1 · Module 3*
