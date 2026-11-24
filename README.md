# Neuro-Symbolic Chest X-Ray Triage System

An industry-grade, safety-critical medical triage system combining deep representation learning (compact hybrid MobileViT-XXS neural backbones) with formal symbolic ontology auditing (Horn clause clinical rule engine) and risk-controlled selective classification.

> **CLINICAL RESEARCH DISCLAIMER**  
> **Research prototype only. Not a medical device and not for clinical decision-making.**  
> This system is designed solely for empirical benchmarking of risk-controlled selective classification and neuro-symbolic clinical auditing.

---

## Deliverables Mapping (D1 – D7)

| Deliverable | Description | Frozen Implementation File | Key Metric / Specification |
|-------------|-------------|----------------------------|----------------------------|
| **D1** | Leak-Free Split Manifest | [`data/processed/nih_splits.csv`](data/processed/nih_splits.csv) | 112,120 rows, 30,805 unique patients, 0% patient overlap across train/calib/test |
| **D2** | PyTorch NIH-14 Dataset | [`src/cxr_triage/data/dataset.py`](src/cxr_triage/data/dataset.py) | `(3, 224, 224)` RGB FloatTensor, 14 multi-label targets, deterministic transforms |
| **D3** | 3-Head Hybrid Backbone | [`src/cxr_triage/models/triage_net.py`](src/cxr_triage/models/triage_net.py) | MobileViT-XXS, 969k inference parameters (< 2.5M budget), sub-35ms CPU latency |
| **D4** | Study Calibrator & Risk Bounds | [`src/cxr_triage/calibration/calibrator.py`](src/cxr_triage/calibration/calibrator.py) | Exact Clopper-Pearson $(1-\delta)$ finite-sample UCB and 2-parameter logistic study calibration |
| **D5** | Auditable Rule Registry | [`configs/rules-v1.0.json`](configs/rules-v1.0.json) + [`src/cxr_triage/rules/engine.py`](src/cxr_triage/rules/engine.py) | 6 guideline-backed Horn clauses, ordinal severity semantics, sub-millisecond evaluation |
| **D6** | Comparative Policy Benchmark | [`src/cxr_triage/eval/benchmark.py`](src/cxr_triage/eval/benchmark.py) + [`scripts/evaluate.py`](scripts/evaluate.py) | De-confounded comparison of $\pi_{\text{prob}}$, $\pi_{\text{conf}}$, and $\pi_{\text{ns}}$ with paired patient bootstrap |
| **D7** | Unified Web Backend & UI | [`src/cxr_triage/api/app.py`](src/cxr_triage/api/app.py) + [`frontend/`](frontend/) | Zero-dependency FastAPI + Vanilla HTML5/CSS/JS clinical radiologist audit dashboard |

---

## Architectural Overview

```
CXR/
├── configs/
│   ├── default.yaml              # Pre-registered study parameters, paths, model configs
│   └── rules-v1.0.json           # 6 guideline-backed clinical Horn clause rules (SHA-256 tracked)
├── data/
│   ├── processed/
│   │   └── nih_splits.csv        # Frozen patient-level disjoint split manifest (112,120 studies)
│   └── raw/nih/images/           # Local radiograph PNG images
├── frontend/
│   ├── index.html                # High-contrast clinical audit dashboard layout
│   ├── styles.css                # Accessible dark mode theme with Emerald/Crimson/Cyan accents
│   └── app.js                    # Zero-dependency controller for live inference and rule visualization
├── scripts/
│   ├── generate_splits.py        # D1 patient-level disjoint split generator CLI
│   ├── train.py                  # Unified CPU/GPU training and calibration CLI
│   └── evaluate.py               # D6 single-command benchmark engine and report generator
├── src/cxr_triage/
│   ├── api/app.py                # Production FastAPI microservice serving backend + frontend
│   ├── calibration/calibrator.py # Post-hoc temperature scaling & exact Clopper-Pearson UCB
│   ├── config.py                 # Immutable dataclass configuration loader
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset and torchvision transform composition
│   │   └── splits.py             # Patient-hash disjoint split generator
│   ├── eval/benchmark.py         # Clustered bootstrap hypothesis testing and CPU latency profiling
│   ├── models/
│   │   ├── loss.py               # Composite multi-task loss (BCE + CrossEntropy + Distillation)
│   │   └── triage_net.py         # 3-Head MobileViT-XXS compact neural backbone
│   ├── policies/abstention.py    # De-confounded 4 triage policies (pi_prob, pi_var, pi_conf, pi_ns)
│   └── rules/engine.py           # Sub-millisecond Horn clause evaluator with ordinal severity
└── tests/                        # 49 unit and integration tests (100% passing)
```

---

## Quickstart Guide

### 1. Environment Setup
The codebase requires Python 3.11 with PyTorch and standard scientific packages:
```powershell
# Set Python path to src directory
$env:PYTHONPATH = "src"
```

### 2. Generate Frozen Splits (Deliverable D1)
```powershell
.venv\Scripts\python.exe scripts/generate_splits.py --raw-csv data/raw/nih/Data_Entry_2017.csv --output-csv data/processed/nih_splits.csv
```

### 3. Run CPU Smoke Training (Deliverable D3/D4)
Fast verification loop on local CPU images (completes in ~2 seconds):
```powershell
.venv\Scripts\python.exe scripts/train.py --epochs 1 --batch-size 4 --max-batches 2 --no-pretrained --output-dir artifacts/models/smoke
```
Generates `checkpoint.pt`, `calibrator.json`, and `run_summary.json` in `artifacts/models/smoke/`.

### 4. Run Policy Benchmark & Evaluation (Deliverable D6)
Evaluates $\pi_{\text{prob}}$, $\pi_{\text{conf}}$, and $\pi_{\text{ns}}$ on the test split:
```powershell
.venv\Scripts\python.exe scripts/evaluate.py --checkpoint artifacts/models/smoke/checkpoint.pt --calibrator artifacts/models/smoke/calibrator.json --batch-size 4 --max-batches 2 --output-dir artifacts/evaluation/smoke
```
Outputs `results_table.json` and its cryptographic hash `results_table.sha256`.

### 5. Launch Unified Web App & API (Deliverable D7)
Starts the unified server hosting both the REST API and the Radiologist Audit UI in a single lightweight process:
```powershell
.venv\Scripts\python.exe -m uvicorn cxr_triage.api.app:app --host 0.0.0.0 --port 8000
```
- **Web Dashboard**: Open `http://localhost:8000/` in any modern browser.
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **System Health Status**: `http://localhost:8000/health`

---

## Full Cloud / Kaggle GPU Training

For complete multi-epoch training on a full NVIDIA GPU (e.g. Kaggle T4 or A100):
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --backbone mobilevit_xxs \
  --epochs 15 \
  --batch-size 32 \
  --lr 5e-4 \
  --device cuda \
  --output-dir artifacts/models/full_run
```

---

## Pre-Registered Clinical Parameters

| Parameter | Symbol | Pre-Registered Value | Clinical Rationale |
|-----------|--------|----------------------|--------------------|
| Maximum False Omission Rate | $\epsilon$ | $0.02$ ($2.0\%$) | Strict ceiling on missed critical abnormalities in auto-reported normal cohort |
| Risk Bound Confidence | $1 - \delta$ | $0.95$ ($95\%$) | Exact finite-sample Clopper-Pearson binomial confidence level |
| Minimum Coverage | $c_{\min}$ | $0.30$ ($30.0\%$) | Minimum normal study triage volume to achieve clinical radiologist workload reduction |
| Candidate Coverage Grid | $\mathcal{C}$ | $[0.50, 0.60, 0.70, 0.80]$ | Grid points for empirical risk-coverage profiling |

---

## Test & Verification Suite

Run all 49 unit and integration tests across the complete neuro-symbolic pipeline:
```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest tests/ -v
```
All modules (configuration, dataset, backbones, temperature scaling, Clopper-Pearson bounds, Horn clause rules, de-confounded policies, bootstrap statistics, and FastAPI endpoints) achieve 100% test pass rates with 0 regressions.
