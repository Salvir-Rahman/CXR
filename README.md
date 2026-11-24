# Neuro-Symbolic Chest X-Ray Triage System

An industry-grade, safety-critical medical triage system combining deep representation learning (Chest X-Ray neural backbones) with formal symbolic ontology auditing and risk-controlled selective classification.

## Architectural Overview

- `src/cxr_triage/data`: Dataset loaders, NIH-14 parsing, patient-level train/val/test splitting, and DICOM/PNG ingestion.
- `src/cxr_triage/models`: Neural perception backbones (e.g. DenseNet121, TorchXRayVision) extracting disease probabilities and spatial feature representations.
- `src/cxr_triage/calibration`: Temperature scaling, conformal prediction, and calibrated probability estimation.
- `src/cxr_triage/rules`: Symbolic clinical ontology, medical guideline constraints, and critical finding escalation logic.
- `src/cxr_triage/policies`: Deferred triage policies optimizing selective classification under rigorous $(\epsilon, \delta)$ risk-coverage bounds.
- `src/cxr_triage/eval`: Safety auditing, clinical triage metrics, AUROC, ECE, and coverage-risk trade-off analysis.
- `src/cxr_triage/api`: Production FastAPI microservice for real-time triage scoring and audit log generation.
- `configs/default.yaml`: Centralized configuration governing all pre-registered thresholds and ontology parameters.

## Pre-Registered Study Parameters
- Allowable Error ($\epsilon$): 0.02
- Confidence Bound ($\delta$): 0.05
- Minimum Coverage ($c_{min}$): 0.30
- Coverage Grid: [0.5, 0.6, 0.7, 0.8]
