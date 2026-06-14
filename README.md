# Sepsis Early Warning System

> **Early prediction of sepsis onset in ICU patients using gradient boosting and deep learning on clinical time-series data.**

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sepsis-early-warning-kczsorrx6h9ultnbx9aeun.streamlit.app/)
[![CI](https://github.com/HarshitaAdlakha/sepsis-early-warning/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshitaAdlakha/sepsis-early-warning/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 🚀 [Live Demo → sepsis-early-warning.streamlit.app](https://sepsis-early-warning-kczsorrx6h9ultnbx9aeun.streamlit.app/)

> Try the **interactive Patient Simulator** — adjust ICU vitals and lab values to see real-time sepsis risk predictions.

---

## Overview

Sepsis is a life-threatening dysregulation of the host response to infection, responsible for **11 million deaths per year** worldwide. Every hour of delayed antibiotic therapy increases mortality by 7–8%. Early automated detection from routine ICU data can directly save lives.

This project implements a complete ML pipeline for **early sepsis prediction** from ICU time-series data, providing:

- **7-hour prediction horizon** before clinical sepsis onset
- **Two model families**: XGBoost (gradient boosting) + LSTM / GRU (recurrent deep learning)
- **44 clinical variables**: 15 vital parameters + 29 laboratory values
- **Sepsis-3 labelling**: co-occurrence of suspected infection + SOFA score increase ≥ 2
- End-to-end pipeline from raw data → preprocessing → training → evaluation

---

## Clinical Variables

### Vital Parameters (15)
| Variable | Description |
|---|---|
| heart_rate | Beats per minute |
| systolic_bp / diastolic_bp / mean_arterial_bp | Blood pressure (mmHg) |
| resp_rate | Respiratory rate (breaths/min) |
| spo2 | Peripheral oxygen saturation (%) |
| temperature | Core body temperature (°C) |
| gcs_total | Glasgow Coma Scale (3–15) |
| urine_output | Hourly urine output (mL/hr) |
| fio2 / peep / tidal_volume / rr_set / plateau_pressure / driving_pressure | Ventilator parameters |

### Laboratory Values (29)
CBC, metabolic panel, liver function, coagulation, blood gas, inflammatory markers (WBC, lactate, creatinine, bilirubin, procalcitonin, CRP, and more).

---

## Models

### XGBoost (Gradient Boosting)
- Each patient's time series is encoded as **statistical features per variable**: count, mean, std, min, max, quartiles, last value, and linear slope — yielding **441 features**
- Handles class imbalance via `scale_pos_weight`
- Hyperparameter tuning with **Optuna** (50 trials)

### Bidirectional LSTM / GRU
- Raw padded sequences (72 time steps × 44 features)
- **Masking layer** ignores padded positions during training
- Forward-fill + median imputation for missing lab values
- Early stopping on validation AUROC

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/HarshitaAdlakha/sepsis-early-warning.git
cd sepsis-early-warning
pip install -r requirements.txt
```

### 2. Run demo (synthetic data, ~3 min on CPU)

```bash
python demo/run_demo.py
```

This generates synthetic patient data, trains XGBoost, evaluates on a held-out test set, and saves ROC / PR curve plots to `results/demo/`.

---

## Full Pipeline (MIMIC-III)

> MIMIC-III requires credentialing via PhysioNet. See [access instructions](https://mimic.mit.edu/docs/gettingstarted/).

### Step 1 — Generate / load data

```bash
# Synthetic (no credentials needed)
python src/data/generate_demo_data.py --n-sepsis 570 --n-control 5618

# MIMIC-III: follow extraction instructions in docs/mimic_extraction.md
```

### Step 2 — Train XGBoost

```bash
# Hyperparameter tuning (optional, ~20 min)
python src/training/train_xgboost.py --mode tune --n-trials 50

# Train final model
python src/training/train_xgboost.py --mode train
```

### Step 3 — Train RNN

```bash
python src/training/train_rnn.py --rnn-type lstm
python src/training/train_rnn.py --rnn-type gru
```

### Step 4 — Results

All metrics, plots and model files are saved under `results/`.

---

## Results (Synthetic Demo Data)

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| XGBoost | ~0.91 | ~0.70 | ~0.83 | ~0.88 | ~0.60 |
| Bi-LSTM | ~0.88 | ~0.67 | ~0.80 | ~0.85 | ~0.57 |
| Bi-GRU | ~0.87 | ~0.65 | ~0.78 | ~0.86 | ~0.55 |

*Results on synthetic data. Performance on real MIMIC-III data varies with exact preprocessing.*

---

## Project Structure

```
sepsis-early-warning/
├── .github/workflows/ci.yml     ← GitHub Actions CI
├── configs/
│   └── config.json              ← Hyperparameters and data settings
├── demo/
│   └── run_demo.py              ← End-to-end demo (no MIMIC needed)
├── src/
│   ├── data/
│   │   ├── generate_demo_data.py  ← Synthetic patient generator
│   │   ├── loader.py              ← Dataset loading utilities
│   │   └── preprocessing.py       ← XGBoost encoding + RNN padding/normalisation
│   ├── models/
│   │   ├── xgboost_model.py       ← XGBoost classifier + Optuna tuning
│   │   └── rnn_model.py           ← Keras Bi-LSTM / Bi-GRU
│   ├── training/
│   │   ├── train_xgboost.py       ← XGBoost training script
│   │   └── train_rnn.py           ← RNN training script
│   ├── evaluation/
│   │   └── metrics.py             ← AUROC, AUPRC, sensitivity, net benefit, ...
│   └── visualization/
│       └── plots.py               ← ROC/PR curves, feature importance, calibration
├── tests/                        ← pytest unit tests
├── results/                      ← Metrics, plots, saved models (git-ignored)
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Why two model families?**
XGBoost excels on tabular statistical features and is fast to train and interpret. RNNs capture temporal dynamics and correlations between variables over time. Comparing both gives a complete view of what structure the data contains.

**Why statistical encoding for XGBoost?**
Tree models cannot ingest raw variable-length time series directly. Encoding each variable as distributional statistics (mean, std, slope, etc.) is a principled way to capture the signal without losing temporal trends.

**Why Bidirectional RNN?**
At inference time we have the complete available history, so a bidirectional architecture can use context from both earlier and later in the ICU stay to make better predictions. This differs from real-time streaming scenarios where only a causal architecture is appropriate.

**Why Sepsis-3 labels?**
Sepsis-3 (2016) is the current clinical gold standard, defining sepsis as suspected infection + acute organ dysfunction (SOFA ≥ 2). Earlier definitions (SIRS-based) had poor specificity. Using Sepsis-3 labels aligns the ML task with current clinical practice.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Data Privacy

**No patient data is included in this repository.** The `data/` directory is fully git-ignored. MIMIC-III data requires signing a Data Use Agreement with PhysioNet/MIT-LCP, which mandates preservation of patient privacy.

---

## References

1. Singer, M. et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810.
2. Johnson, A. E. W. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.
3. Moor, M. et al. (2019). Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks. *PMLR*, 106.
4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a portfolio project for the Amazon ML Summer Program 2024.*
