"""
Synthetic ICU time-series data generator for sepsis prediction demo.

Generates realistic-looking vital signs and lab values for ICU patients,
with a subset developing sepsis (labelled by Sepsis-3 criteria logic).
Use this when you do not have access to MIMIC-III.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Clinical variable definitions ------------------------------------------------
VITALS = [
    "heart_rate",        # bpm
    "systolic_bp",       # mmHg
    "diastolic_bp",      # mmHg
    "mean_arterial_bp",  # mmHg
    "resp_rate",         # breaths/min
    "spo2",              # %
    "temperature",       # °C
    "gcs_total",         # Glasgow Coma Scale (3–15)
    "urine_output",      # mL/hr (hourly)
    "fio2",              # fraction
    "peep",              # cmH2O (ventilated)
    "tidal_volume",      # mL (ventilated)
    "rr_set",            # set respiratory rate
    "plateau_pressure",  # cmH2O
    "driving_pressure",  # cmH2O
]

LABS = [
    "wbc",           # 10^9/L
    "hemoglobin",    # g/dL
    "hematocrit",    # %
    "platelets",     # 10^9/L
    "sodium",        # mEq/L
    "potassium",     # mEq/L
    "chloride",      # mEq/L
    "bicarbonate",   # mEq/L
    "bun",           # mg/dL
    "creatinine",    # mg/dL
    "glucose",       # mg/dL
    "lactate",       # mmol/L
    "bilirubin",     # mg/dL
    "alt",           # U/L
    "ast",           # U/L
    "albumin",       # g/dL
    "inr",           # ratio
    "ptt",           # seconds
    "ph",            # arterial
    "pao2",          # mmHg
    "paco2",         # mmHg
    "base_excess",   # mEq/L
    "troponin",      # ng/mL
    "procalcitonin", # ng/mL
    "crp",           # mg/L
    "fibrinogen",    # mg/dL
    "d_dimer",       # mcg/mL FEU
    "bnp",           # pg/mL
    "ferritin",      # ng/mL
]

ALL_FEATURES = VITALS + LABS

# Normal distributions for each variable (mean, std) --------------------------
NORMAL_PARAMS = {
    "heart_rate":       (75,  12),
    "systolic_bp":      (120, 15),
    "diastolic_bp":     (75,  10),
    "mean_arterial_bp": (90,  10),
    "resp_rate":        (16,   3),
    "spo2":             (97,   1.5),
    "temperature":      (37.0, 0.4),
    "gcs_total":        (14,   1),
    "urine_output":     (60,  20),
    "fio2":             (0.35, 0.1),
    "peep":             (5,    1),
    "tidal_volume":     (500,  50),
    "rr_set":           (14,   2),
    "plateau_pressure": (18,   3),
    "driving_pressure": (12,   2),
    "wbc":              (8,    2.5),
    "hemoglobin":       (12,   1.5),
    "hematocrit":       (37,   5),
    "platelets":        (220,  60),
    "sodium":           (140,  4),
    "potassium":        (4.0,  0.4),
    "chloride":         (102,  4),
    "bicarbonate":      (24,   3),
    "bun":              (18,   8),
    "creatinine":       (0.9,  0.2),
    "glucose":          (110,  20),
    "lactate":          (1.2,  0.4),
    "bilirubin":        (0.8,  0.4),
    "alt":              (30,   15),
    "ast":              (28,   12),
    "albumin":          (3.8,  0.4),
    "inr":              (1.1,  0.15),
    "ptt":              (30,   5),
    "ph":               (7.40, 0.03),
    "pao2":             (90,   12),
    "paco2":            (40,   4),
    "base_excess":      (0,    2),
    "troponin":         (0.02, 0.01),
    "procalcitonin":    (0.05, 0.03),
    "crp":              (5,    4),
    "fibrinogen":       (300,  60),
    "d_dimer":          (0.4,  0.2),
    "bnp":              (50,   30),
    "ferritin":         (150,  80),
}

# Sepsis derangement multipliers (septic patients drift in these directions)
SEPSIS_DELTA = {
    "heart_rate":       +25,
    "resp_rate":        +8,
    "temperature":      +1.2,
    "systolic_bp":      -25,
    "mean_arterial_bp": -18,
    "spo2":             -4,
    "lactate":          +2.5,
    "wbc":              +8,
    "creatinine":       +0.8,
    "bilirubin":        +1.5,
    "platelets":        -80,
    "procalcitonin":    +8,
    "crp":              +60,
    "ph":               -0.08,
    "bicarbonate":      -5,
    "base_excess":      -5,
}


def _generate_patient_series(
    n_hours: int,
    is_septic: bool,
    onset_hour: int,
    rng: np.random.Generator,
    lab_frequency: int = 4,
) -> pd.DataFrame:
    """Generate one patient's time series."""
    records = []
    for h in range(n_hours):
        row = {"hour": h}
        for feat in ALL_FEATURES:
            mu, sigma = NORMAL_PARAMS[feat]

            if is_septic and h >= onset_hour:
                hours_after = h - onset_hour
                progress = min(hours_after / 12.0, 1.0)
                delta = SEPSIS_DELTA.get(feat, 0) * progress
                mu = mu + delta

            is_lab = feat in LABS
            if is_lab and (h % lab_frequency != 0):
                row[feat] = np.nan
            else:
                value = rng.normal(mu, sigma)
                row[feat] = round(float(value), 3)

        records.append(row)

    return pd.DataFrame(records)


def generate_dataset(
    n_sepsis: int = 570,
    n_control: int = 5618,
    min_hours: int = 7,
    max_hours: int = 72,
    prediction_horizon: int = 7,
    seed: int = 42,
    output_dir: str = "data/processed",
) -> dict:
    """
    Generate synthetic patient dataset.

    Returns dict with keys 'train', 'val', 'test', each a tuple of
    (list_of_dataframes, labels_array).
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_series = []
    all_labels = []
    all_meta = []

    # Generate sepsis patients
    for i in range(n_sepsis):
        n_hours = int(rng.integers(max(min_hours + prediction_horizon + 1, 20), max_hours))
        # onset between hour 7 and (n_hours - prediction_horizon)
        earliest_onset = prediction_horizon
        latest_onset = max(n_hours - prediction_horizon, earliest_onset + 1)
        onset_hour = int(rng.integers(earliest_onset, latest_onset))
        df = _generate_patient_series(n_hours, True, onset_hour, rng)
        all_series.append(df)
        all_labels.append(1)
        all_meta.append({"patient_id": f"S{i:04d}", "septic": True, "onset_hour": onset_hour, "n_hours": n_hours})

    # Generate control patients
    for i in range(n_control):
        n_hours = int(rng.integers(min_hours, max_hours))
        df = _generate_patient_series(n_hours, False, onset_hour=9999, rng=rng)
        all_series.append(df)
        all_labels.append(0)
        all_meta.append({"patient_id": f"C{i:04d}", "septic": False, "onset_hour": None, "n_hours": n_hours})

    labels = np.array(all_labels)
    meta = pd.DataFrame(all_meta)

    # Stratified 70/15/15 split
    sepsis_idx = np.where(labels == 1)[0]
    control_idx = np.where(labels == 0)[0]
    rng.shuffle(sepsis_idx)
    rng.shuffle(control_idx)

    def split_indices(idx):
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    s_train, s_val, s_test = split_indices(sepsis_idx)
    c_train, c_val, c_test = split_indices(control_idx)

    splits = {
        "train": (np.concatenate([s_train, c_train])),
        "val":   (np.concatenate([s_val,   c_val])),
        "test":  (np.concatenate([s_test,  c_test])),
    }

    dataset = {}
    for split_name, idx in splits.items():
        split_series = [all_series[i] for i in idx]
        split_labels = labels[idx]
        split_meta   = meta.iloc[idx].reset_index(drop=True)

        # Save metadata
        split_meta.to_csv(output_path / f"{split_name}_meta.csv", index=False)
        np.save(output_path / f"{split_name}_labels.npy", split_labels)

        # Save each series as parquet for efficiency
        series_dir = output_path / split_name
        series_dir.mkdir(exist_ok=True)
        for j, (df, pid) in enumerate(zip(split_series, split_meta["patient_id"])):
            df.to_parquet(series_dir / f"{pid}.parquet", index=False)

        dataset[split_name] = (split_series, split_labels)

    print(f"Dataset saved to {output_path}")
    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val:   {len(splits['val'])} patients")
    print(f"  Test:  {len(splits['test'])} patients")
    print(f"  Sepsis prevalence: {labels.mean()*100:.1f}%")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sepsis dataset")
    parser.add_argument("--n-sepsis",  type=int, default=570)
    parser.add_argument("--n-control", type=int, default=5618)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    type=str, default="data/processed")
    args = parser.parse_args()

    generate_dataset(
        n_sepsis=args.n_sepsis,
        n_control=args.n_control,
        seed=args.seed,
        output_dir=args.output,
    )
