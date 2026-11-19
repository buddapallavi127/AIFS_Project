"""
Person B (Multimodal) — Train Models on Real Datasets
======================================================
1. Vision CNN: MobileNet-style lightweight classifier on Zenodo TSS images
2. Classical Vision ML: Random Forest on extracted image features
3. Numerical ML: Ensemble on Figshare water quality parameters

Can run on: Colab (GPU), Kaggle, or local CPU (slower).

Outputs:
  - models/vision_model.pth  (CNN weights — requires PyTorch)
  - models/vision_rf.joblib  (classical vision Random Forest)
  - models/numerical_model.joblib  (numerical ensemble)
  - models/multimodal_meta.json  (metadata for both models)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    balanced_accuracy_score, precision_score, recall_score,
    accuracy_score
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "personC_app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. CLASSICAL VISION MODEL (Random Forest on image features)
# ═══════════════════════════════════════════════════════════════════════

VISION_FEATURE_COLS = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_brightness", "std_brightness",
    "blue_dominance", "rg_ratio", "rb_ratio",
    "contrast", "edge_density", "entropy",
    "brown_index", "green_index",
]


def train_vision_rf():
    """Train a classical vision Random Forest on extracted image features."""
    print("\n" + "=" * 60)
    print("TRAINING VISION MODEL (Random Forest on image features)")
    print("=" * 60)

    feat_path = PROCESSED_DIR / "image_features.csv"
    if not feat_path.exists():
        print("  ⚠️  image_features.csv not found. Run data_pipeline.py first.")
        return None

    df = pd.read_csv(feat_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train = train[VISION_FEATURE_COLS].values
    y_train_tss = train["class_id"].values      # 0=low, 1=medium, 2=high
    y_train_pot = train["potable"].values        # binary

    X_val = val[VISION_FEATURE_COLS].values
    y_val_tss = val["class_id"].values
    y_val_pot = val["potable"].values

    X_test = test[VISION_FEATURE_COLS].values
    y_test_tss = test["class_id"].values
    y_test_pot = test["potable"].values

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Features: {len(VISION_FEATURE_COLS)}")

    # --- TSS 3-class model ---
    print("\n  Training TSS 3-class classifier...")
    tss_model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    tss_model.fit(X_train, y_train_tss)

    tss_val_pred = tss_model.predict(X_val)
    tss_val_acc = accuracy_score(y_val_tss, tss_val_pred)
    tss_test_pred = tss_model.predict(X_test)
    tss_test_acc = accuracy_score(y_test_tss, tss_test_pred)

    print(f"  TSS Val Accuracy: {tss_val_acc:.3f}")
    print(f"  TSS Test Accuracy: {tss_test_acc:.3f}")
    print("\n  Test Classification Report (TSS):")
    print(classification_report(y_test_tss, tss_test_pred,
                                target_names=["Low", "Medium", "High"]))

    # --- Binary potability model ---
    print("  Training binary potability classifier...")
    pot_model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pot_model.fit(X_train, y_train_pot)

    pot_val_proba = pot_model.predict_proba(X_val)[:, 1]
    pot_test_proba = pot_model.predict_proba(X_test)[:, 1]
    pot_val_auc = roc_auc_score(y_val_pot, pot_val_proba)
    pot_test_auc = roc_auc_score(y_test_pot, pot_test_proba)

    pot_test_pred = (pot_test_proba >= 0.5).astype(int)
    pot_test_f1 = f1_score(y_test_pot, pot_test_pred)

    print(f"  Val ROC-AUC: {pot_val_auc:.3f}")
    print(f"  Test ROC-AUC: {pot_test_auc:.3f}, F1: {pot_test_f1:.3f}")

    # Save both models
    joblib.dump(tss_model, MODEL_DIR / "vision_tss_model.joblib")
    joblib.dump(pot_model, MODEL_DIR / "vision_potability_model.joblib")

    # Feature importance
    rf_model = pot_model.named_steps["rf"]
    importances = dict(zip(VISION_FEATURE_COLS,
                           [float(x) for x in rf_model.feature_importances_]))

    print(f"\n  ✅ Vision models saved to {MODEL_DIR}")

    return {
        "tss_test_accuracy": round(tss_test_acc, 4),
        "potability_test_roc_auc": round(pot_test_auc, 4),
        "potability_test_f1": round(pot_test_f1, 4),
        "feature_importances": importances,
        "features": VISION_FEATURE_COLS,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. NUMERICAL ML MODEL (Ensemble on Figshare parameters)
# ═══════════════════════════════════════════════════════════════════════

NUMERICAL_FEATURES = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]


def train_numerical_model():
    """Train numerical ML ensemble on Figshare water quality dataset."""
    print("\n" + "=" * 60)
    print("TRAINING NUMERICAL MODEL (Ensemble on water quality parameters)")
    print("=" * 60)

    num_dir = PROCESSED_DIR / "numerical"
    train_path = num_dir / "train.csv"
    test_path = num_dir / "test.csv"

    if not train_path.exists():
        print("  ⚠️  Numerical data not found. Run data_pipeline.py first.")
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[NUMERICAL_FEATURES].values
    y_train = train_df["potable"].values
    X_test = test_df[NUMERICAL_FEATURES].values
    y_test = test_df["potable"].values

    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")
    print(f"  Features: {len(NUMERICAL_FEATURES)}")
    print(f"  Class balance — Train: {y_train.mean():.3f} potable, Test: {y_test.mean():.3f} potable")

    # Subsample for faster training if dataset is very large
    MAX_TRAIN = 100_000
    if len(X_train) > MAX_TRAIN:
        print(f"  Subsampling train to {MAX_TRAIN:,} for tractable training...")
        idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Build ensemble
    print("\n  Training Random Forest...")
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train_sub, y_train_sub)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"    RF Test ROC-AUC: {rf_auc:.4f}")

    print("  Training Gradient Boosting...")
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])
    gb.fit(X_train_sub, y_train_sub)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"    GB Test ROC-AUC: {gb_auc:.4f}")

    # Soft-voting ensemble
    print("  Building Soft-Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[("RF", rf), ("GB", gb)],
        voting="soft",
    )
    ensemble.fit(X_train_sub, y_train_sub)

    # Evaluate
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_acc": round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n  Ensemble Test Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Potable", "Potable"]))

    # Save
    joblib.dump(ensemble, MODEL_DIR / "numerical_model.joblib")

    # Compute medians for imputation
    train_medians = {col: float(train_df[col].median()) for col in NUMERICAL_FEATURES}

    # Feature ranges
    feature_ranges = {}
    for col in NUMERICAL_FEATURES:
        feature_ranges[col] = [float(train_df[col].quantile(0.01)),
                               float(train_df[col].quantile(0.99))]

    print(f"\n  ✅ Numerical model saved to {MODEL_DIR}")

    return {
        "metrics": metrics,
        "individual_aucs": {"RF": round(rf_auc, 4), "GB": round(gb_auc, 4)},
        "features": NUMERICAL_FEATURES,
        "train_medians": train_medians,
        "feature_ranges": feature_ranges,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. SAVE COMBINED METADATA
# ═══════════════════════════════════════════════════════════════════════

def save_multimodal_meta(vision_results, numerical_results):
    """Save combined metadata for the multimodal dashboard."""
    meta = {
        "vision": vision_results,
        "numerical": numerical_results,
        "fusion": {
            "method": "Dempster-Shafer Evidence Fusion",
            "vision_base_weight": 0.30,
            "numerical_base_weight": 0.70,
        },
        "datasets": {
            "vision_source": "Zenodo 10.5281/zenodo.19899828 — TSS Image Classification",
            "numerical_source": "Nature Scientific Data 10.1038/s41597-025-04715-4 — Surface Water Quality 1940-2023",
        },
    }

    with open(MODEL_DIR / "multimodal_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Multimodal metadata saved → {MODEL_DIR / 'multimodal_meta.json'}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧬 Multimodal Water Intelligence — Model Training Pipeline")
    print("=" * 60)

    vision_results = train_vision_rf()
    numerical_results = train_numerical_model()

    if vision_results and numerical_results:
        save_multimodal_meta(vision_results, numerical_results)

    print("\n" + "=" * 60)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 60)
