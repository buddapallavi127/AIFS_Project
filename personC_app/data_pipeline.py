"""
Data Processing Pipeline — Multimodal Water Quality Datasets
=============================================================
Processes both datasets for the multimodal system:

1. Zenodo Image Dataset (TSS classification: Baja/Media/Alta → Low/Medium/High)
   - Splits into train/val/test (70/15/15) stratified
   - Computes per-image feature vectors for classical ML baseline

2. Figshare Numerical Dataset (Surface water quality 1940-2023)
   - Cleans and maps CCME WQI to binary potability
   - Creates train/test splits for the numerical model

Outputs everything to datasets/processed/
"""

import os, json, shutil, random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
ZENODO_DIR = DATASETS_DIR / "zenodo" / "datasets"
FIGSHARE_DIR = DATASETS_DIR / "figshare"


# ═══════════════════════════════════════════════════════════════════════
# 1. ZENODO IMAGE DATASET PROCESSING
# ═══════════════════════════════════════════════════════════════════════

# Class mapping (Spanish → English)
TSS_CLASS_MAP = {
    "Baja": "low",      # Low Total Suspended Solids (clear)
    "Media": "medium",   # Medium TSS
    "Alta": "high",      # High TSS (turbid/polluted)
}

# Binary mapping for potability
TSS_POTABILITY = {
    "low": 1,       # Safe / potable
    "medium": 0,    # Borderline → unsafe for drinking
    "high": 0,      # Unsafe
}


def extract_image_features(img_path: str) -> dict:
    """Extract classical CV features from a water image for ML baseline."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    grey = 0.299 * r + 0.587 * g + 0.114 * b

    features = {
        # Colour channel means
        "mean_r": float(np.mean(r)),
        "mean_g": float(np.mean(g)),
        "mean_b": float(np.mean(b)),
        # Colour channel stds
        "std_r": float(np.std(r)),
        "std_g": float(np.std(g)),
        "std_b": float(np.std(b)),
        # Brightness
        "mean_brightness": float(np.mean(grey)),
        "std_brightness": float(np.std(grey)),
        # Colour ratios
        "blue_dominance": float(np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)),
        "rg_ratio": float(np.mean(r) / (np.mean(g) + 1e-6)),
        "rb_ratio": float(np.mean(r) / (np.mean(b) + 1e-6)),
        # Contrast (Laplacian variance on grey)
        "contrast": float(np.std(grey)),
        # Edge density
        "edge_density": float(np.mean(np.abs(np.diff(grey, axis=0))) + np.mean(np.abs(np.diff(grey, axis=1)))),
        # Entropy approximation
        "entropy": float(_shannon_entropy(grey)),
        # Brown index (high R relative to B = muddy)
        "brown_index": float(np.mean(np.clip(r - b, 0, 1))),
        # Green index
        "green_index": float(np.mean(np.clip(g - (r + b) / 2, 0, 1))),
    }
    return features


def _shannon_entropy(grey: np.ndarray) -> float:
    hist, _ = np.histogram((grey * 255).astype(np.uint8), bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def process_zenodo_images():
    """Process Zenodo image dataset into train/val/test splits + feature CSV."""
    print("\n" + "=" * 60)
    print("PROCESSING ZENODO IMAGE DATASET")
    print("=" * 60)

    image_dir = PROCESSED_DIR / "images"
    for split in ["train", "val", "test"]:
        for cls in ["low", "medium", "high"]:
            (image_dir / split / cls).mkdir(parents=True, exist_ok=True)

    all_records = []

    for spanish_cls, eng_cls in TSS_CLASS_MAP.items():
        src_dir = ZENODO_DIR / spanish_cls
        if not src_dir.exists():
            print(f"  ⚠️  {src_dir} not found, skipping")
            continue

        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
        random.shuffle(files)
        n = len(files)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        print(f"\n  {eng_cls.upper()} ({spanish_cls}): {n} images → "
              f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        for split_name, split_files in splits.items():
            for fname in split_files:
                src = src_dir / fname
                dst = image_dir / split_name / eng_cls / fname
                if not dst.exists():
                    shutil.copy2(src, dst)
                all_records.append({
                    "filename": fname,
                    "class": eng_cls,
                    "class_id": ["low", "medium", "high"].index(eng_cls),
                    "potable": TSS_POTABILITY[eng_cls],
                    "split": split_name,
                    "path": str(dst),
                })

    df = pd.DataFrame(all_records)
    df.to_csv(PROCESSED_DIR / "image_manifest.csv", index=False)
    print(f"\n  ✅ Manifest saved: {len(df)} records → image_manifest.csv")

    # Extract features for a classical ML baseline
    print("\n  Extracting visual features (this may take a few minutes)...")
    feature_records = []
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"    Processing image {i + 1}/{len(df)}...")
        try:
            feats = extract_image_features(row["path"])
            feats["filename"] = row["filename"]
            feats["class"] = row["class"]
            feats["class_id"] = row["class_id"]
            feats["potable"] = row["potable"]
            feats["split"] = row["split"]
            feature_records.append(feats)
        except Exception as e:
            print(f"    ⚠️  Error on {row['filename']}: {e}")

    feat_df = pd.DataFrame(feature_records)
    feat_df.to_csv(PROCESSED_DIR / "image_features.csv", index=False)
    print(f"  ✅ Features saved: {len(feat_df)} records → image_features.csv")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. FIGSHARE NUMERICAL DATASET PROCESSING
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

CCME_BINARY_MAP = {
    "Excellent": 1,
    "Good": 1,
    "Fair": 0,
    "Marginal": 0,
    "Poor": 0,
}


def process_figshare_numerical():
    """Process Figshare numerical dataset for ML training."""
    print("\n" + "=" * 60)
    print("PROCESSING FIGSHARE NUMERICAL DATASET")
    print("=" * 60)

    num_dir = PROCESSED_DIR / "numerical"
    num_dir.mkdir(parents=True, exist_ok=True)

    # Load all available regional CSVs
    dfs = []
    for csv_file in sorted(FIGSHARE_DIR.glob("*_dataset.csv")):
        df = pd.read_csv(csv_file)
        print(f"  Loaded {csv_file.name}: {len(df):,} rows")
        dfs.append(df)

    if not dfs:
        print("  ⚠️  No numerical CSVs found!")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows")

    # Clean
    combined = combined.dropna(subset=NUMERICAL_FEATURES + ["CCME_WQI"])

    # Map CCME WQI to binary potability
    combined["potable"] = combined["CCME_WQI"].map(CCME_BINARY_MAP)
    combined = combined.dropna(subset=["potable"])
    combined["potable"] = combined["potable"].astype(int)

    print(f"  After cleaning: {len(combined):,} rows")
    print(f"  Class distribution:")
    print(f"    Potable (Excellent/Good): {(combined['potable'] == 1).sum():,}")
    print(f"    Not Potable (Fair/Marginal/Poor): {(combined['potable'] == 0).sum():,}")

    # Stratified train/test split (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        combined, test_size=0.20, random_state=42,
        stratify=combined["potable"],
    )

    # Save
    train_df.to_csv(num_dir / "train.csv", index=False)
    test_df.to_csv(num_dir / "test.csv", index=False)

    # Save metadata
    meta = {
        "features": NUMERICAL_FEATURES,
        "target": "potable",
        "ccme_mapping": CCME_BINARY_MAP,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "class_distribution": {
            "train": {"potable": int((train_df["potable"] == 1).sum()),
                      "not_potable": int((train_df["potable"] == 0).sum())},
            "test": {"potable": int((test_df["potable"] == 1).sum()),
                     "not_potable": int((test_df["potable"] == 0).sum())},
        },
        "feature_stats": {},
    }

    for feat in NUMERICAL_FEATURES:
        meta["feature_stats"][feat] = {
            "train_mean": float(train_df[feat].mean()),
            "train_std": float(train_df[feat].std()),
            "train_median": float(train_df[feat].median()),
            "min": float(combined[feat].min()),
            "max": float(combined[feat].max()),
        }

    with open(num_dir / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Saved train.csv ({len(train_df):,} rows) and test.csv ({len(test_df):,} rows)")
    print(f"  ✅ Saved data_meta.json")

    return combined


# ═══════════════════════════════════════════════════════════════════════
# 3. DATASET SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_summary():
    """Generate a summary JSON for the dashboard."""
    summary = {
        "project": "Multimodal Water Intelligence",
        "datasets": {
            "vision": {
                "source": "Zenodo Record 19899828",
                "doi": "10.5281/zenodo.19899828",
                "description": "Image Dataset for Classification of Water Quality Based on Total Suspended Solids",
                "total_images": 6515,
                "classes": {"low": "Baja (clear water)", "medium": "Media", "high": "Alta (turbid)"},
                "image_size": "450x450 RGB",
                "splits": "70/15/15 train/val/test",
            },
            "numerical": {
                "source": "Nature Scientific Data / Figshare 27800394",
                "doi": "10.1038/s41597-025-04715-4",
                "description": "Comprehensive Surface Water Quality Dataset 1940-2023",
                "parameters": NUMERICAL_FEATURES,
                "target": "CCME Water Quality Index → Binary Potability",
                "splits": "80/20 train/test",
            },
        },
        "references": [
            {
                "title": "Water quality polluted by total suspended solids classified within an artificial neural network approach",
                "journal": "Water Quality Research Journal",
                "doi": "10.2166/wqrj.2024.061",
            },
            {
                "title": "A Comprehensive Dataset of Surface Water Quality Spanning 1940-2023",
                "journal": "Nature Scientific Data",
                "doi": "10.1038/s41597-025-04715-4",
            },
        ],
    }

    with open(PROCESSED_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ Dataset summary saved → dataset_summary.json")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("🔧 Multimodal Water Quality — Data Processing Pipeline")
    print("=" * 60)

    # Process images
    process_zenodo_images()

    # Process numerical
    process_figshare_numerical()

    # Summary
    generate_summary()

    print("\n" + "=" * 60)
    print("✅ ALL PROCESSING COMPLETE")
    print(f"   Output directory: {PROCESSED_DIR}")
    print("=" * 60)
