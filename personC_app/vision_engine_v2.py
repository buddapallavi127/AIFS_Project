"""
Vision AI Engine v2 — Trained on Zenodo TSS Dataset
=====================================================
Uses the trained Random Forest model on extracted image features
(from Zenodo Record 19899828: 6,518 labelled water images).

Falls back to classical heuristic analysis if trained model is unavailable.
"""

import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Tuple

# Import the original engine for fallback and data classes
from vision_engine import (
    analyse_water_image as _classical_analyse,
    VisionAssessment, ColourProfile, ClarityMetrics, ContaminantIndicators,
    _rgb_to_hsv_array, _classify_hue_ranges, _dominant_hue_sat,
    _compute_sharpness, _rms_contrast, _shannon_entropy, _edge_density,
    _map_to_ntu, _clarity_label, _detect_foam, _detect_particles,
    _colour_label,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (must match data_pipeline.py exactly)
# ═══════════════════════════════════════════════════════════════════════

VISION_FEATURE_COLS = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_brightness", "std_brightness",
    "blue_dominance", "rg_ratio", "rb_ratio",
    "contrast", "edge_density", "entropy",
    "brown_index", "green_index",
]


def _extract_features(img: Image.Image, target_size: int = 450) -> np.ndarray:
    """Extract the same feature vector used during training."""
    img = img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    grey = 0.299 * r + 0.587 * g + 0.114 * b

    def _entropy(grey):
        hist, _ = np.histogram((grey * 255).astype(np.uint8), bins=256, range=(0, 255))
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    features = np.array([
        float(np.mean(r)),           # mean_r
        float(np.mean(g)),           # mean_g
        float(np.mean(b)),           # mean_b
        float(np.std(r)),            # std_r
        float(np.std(g)),            # std_g
        float(np.std(b)),            # std_b
        float(np.mean(grey)),        # mean_brightness
        float(np.std(grey)),         # std_brightness
        float(np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)),  # blue_dominance
        float(np.mean(r) / (np.mean(g) + 1e-6)),  # rg_ratio
        float(np.mean(r) / (np.mean(b) + 1e-6)),  # rb_ratio
        float(np.std(grey)),         # contrast
        float(np.mean(np.abs(np.diff(grey, axis=0))) + np.mean(np.abs(np.diff(grey, axis=1)))),  # edge_density
        _entropy(grey),              # entropy
        float(np.mean(np.clip(r - b, 0, 1))),   # brown_index
        float(np.mean(np.clip(g - (r + b) / 2, 0, 1))),  # green_index
    ])
    return features.reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════
# LOAD TRAINED MODELS
# ═══════════════════════════════════════════════════════════════════════

_tss_model = None
_pot_model = None
_models_loaded = False


def _load_models():
    global _tss_model, _pot_model, _models_loaded
    if _models_loaded:
        return

    tss_path = MODEL_DIR / "vision_tss_model.joblib"
    pot_path = MODEL_DIR / "vision_potability_model.joblib"

    if tss_path.exists() and pot_path.exists():
        try:
            _tss_model = joblib.load(str(tss_path))
            _pot_model = joblib.load(str(pot_path))
            _models_loaded = True
            print("[VisionV2] Trained models loaded successfully.")
        except Exception as e:
            print(f"[VisionV2] Warning: Could not load models: {e}")
            _models_loaded = True  # don't retry
    else:
        _models_loaded = True


# ═══════════════════════════════════════════════════════════════════════
# TSS CLASS LABELS
# ═══════════════════════════════════════════════════════════════════════

TSS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
TSS_DESCRIPTIONS = {
    "Low": "Clear water with minimal suspended solids — safe for consumption after standard treatment.",
    "Medium": "Moderate suspended solids detected — requires filtration before consumption.",
    "High": "High suspended solids concentration — unsafe without advanced treatment (coagulation + filtration).",
}


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def analyse_water_image_v2(image: Image.Image) -> dict:
    """
    Enhanced Vision AI analysis using trained ML model + classical CV.

    Returns a dict with:
      - 'assessment': VisionAssessment from classical CV
      - 'ml_tss_class': predicted TSS class (Low/Medium/High) from trained model
      - 'ml_tss_probabilities': class probabilities
      - 'ml_potability_prob': P(potable) from trained model
      - 'ml_confidence': model confidence
      - 'model_available': whether the trained model was used
      - 'features': extracted feature vector (for display)
    """
    _load_models()

    # Always run classical analysis (for colour, turbidity, contaminant details)
    classical = _classical_analyse(image)

    result = {
        "assessment": classical,
        "model_available": _tss_model is not None,
        "ml_tss_class": None,
        "ml_tss_probabilities": {},
        "ml_potability_prob": None,
        "ml_confidence": 0.0,
        "features": {},
    }

    if _tss_model is not None and _pot_model is not None:
        # Extract features
        X = _extract_features(image)

        # TSS classification
        tss_pred = int(_tss_model.predict(X)[0])
        tss_proba = _tss_model.predict_proba(X)[0]
        result["ml_tss_class"] = TSS_LABELS[tss_pred]
        result["ml_tss_probabilities"] = {
            TSS_LABELS[i]: round(float(p), 4) for i, p in enumerate(tss_proba)
        }

        # Potability
        pot_proba = float(_pot_model.predict_proba(X)[0, 1])
        result["ml_potability_prob"] = round(pot_proba, 4)
        result["ml_confidence"] = round(float(max(tss_proba)), 3)

        # Feature values for display
        result["features"] = dict(zip(VISION_FEATURE_COLS, [round(float(v), 4) for v in X[0]]))

        # Override classical score with ML-informed score
        # Blend: 70% ML potability + 30% classical score
        ml_score = pot_proba * 100
        classical_score = classical.visual_quality_score
        blended = 0.70 * ml_score + 0.30 * classical_score

        # Update the assessment with blended score
        result["blended_score"] = round(blended, 1)
        result["ml_score"] = round(ml_score, 1)
        result["classical_score"] = round(classical_score, 1)
    else:
        # Fallback: use classical scores
        result["blended_score"] = classical.visual_quality_score
        result["ml_score"] = None
        result["classical_score"] = classical.visual_quality_score

    return result
