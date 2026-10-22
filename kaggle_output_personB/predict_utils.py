
import json, joblib, pandas as pd

_model = joblib.load("models/best_model.joblib")
_meta = json.load(open("models/model_meta.json"))
BASE = _meta["base_features"]; FEATS = _meta["features"]; THR = _meta["threshold"]

def predict_potability(sample: dict) -> dict:
    # sample: raw water measurements, e.g. {"ph": 7.1, "Hardness": 190, ...}. Missing/None values are allowed
    # for ph, Sulfate, Trihalomethanes and are filled with the training median (flag set to 1), as in training.
    row = {}
    for c in BASE:
        v = sample.get(c)
        missing = v is None or v != v
        row[c] = _meta["train_medians"][c] if missing else float(v)
        if f"{c}_was_missing" in FEATS:
            row[f"{c}_was_missing"] = int(missing)
    X = pd.DataFrame([row])[FEATS]
    p = float(_model.predict_proba(X)[0, 1])
    return {"probability_potable": round(p, 4), "potable": bool(p >= THR), "threshold": THR}
