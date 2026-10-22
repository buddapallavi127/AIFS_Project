# Person C – AquaGuard AI Dashboard (App & Deployment)

## Overview
A **Streamlit** dashboard that loads Person B's trained Soft-Voting Ensemble model
and provides an interactive UI for water potability prediction.

## Features

| Page | What it does |
|------|-------------|
| 🔬 **Predict** | Single-sample prediction with live SHAP waterfall chart, or batch CSV upload with downloadable results |
| 📊 **Model Performance** | Test metrics, radar chart, model comparison bar charts, cross-validation tables, evaluation plots |
| 🧠 **Feature Insights** | SHAP bar/beeswarm plots, permutation importance, interactive feature range explorer |

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
personC_app/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file

kaggle_output_personB/   # (produced by Person B, read by this app)
├── models/
│   ├── best_model.joblib
│   └── model_meta.json
└── outputs_B/
    ├── baseline_cv.csv
    ├── tuned_cv.csv
    ├── results_table.csv
    ├── *.png (evaluation plots)
    └── ...
```

## Model Details

- **Model:** Soft-Voting Ensemble (Random Forest + XGBoost + LightGBM + SVM)
- **Threshold:** 0.38 (tuned for recall)
- **Features:** 9 physicochemical parameters + 3 missing-value flags
- **Test ROC-AUC:** 67.0%
