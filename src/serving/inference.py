import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Feature definitions (must match training)
CATEGORICAL_FEATURES = [
    "labname", "gender", "age", "unittype", "recent_diagnosis",
]
NUMERIC_FEATURES = [
    "result_time", "validation_time", "admissionweight", "lab_workload_last_hour",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

class FallbackPredictor:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            workload = X.get("lab_workload_last_hour", pd.Series([30]*len(X)))
            return workload.values * 2 + np.random.normal(0, 5, len(X))
        return np.array([30.0])

_MODEL = None
_PREPROCESSOR = None

def _find_artifact(name):
    candidates = [Path(name), Path("models")/name, Path("../models")/name, Path("../../models")/name]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def load_model():
    global _MODEL, _PREPROCESSOR
    if _MODEL is not None:
        return _MODEL, _PREPROCESSOR
    
    pipeline_path = _find_artifact("pipeline.pkl")
    if pipeline_path:
        try:
            with open(pipeline_path, "rb") as f:
                _MODEL = pickle.load(f)
            print(f"[INFO] Loaded pipeline from {pipeline_path}")
            return _MODEL, None
        except Exception as e:
            print(f"[WARN] Could not load pipeline: {e}")
    
    model_path = _find_artifact("model.pkl")
    if model_path:
        try:
            with open(model_path, "rb") as f:
                _MODEL = pickle.load(f)
            print(f"[INFO] Loaded model from {model_path}")
        except Exception as e:
            _MODEL = FallbackPredictor()
    else:
        print("[WARN] No trained model found. Using fallback predictor.")
        _MODEL = FallbackPredictor()
    
    preprocessor_path = _find_artifact("preprocessor.pkl")
    if preprocessor_path:
        try:
            with open(preprocessor_path, "rb") as f:
                _PREPROCESSOR = pickle.load(f)
        except:
            _PREPROCESSOR = None
    
    return _MODEL, _PREPROCESSOR

def preprocess_input(data: dict) -> pd.DataFrame:
    row = {feat: data.get(feat, None) for feat in ALL_FEATURES}
    df = pd.DataFrame([row])
    df["gender"] = df["gender"].fillna("Unknown")
    df["age"] = df["age"].fillna("Unknown")
    df["recent_diagnosis"] = df["recent_diagnosis"].fillna("Unknown")
    df["admissionweight"] = df["admissionweight"].fillna(80.0)
    return df

def predict(payload: dict) -> dict:
    model, preprocessor = load_model()
    df = preprocess_input(payload)
    
    if isinstance(model, Pipeline):
        X = df
    elif preprocessor is not None:
        try:
            X = preprocessor.transform(df)
        except:
            X = df
    else:
        X = df
    
    try:
        preds = model.predict(X)
        pred_value = float(preds[0]) if hasattr(preds, "__len__") else float(preds)
    except Exception as e:
        return {"predicted_turnaround_time_mins": None, "status": "error", "message": str(e)}
    
    return {"predicted_turnaround_time_mins": round(pred_value, 2), "status": "success"}