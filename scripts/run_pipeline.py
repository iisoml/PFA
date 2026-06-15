#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → train → evaluate
"""

import os
import sys
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules
from load_data import load_data
from preprocess import preprocess_data
from build_features import build_features

def validate_lab_data(df):
    """
    Basic data quality validation for lab prediction data.
    Returns (is_valid, list of issues).
    """
    issues = []
    
    required_cols = [
        "labid", "labname", "result_time", "validation_time",
        "turnaround_time_mins", "gender", "age", "unittype",
        "admissionweight", "recent_diagnosis", "lab_workload_last_hour"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    if "turnaround_time_mins" in df.columns:
        if df["turnaround_time_mins"].isna().sum() > 0:
            issues.append("Target has NaN values")
        if (df["turnaround_time_mins"] <= 0).any():
            issues.append("Target has non-positive values")
    
    return len(issues) == 0, issues


def main(args):
    """
    Main training pipeline for lab turnaround time prediction.
    """
    
    # === MLflow Setup ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # === Log configuration ===
        mlflow.log_param("model", "xgboost_regressor")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)

        # === STAGE 1: Data Loading ===
        print("Loading data...")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === STAGE 2: Data Validation ===
        print("Validating data quality...")
        is_valid, issues = validate_lab_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(issues, indent=2), artifact_file="validation_issues.json")
            raise ValueError(f"Data validation failed: {issues}")
        else:
            print("Data validation passed.")

        # === STAGE 3: Preprocessing ===
        print("Preprocessing data...")
        df = preprocess_data(df, target_column=args.target)

        processed_path = os.path.join(project_root, "data", "processed", "lab_pred_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 4: Feature Engineering ===
        print("Building features...")
        target = args.target
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc, selected_features = build_features(df, target_column=target)
        
        # Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        
        print(f"Feature engineering completed: {len(selected_features)} features selected")

        # === Save Feature Metadata ===
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = selected_features  # Already excludes target from build_features
        
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 5: Train/Test Split ===
        print("Splitting data...")
        X = df_enc[feature_cols]
        y = df_enc[target]
        
        # No stratify for regression (continuous target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === STAGE 6: Model Training ===
        print("Training XGBoost regressor...")
        
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 7: Model Evaluation (Regression Metrics) ===
        print("Evaluating model performance...")
        
        t1 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t1
        
        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("pred_time", pred_time)
        
        print(f"   Model Performance:")
        print(f"   MAE:  {mae:.2f} minutes")
        print(f"   RMSE: {rmse:.2f} minutes")
        print(f"   R²:   {r2:.4f}")

        # === Residual Analysis ===
        residuals = y_test - y_pred
        mlflow.log_metric("residual_mean", residuals.mean())
        mlflow.log_metric("residual_std", residuals.std())
        
        # Log prediction vs actual scatter data for visualization
        pred_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": residuals.values
        })
        pred_df.to_csv(os.path.join(artifacts_dir, "predictions.csv"), index=False)
        mlflow.log_artifact(os.path.join(artifacts_dir, "predictions.csv"))

        # === Feature Importance ===
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importance.nlargest(20)
        mlflow.log_text(top_features.to_string(), artifact_file="feature_importance.txt")
        print(f"\n Top 10 Important Features:")
        print(top_features.head(10).to_string())

        # === STAGE 8: Model Serialization ===
        print("Saving model to MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Also save locally
        model_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print("Model saved to MLflow")

        # === Performance Summary ===
        print(f"\n Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Residual Statistics:")
        print(f"   Mean residual: {residuals.mean():.2f} mins")
        print(f"   Std residual:  {residuals.std():.2f} mins")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run lab TAT prediction pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., lab_pred.csv)")
    p.add_argument("--target", type=str, default="turnaround_time_mins")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--experiment", type=str, default="Lab Turnaround Time")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI")

    args = p.parse_args()
    main(args)

"""
# Run the pipeline:
python run_pipeline.py --input lab_pred.csv
"""