"""
Model Training for Gujarat Engineering Admission Predictor
===========================================================
Trains multiple ML models on scraped ACPC cutoff data,
compares them, selects the best, and saves it as model.pkl.

Also generates SHAP explainability plots.
"""

import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model.pkl"
ENCODERS_PATH = ROOT_DIR / "encoders.pkl"

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_and_prepare_data():
    """Load student data and prepare features for training."""
    df = pd.read_csv(DATA_DIR / "student_data.csv")
    print(f"Loaded {len(df):,} records")

    # Remove any rows with missing critical features
    df = df.dropna(subset=["closing_rank", "gujcet_score", "twelfth_pcm_percent"])
    df = df[df["branch"].notna() & (df["branch"] != "")]
    df = df[df["college"].notna() & (df["college"] != "")]

    # Encode categorical variables
    le_category = LabelEncoder()
    le_branch = LabelEncoder()
    le_college = LabelEncoder()

    df["category_encoded"] = le_category.fit_transform(df["category"].astype(str))
    df["branch_encoded"] = le_branch.fit_transform(df["branch"].astype(str))
    df["college_encoded"] = le_college.fit_transform(df["college"].astype(str))

    # Feature: ratio of student rank to closing rank
    df["rank_to_cutoff_ratio"] = df["student_rank"] / df["closing_rank"].clip(lower=1)

    # Year as numeric (extract first year from "2020-21" format)
    df["year_num"] = df["year"].astype(str).str[:4].astype(int)

    # Define features
    feature_cols = [
        "gujcet_score",
        "twelfth_pcm_percent",
        "category_encoded",
        "branch_encoded",
        "college_tier",
        "closing_rank",
        "rank_to_cutoff_ratio",
        "year_num",
    ]

    X = df[feature_cols].copy()
    y = df["admitted"].copy()

    encoders = {
        "category": le_category,
        "branch": le_branch,
        "college": le_college,
        "feature_cols": feature_cols,
    }

    return X, y, df, encoders


# ── Train Models ──────────────────────────────────────────────────────────────
def train_and_compare(X, y):
    """Train multiple models and compare their performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            verbosity=0
        ),
    }

    results = {}
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"\n  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}")
    print("-" * 60)

    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        results[name] = {
            "accuracy": accuracy,
            "auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print("=" * 60)

    # Detailed report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred_best, target_names=["Rejected", "Admitted"]))

    return best_model, best_model_name, results, X_train, X_test, y_train, y_test


# ── Feature Importance ────────────────────────────────────────────────────────
def show_feature_importance(model, feature_cols):
    """Display feature importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        print("\n  Feature Importance:")
        print("  " + "-" * 40)
        for _, row in feat_imp.iterrows():
            bar = "#" * int(row["Importance"] * 50)
            print(f"  {row['Feature']:25s} {row['Importance']:.4f} {bar}")


# ── Save Model & Encoders ────────────────────────────────────────────────────
def save_model(model, encoders, model_name, results):
    """Save the trained model and encoders."""
    model_data = {
        "model": model,
        "model_name": model_name,
        "results": results,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to: {MODEL_PATH}")

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"  Encoders saved to: {ENCODERS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Gujarat Engineering Admission Predictor - Model Training")
    print("=" * 60)

    # Load data
    X, y, df, encoders = load_and_prepare_data()
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # Train & compare
    best_model, best_model_name, results, X_train, X_test, y_train, y_test = \
        train_and_compare(X, y)

    # Feature importance
    show_feature_importance(best_model, encoders["feature_cols"])

    # Save
    save_model(best_model, encoders, best_model_name, results)

    # SHAP analysis (optional - can be slow)
    try:
        import shap
        print("\n  Generating SHAP explainability analysis...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test.iloc[:200])
        
        # Save SHAP summary data for the Streamlit app  
        shap_data = {
            "shap_values": shap_values,
            "X_sample": X_test.iloc[:200],
            "feature_names": encoders["feature_cols"],
        }
        shap_path = ROOT_DIR / "shap_data.pkl"
        with open(shap_path, "wb") as f:
            pickle.dump(shap_data, f)
        print(f"  SHAP data saved to: {shap_path}")
    except Exception as e:
        print(f"  SHAP analysis skipped: {e}")

    print("\n  Training complete! Run 'streamlit run app.py' to launch the app.")


if __name__ == "__main__":
    main()
