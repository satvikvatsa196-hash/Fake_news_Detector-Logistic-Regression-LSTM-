"""
Model training pipeline for fake news detection.
Supports Logistic Regression (TF-IDF) and a simulated LSTM evaluation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings("ignore")

from utils.features import LinguisticFeatureExtractor, clean_text


def _clean_texts(X):
    return [clean_text(t) for t in X]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """
    Logistic Regression pipeline:
      TF-IDF (word + char n-grams)  +  hand-crafted linguistic features
      → FeatureUnion → Logistic Regression
    """
    tfidf_word = Pipeline([
        ("clean", FunctionTransformer(_clean_texts)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,
            min_df=2,
            analyzer="word",
        )),
    ])

    tfidf_char = Pipeline([
        ("clean", FunctionTransformer(_clean_texts)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=10_000,
            sublinear_tf=True,
            min_df=3,
            analyzer="char_wb",
        )),
    ])

    linguistic = LinguisticFeatureExtractor()

    features = FeatureUnion([
        ("tfidf_word", tfidf_word),
        ("tfidf_char", tfidf_char),
        ("linguistic", linguistic),
    ])

    pipeline = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Train the LR pipeline, evaluate on held-out test set,
    compute full metrics, and persist the model.
    Returns a results dict consumed by the Streamlit app.
    """
    X = df["full_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # ── Train ──────────────────────────────────────────────────────────────
    pipeline = build_lr_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Predictions ────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ── Core metrics ───────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # ── Curves ─────────────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)

    # ── Cross-validation ───────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    # ── Top features ───────────────────────────────────────────────────────
    top_features = get_top_features(pipeline, n=20)

    # ── Persist model ──────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "lr_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    results = {
        "model": pipeline,
        "model_path": model_path,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        },
        "confusion_matrix": cm,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {"precision": prec_curve, "recall": rec_curve, "thresholds": pr_thresholds},
        "cv_scores": cv_scores,
        "top_features": top_features,
        "split": {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Real", "Fake"]
        ),
    }
    return results


def get_top_features(pipeline: Pipeline, n: int = 20) -> dict:
    """Extract top positive/negative LR coefficients with feature names."""
    try:
        # Feature names: TF-IDF word, TF-IDF char, linguistic
        fu = pipeline.named_steps["features"]
        word_names = fu.transformer_list[0][1].named_steps["tfidf"].get_feature_names_out()
        char_names = [f"char::{n}" for n in fu.transformer_list[1][1].named_steps["tfidf"].get_feature_names_out()]
        ling_names = LinguisticFeatureExtractor().get_feature_names_out()
        all_names = np.concatenate([word_names, char_names, ling_names])

        coefs = pipeline.named_steps["clf"].coef_[0]
        top_pos_idx = np.argsort(coefs)[-n:][::-1]
        top_neg_idx = np.argsort(coefs)[:n]

        return {
            "fake_features": [(all_names[i], coefs[i]) for i in top_pos_idx],
            "real_features": [(all_names[i], coefs[i]) for i in top_neg_idx],
        }
    except Exception:
        return {"fake_features": [], "real_features": []}


def load_model(path: str = None) -> Pipeline | None:
    """Load a persisted model from disk."""
    if path is None:
        path = os.path.join(MODEL_DIR, "lr_pipeline.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def predict_article(text: str, pipeline: Pipeline) -> dict:
    """Run inference on a single article text."""
    proba = pipeline.predict_proba([text])[0]
    label = pipeline.predict([text])[0]
    from utils.features import LinguisticFeatureExtractor
    ling = LinguisticFeatureExtractor().transform([text])[0]
    ling_names = LinguisticFeatureExtractor().get_feature_names_out()
    return {
        "label": int(label),
        "label_name": "fake" if label == 1 else "real",
        "prob_real": float(proba[0]),
        "prob_fake": float(proba[1]),
        "confidence": float(max(proba)),
        "features": dict(zip(ling_names, ling.tolist())),
    }
