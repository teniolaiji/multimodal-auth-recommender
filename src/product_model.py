"""Product recommendation model: predicts product_category from the merged dataset.

Train:    python src/product_model.py
Predict:  python src/product_model.py --customer 101

From the app:
    from product_model import predict_for_customer
    category, probs = predict_for_customer(101)

Evaluation is split BY CUSTOMER (a customer's purchases are all in train or all in test).
Several rows share the same customer-level social features, so a random row split would
let the model memorise customers and overstate accuracy.
"""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MERGED_CSV, PRODUCT_METRICS_PATH, PRODUCT_MODEL_PATH, RANDOM_STATE, TARGET

NUMERIC_FEATURES = [
    "purchase_amount", "customer_rating", "rating_missing",
    "purchase_month", "purchase_dayofweek", "is_weekend",
    "prior_purchases", "prev_amount",
    "n_social_profiles", "n_platforms",
    "engagement_mean", "engagement_max", "interest_mean", "interest_max",
    "sentiment_mean", "pct_positive", "pct_negative", "engagement_x_interest",
    "on_facebook", "on_instagram", "on_linkedin", "on_tiktok", "on_twitter",
]
CATEGORICAL_FEATURES = ["prev_category", "top_platform"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

try:  # XGBoost is in requirements.txt but optional here
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def load_data():
    df = pd.read_csv(MERGED_CSV, keep_default_na=False, na_values=[""])
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def candidate_models():
    models = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Baseline (class prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(C=0.3, max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE, eval_metric="mlogloss")
    return models


def make_pipeline(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


class _LabelEncodedPipeline:
    """Wraps a pipeline so XGBoost (which needs integer labels) takes string labels too."""

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.unique(y)))
        self.pipe.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build(name, model):
    pipe = make_pipeline(model)
    return _LabelEncodedPipeline(pipe) if name == "XGBoost" else pipe


def scores(y_true, proba, classes):
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=classes),
    }


def cross_validate(df, n_splits=5, seeds=(0, 1, 2)):
    """Grouped, stratified CV repeated over a few seeds -> mean +/- std per model."""
    X, y, groups = df[FEATURES], df[TARGET].to_numpy(), df["customer_id"]
    classes = sorted(np.unique(y))
    rows = []
    for name, model in candidate_models().items():
        fold_scores = []
        for seed in seeds:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups):
                est = build(name, model).fit(X.iloc[tr], y[tr])
                proba = _proba_full(est, X.iloc[te], classes)
                fold_scores.append(scores(y[te], proba, classes))
        fs = pd.DataFrame(fold_scores)
        rows.append({"model": name,
                     **{f"{m}_mean": fs[m].mean() for m in fs},
                     **{f"{m}_std": fs[m].std() for m in fs}})
    return pd.DataFrame(rows).set_index("model")


def _proba_full(est, X, classes):
    """predict_proba with a column for every class, even ones missing from a train fold."""
    proba = est.predict_proba(X)
    est_classes = list(est.classes_)
    full = np.full((len(X), len(classes)), 1e-6)
    for j, c in enumerate(est_classes):
        full[:, classes.index(c)] = proba[:, j]
    return full / full.sum(axis=1, keepdims=True)


def holdout_split(df, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, df[TARGET], df["customer_id"]))
    return df.iloc[tr], df.iloc[te]


def train(verbose=True):
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_data()
    classes = sorted(df[TARGET].unique())

    cv = cross_validate(df)
    real_models = cv.drop(index=[i for i in cv.index if i.startswith("Baseline")])
    best_name = real_models["log_loss_mean"].idxmin()  # best-calibrated real model

    # one honest held-out test on unseen customers, for the report / confusion matrix
    train_df, test_df = holdout_split(df)
    holdout = {}
    for name, model in candidate_models().items():
        est = build(name, model).fit(train_df[FEATURES], train_df[TARGET].to_numpy())
        proba = _proba_full(est, test_df[FEATURES], classes)
        holdout[name] = scores(test_df[TARGET], proba, classes)
        if name == best_name:
            cm = confusion_matrix(test_df[TARGET], np.asarray(classes)[proba.argmax(1)],
                                  labels=classes)

    # final model: refit on ALL data for the app
    final = build(best_name, candidate_models()[best_name]).fit(df[FEATURES], df[TARGET].to_numpy())
    PRODUCT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": best_name, "features": FEATURES, "classes": classes},
                PRODUCT_MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "n_rows": len(df), "n_customers": int(df["customer_id"].nunique()),
        "holdout_rows": len(test_df), "holdout_customers": int(test_df["customer_id"].nunique()),
        "cv": cv.round(4).to_dict(orient="index"),
        "holdout": {k: {m: round(v, 4) for m, v in s.items()} for k, s in holdout.items()},
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }
    PRODUCT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    if verbose:
        pd.set_option("display.width", 140)
        print("Grouped 5-fold CV x 3 seeds (mean):")
        print(cv[["accuracy_mean", "f1_macro_mean", "log_loss_mean"]].round(3))
        print("\nHeld-out customers:")
        print(pd.DataFrame(holdout).T.round(3))
        print(f"\nselected: {best_name}  -> {PRODUCT_MODEL_PATH}")
    return metrics


# ------------------------------------------------------------------ app API
_BUNDLE = None


def load_model():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(PRODUCT_MODEL_PATH)
    return _BUNDLE


def customer_feature_row(customer_id, df=None):
    """Features describing a customer's NEXT purchase, from their latest transaction."""
    df = load_data() if df is None else df
    hist = df[df["customer_id"] == int(customer_id)].sort_values(["purchase_date", "transaction_id"])
    if hist.empty:
        raise KeyError(f"customer {customer_id} is not in {MERGED_CSV.name}")
    last = hist.iloc[[-1]].copy()
    last["prior_purchases"] = len(hist)
    last["prev_category"] = hist[TARGET].iloc[-1]
    last["prev_amount"] = hist["purchase_amount"].iloc[-1]
    last["purchase_amount"] = hist["purchase_amount"].mean()
    last["customer_rating"] = hist["customer_rating"].mean()
    return last[FEATURES]


def predict_for_customer(customer_id):
    """Return (predicted_category, {category: probability}) for a customer_id."""
    bundle = load_model()
    X = customer_feature_row(customer_id)
    proba = _proba_full(bundle["model"], X, bundle["classes"])[0]
    probs = dict(sorted(zip(bundle["classes"], (round(float(p), 3) for p in proba)), key=lambda kv: -kv[1]))
    return next(iter(probs)), probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", type=int, help="predict for this customer_id instead of training")
    args = ap.parse_args()
    if args.customer is None:
        train()
    else:
        cat, probs = predict_for_customer(args.customer)
        print(f"customer {args.customer}: recommend {cat}  {probs}")
