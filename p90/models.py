"""Stacked ensemble: Logistic Regression + XGBoost + LightGBM + Neural Net.

How the stack is trained (no look-ahead):
1. the training matches are sorted by date
2. the 4 base models learn on the older 85%
3. a meta model (logistic regression) learns how much to trust each base
   model on the newest 15% - this also calibrates the probabilities
4. the base models are trained again on 100% of the training matches
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
# scikit-learn must load before LightGBM: LightGBM hides the reason if scikit-learn fails to load
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold  # noqa: F401
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight  # noqa: F401
from sklearn.utils.multiclass import check_classification_targets  # noqa: F401
from sklearn.utils.validation import _check_sample_weight  # noqa: F401

import lightgbm as lgb
import xgboost as xgb

from .config import MODELS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

PARAMS_FILE = MODELS / "best_params.json"

DEFAULT_PARAMS = {
    "xgb": dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0, gamma=0.0),
    "lgbm": dict(n_estimators=600, num_leaves=15, learning_rate=0.03, subsample=0.8,
                 colsample_bytree=0.7, min_child_samples=100, reg_lambda=5.0),
    "lr": dict(C=0.3),
    "mlp": dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3),
}


def load_params() -> dict:
    p = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    if PARAMS_FILE.exists():
        tuned = json.loads(PARAMS_FILE.read_text())
        for k, v in tuned.items():
            if k in p:
                p[k].update(v)
    return p


def make_base(name: str, params: dict, n_classes: int):
    p = params[name]
    if name == "lr":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=p["C"], max_iter=3000))
    if name == "xgb":
        return xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1, verbosity=0)
    if name == "lgbm":
        return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1,
                                  deterministic=True, force_row_wise=True)
    if name == "mlp":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             MLPClassifier(**{**p, "hidden_layer_sizes": tuple(p["hidden_layer_sizes"])},
                                           batch_size=512, max_iter=120, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=8,
                                           random_state=SEED))
    raise ValueError(name)


def _z(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-4, 1 - 1e-4)
    if P.shape[1] == 2:
        return np.log(P[:, 1:] / P[:, :1])
    return np.log(P)


class StackModel:
    def __init__(self, features: list[str], n_classes: int = 3, bases=("lr", "xgb", "lgbm", "mlp"),
                 params: dict | None = None, hold_share: float = 0.15):
        self.features = list(features)
        self.n_classes = n_classes
        self.bases = list(bases)
        self.params = params or load_params()
        self.hold_share = hold_share
        self.models: dict = {}
        self.meta = None
        self.trained_until = None
        self.n_train = 0

    # -------------------------------------------------------------- fit
    def _fit_bases(self, X, y):
        out = {}
        for name in self.bases:
            m = make_base(name, self.params, self.n_classes)
            m.fit(X, y)
            out[name] = m
        return out

    def _base_probs(self, models, X) -> list[np.ndarray]:
        return [models[n].predict_proba(X) for n in self.bases]

    def fit(self, df: pd.DataFrame, target: str, verbose=False):
        d = df.sort_values("date", kind="mergesort")
        X = d[self.features].to_numpy(np.float32)
        y = d[target].astype(int).to_numpy()
        cut = int(len(d) * (1 - self.hold_share))
        first = self._fit_bases(X[:cut], y[:cut])
        Z = np.hstack([_z(p) for p in self._base_probs(first, X[cut:])])
        self.meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y[cut:])
        self.models = self._fit_bases(X, y)
        self.trained_until = d["date"].max()
        self.n_train = len(d)
        return self

    # ---------------------------------------------------------- predict
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(np.float32)
        Z = np.hstack([_z(p) for p in self._base_probs(self.models, X)])
        return self.meta.predict_proba(Z)

    def base_proba(self, df: pd.DataFrame) -> dict:
        X = df[self.features].to_numpy(np.float32)
        return dict(zip(self.bases, self._base_probs(self.models, X)))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "StackModel":
        return joblib.load(path)
