"""Optuna tuning for XGBoost / LightGBM (and the LR strength).

Train on matches before `valid_start`, score log-loss on the next two seasons.
Tuning stops before 2018/19, the first backtest season, so the backtest never sees it.
Results go to models/best_params.json and are used by every later run.
"""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from . import models as M
from .features import FEATURES_1X2

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(feat: pd.DataFrame, valid_start="2016-07-01", valid_end="2018-07-01",
         trials: int = 25, timeout_min: float = 40, verbose=True) -> dict:
    d = feat[feat["y"].notna() & (feat["has_mk"] == 1)]
    tr = d[d["date"] < valid_start]
    va = d[(d["date"] >= valid_start) & (d["date"] < valid_end)]
    Xtr, ytr = tr[FEATURES_1X2].to_numpy(np.float32), tr["y"].astype(int).to_numpy()
    Xva, yva = va[FEATURES_1X2].to_numpy(np.float32), va["y"].astype(int).to_numpy()
    best = {}

    def run(name, space):
        def objective(trial):
            params = M.load_params()
            params[name].update(space(trial))
            m = M.make_base(name, params, 3).fit(Xtr, ytr)
            return log_loss(yva, m.predict_proba(Xva))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        t = time.time()
        study.optimize(objective, n_trials=trials, timeout=timeout_min * 60 / 2)
        best[name] = study.best_params
        if verbose:
            print(f"  {name}: best log-loss {study.best_value:.4f} after {len(study.trials)} trials "
                  f"({time.time() - t:.0f}s)")

    run("xgb", lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 200, 900, step=50),
        max_depth=t.suggest_int("max_depth", 2, 6),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 5, 100, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 0.5, 30, log=True),
        gamma=t.suggest_float("gamma", 0.0, 2.0)))
    run("lgbm", lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 200, 1000, step=50),
        num_leaves=t.suggest_int("num_leaves", 4, 48),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 20, 400, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 0.5, 30, log=True)))

    scores = {}
    for C in (0.03, 0.1, 0.3, 1.0):
        params = M.load_params()
        params["lr"]["C"] = C
        m = M.make_base("lr", params, 3).fit(Xtr, ytr)
        scores[C] = log_loss(yva, m.predict_proba(Xva))
    best["lr"] = {"C": min(scores, key=scores.get)}
    if verbose:
        print(f"  lr: best C={best['lr']['C']} log-loss {min(scores.values()):.4f}")

    M.PARAMS_FILE.write_text(json.dumps(best, indent=2))
    return best
