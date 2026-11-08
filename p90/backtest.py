"""Walk-forward backtest: for every season, train only on the past, predict that season.

Season window = 1 July to 30 June. For calendar-year leagues (Brazil, Japan,
USA ...) the window simply cuts through their season - still only past data
is used for training.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import BASE_FEATURES, FEATURES_1X2, FEATURES_GOALS
from .models import StackModel

KEEP = ["league", "country", "tier", "season", "date", "time", "home", "away", "res", "hg", "ag",
        "y", "over25", "btts", "o_h", "o_d", "o_a", "m_h", "m_d", "m_a", "b_h", "b_d", "b_a",
        "o_over", "o_under", "m_over", "m_under", "b_over", "b_under", "mk_h", "mk_d", "mk_a", "mk_over", "has_mk",
        "pois_o25", "pois_btts", "xg_h", "xg_a", "elo_diff"]

GOAL_BASES = ("lr", "lgbm")


def past_warmup(feat: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    """Drop each league's first year: ratings need time to settle."""
    first = feat.groupby("league")["date"].transform("min")
    return feat[feat["date"] >= first + pd.Timedelta(days=days)]


def fit_all(train: pd.DataFrame, goals: bool = True, nomarket: bool = True) -> dict:
    """Train every model the system uses. Returns a dict of StackModels."""
    played = train[train["y"].notna()]
    out = {"1x2": StackModel(FEATURES_1X2, 3).fit(played[played["has_mk"] == 1], "y")}
    if nomarket:
        out["1x2_nomarket"] = StackModel(BASE_FEATURES, 3, bases=("lr", "lgbm")).fit(played, "y")
    if goals:
        out["over25"] = StackModel(FEATURES_GOALS, 2, bases=GOAL_BASES).fit(played, "over25")
        out["btts"] = StackModel(FEATURES_GOALS, 2, bases=GOAL_BASES).fit(played, "btts")
    return out


def predict_all(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in KEEP if c in df.columns]
    out = df[keep].copy()
    P = np.full((len(df), 3), np.nan)
    mk = (df["has_mk"] == 1).to_numpy()
    if mk.any():
        P[mk] = models["1x2"].predict_proba(df[mk])
    if (~mk).any() and "1x2_nomarket" in models:
        P[~mk] = models["1x2_nomarket"].predict_proba(df[~mk])
    out["p_h"], out["p_d"], out["p_a"] = P[:, 0], P[:, 1], P[:, 2]
    out["model"] = np.where(mk, "with odds", "no odds")
    if "over25" in models:
        out["p_over"] = models["over25"].predict_proba(df)[:, 1]
        out["p_btts"] = models["btts"].predict_proba(df)[:, 1]
    return out


def walk_forward(feat: pd.DataFrame, test_from: int, test_to: int, goals=True, verbose=True) -> pd.DataFrame:
    feat = past_warmup(feat)
    played = feat[feat["y"].notna()]
    preds = []
    for y in range(test_from, test_to + 1):
        start, end = pd.Timestamp(y, 7, 1), pd.Timestamp(y + 1, 7, 1)
        train = played[played["date"] < start]
        test = played[(played["date"] >= start) & (played["date"] < end)]
        if test.empty:
            continue
        t = time.time()
        models = fit_all(train, goals=goals, nomarket=(test["has_mk"] == 0).any())
        p = predict_all(models, test)
        p["fold"] = f"{y}/{(y + 1) % 100:02d}"
        preds.append(p)
        if verbose:
            acc = (p[["p_h", "p_d", "p_a"]].to_numpy().argmax(1) == p["y"].to_numpy()).mean()
            print(f"  {p['fold'].iat[0]}: trained on {len(train):,}, tested {len(test):,} matches, "
                  f"accuracy {acc:.1%} ({time.time() - t:.0f}s)")
    return pd.concat(preds, ignore_index=True)
