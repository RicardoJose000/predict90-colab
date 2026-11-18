"""Underdog handicap: the underdog +1.5 up to +5.5 against a big favourite.

Underdog +3.5 wins unless the favourite wins by 4 or more goals (the same bet as +4 on the 3-way
handicap, like Sportybet's 4:0). The chance of every winning margin comes from a score grid
(two Poisson numbers) fitted to the model's own home / away / over 2.5 chances, so it needs no
new model. For every match with a big favourite (60%+) the table shows the smallest line that
is still 80%+ (the best price at that safety) and the chance of every line.

Test, 9 seasons of clubs (each predicted only with the seasons before it), favourite 60%+, line 80%+:
20,555 picks, about 8 a day, right 88.0% (the model said 86.1%), lowest season 85.5%.
National teams, competitive matches: 2,952 picks, right 86.8% (said 86.0%), lowest year 82.1%.
Friendlies are left out (85%, squads change too much). The grid says big wins a bit more often
than they happen, so the lines are, if anything, safer than shown. No old handicap odds: profit
is not tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import poisson

from .betting import EST_MARGIN

LINES = (1.5, 2.5, 3.5, 4.5, 5.5)
SKIP = {"INT_FRIENDLY", "INT_OTHER"}
TAB = "Underdog handicap"
_GRID = None


def _grid():
    """Every pair of expected goals 0.05 to 5: its home / away / over 2.5 chances, and the chance the home
    (away) team wins by 2, 3, 4, 5 or more."""
    global _GRID
    if _GRID is None:
        k = np.arange(13)
        lam = np.arange(0.05, 5.0001, 0.025)
        lh, la = (x.ravel() for x in np.meshgrid(lam, lam, indexing="ij"))
        J = poisson.pmf(k[None, :, None], lh[:, None, None]) * poisson.pmf(k[None, None, :], la[:, None, None])
        i, j = np.meshgrid(k, k, indexing="ij")
        m = i - j
        keys = np.column_stack([(J * (m > 0)).sum((1, 2)), (J * (m < 0)).sum((1, 2)), (J * (i + j > 2)).sum((1, 2))])
        by = {L: ((J * (m > L)).sum((1, 2)), (J * (m < -L)).sum((1, 2))) for L in LINES}
        _GRID = (cKDTree(keys), by)
    return _GRID


def chances(p_h, p_a, p_over) -> tuple[np.ndarray, np.ndarray, dict]:
    """(favourite is home, favourite's win chance, {line: chance the underdog + line wins}). NaN where unknown."""
    p_h, p_a, p_over = (np.asarray(x, float) for x in (p_h, p_a, p_over))
    ok = ~(np.isnan(p_h) | np.isnan(p_a) | np.isnan(p_over))
    fav_home = p_h >= p_a
    favp = np.where(ok, np.maximum(p_h, p_a), np.nan)
    tree, by = _grid()
    idx = np.zeros(len(p_h), int)
    if ok.any():
        idx[ok] = tree.query(np.column_stack([p_h[ok], p_a[ok], p_over[ok]]))[1]
    dog = {L: np.where(ok, 1 - np.where(fav_home, by[L][0][idx], by[L][1][idx]), np.nan) for L in LINES}
    return fav_home, favp, dog


def _dog_price(more, fav_home: bool, line: float) -> float:
    """Bet365 Asian handicap price of the underdog + line. API-Football writes the line from the home team's
    side for both teams: with a home favourite "Away -2.5" is the away underdog +2.5 (checked on real odds:
    home 1.25, "Away -2.5" 1.45, "Away -3.5" 1.17); with an away favourite "Home +2.5" is the home underdog +2.5."""
    ah = more.get("ah") if isinstance(more, dict) else None
    if not isinstance(ah, dict):
        return np.nan
    v = ah.get(f"Away -{line:g}" if fav_home else f"Home +{line:g}")
    try:
        return float(v) if v and float(v) > 1 else np.nan
    except (TypeError, ValueError):
        return np.nan


def table(pred: pd.DataFrame, s: dict, league_label, league_type) -> pd.DataFrame:
    """One row per match with a big favourite: the underdog's safest line and the chance of every line."""
    fav_min = float(s.get("underdog_fav_min_prob", 0.60))
    safe = float(s.get("underdog_min_prob", 0.80))
    cols = ["Date", "Time", "League", "League type", "Home", "Away", "Pick", "Pick %", "Odds", "Price", "Favourite %",
            *[f"+{L:g} %" for L in LINES]]
    if pred.empty or "p_h" not in pred:
        return pd.DataFrame(columns=cols)
    p_over = pred["p_over"] if "p_over" in pred else pd.Series(np.nan, index=pred.index)
    fav_home, favp, dog = chances(pred["p_h"], pred["p_a"], p_over)
    line = np.full(len(pred), np.nan)
    for L in LINES:                                     # the smallest line that is still safe = the best price
        line = np.where(np.isnan(line) & (dog[L] >= safe), L, line)
    use = (favp >= fav_min) & ~np.isnan(line) & ~pred["league"].isin(SKIP).to_numpy()
    if not use.any():
        return pd.DataFrame(columns=cols)
    d = pred[use]
    fh, ln = fav_home[use], line[use]
    prob = np.array([dog[L][use][n] for n, L in enumerate(ln)])
    more = d["more_odds"] if "more_odds" in d else pd.Series(None, index=d.index)
    price = np.array([_dog_price(m, h, L) for m, h, L in zip(more, fh, ln)], float)
    est = np.isnan(price)
    underdog = np.where(fh, d["away"], d["home"])
    t = pd.DataFrame({
        "Date": pd.to_datetime(d["date_local"]).dt.strftime("%a %d %b").to_numpy(), "Time": d["time_local"].to_numpy(),
        "League": d["league"].map(league_label).to_numpy(), "League type": league_type(d["league"]).to_numpy(),
        "Home": d["home"].to_numpy(), "Away": d["away"].to_numpy(),
        "Pick": [f"{u} +{L:g}" for u, L in zip(underdog, ln)], "Pick %": prob,
        "Odds": np.where(est, 1 / (prob * EST_MARGIN), price), "Price": np.where(est, "estimate", ""),
        "Favourite %": favp[use], **{f"+{L:g} %": dog[L][use] for L in LINES}})
    return t.sort_values(["Favourite %", "Date", "Time"], ascending=[False, True, True]).reset_index(drop=True)[cols]
