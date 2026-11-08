"""Pre-match features. Every number is built only from matches played BEFORE kickoff.

Played and upcoming matches go through the same loop, in date order: a match
first reads the current state (features), then - only if it has a result -
updates the state. Upcoming matches therefore get exactly the same kind of
features the model was trained on.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import poisson

K = 20.0            # Elo update speed
HFA = 60.0          # Elo home advantage
ALPHA = 0.12        # weight of the newest match in attack / defence strength
LEAGUE_WINDOW = 400  # matches used for league trends
LABEL = {"H": 0, "D": 1, "A": 2}


def tier_base(tier: int) -> float:
    return 1500.0 - 70.0 * (int(tier) - 1)


def _mean(vals):
    v = [x for x in vals if x == x]          # drop NaN
    return sum(v) / len(v) if v else np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "league", "home"], kind="mergesort").reset_index(drop=True)

    elo: dict = {}
    tier_of: dict = {}
    season_of: dict = {}
    hist = defaultdict(lambda: deque(maxlen=10))     # (pts, gf, ga, sot_f, sot_a, sh_f, sh_a)
    home_pts = defaultdict(lambda: deque(maxlen=5))
    away_pts = defaultdict(lambda: deque(maxlen=5))
    att: dict = {}
    dfn: dict = {}
    last_date: dict = {}
    n_season = defaultdict(int)
    lg = defaultdict(lambda: deque(maxlen=LEAGUE_WINDOW))   # (res, hg, ag)
    meetings = defaultdict(lambda: deque(maxlen=6))   # last meetings of each pair of clubs
    pending = []        # league results of the current day: only counted from the next day on
    current_day = None

    cols = defaultdict(list)
    get = df.to_dict("list")
    n = len(df)

    for i in range(n):
        country, league, tier, season = get["country"][i], get["league"][i], get["tier"][i], get["season"][i]
        date = get["date"][i]
        kh, ka = (country, get["home"][i]), (country, get["away"][i])
        if date != current_day:
            for lg_code, item in pending:
                lg[lg_code].append(item)
            pending, current_day = [], date

        # league trends
        L = lg[league]
        if len(L) >= 30:
            lg_hw = sum(1 for r in L if r[0] == 0) / len(L)
            lg_dr = sum(1 for r in L if r[0] == 1) / len(L)
            lg_hg = sum(r[1] for r in L) / len(L)
            lg_ag = sum(r[2] for r in L) / len(L)
        else:
            lg_hw, lg_dr, lg_hg, lg_ag = 0.45, 0.26, 1.50, 1.15
        lg_g = (lg_hg + lg_ag) / 2

        # team state: new team, tier change, new season
        for k in (kh, ka):
            if k not in elo:
                elo[k] = tier_base(tier) - 40.0
                att[k], dfn[k] = lg_g, lg_g
            elif tier_of[k] != tier:
                old = tier_of[k]
                rel = 0.3 * (elo[k] - tier_base(old))
                elo[k] = tier_base(tier) + (-50.0 if tier < old else 40.0) + rel
                att[k], dfn[k] = 0.5 * att[k] + 0.5 * lg_g, 0.5 * dfn[k] + 0.5 * lg_g
                n_season[k] = 0
            elif season_of[k] != season and season:
                elo[k] = 0.8 * elo[k] + 0.2 * tier_base(tier)
                att[k], dfn[k] = 0.8 * att[k] + 0.2 * lg_g, 0.8 * dfn[k] + 0.2 * lg_g
                n_season[k] = 0
            tier_of[k], season_of[k] = tier, season

        eh, ea = elo[kh], elo[ka]
        cols["elo_h"].append(eh)
        cols["elo_a"].append(ea)
        cols["elo_diff"].append(eh + HFA - ea)
        for side, k in (("h", kh), ("a", ka)):
            H = hist[k]
            last5 = list(H)[-5:]
            cols[f"pts5_{side}"].append(_mean([x[0] for x in last5]))
            cols[f"pts10_{side}"].append(_mean([x[0] for x in H]))
            cols[f"gf10_{side}"].append(_mean([x[1] for x in H]))
            cols[f"ga10_{side}"].append(_mean([x[2] for x in H]))
            cols[f"gd5_{side}"].append(_mean([x[1] - x[2] for x in last5]))
            cols[f"sot10_{side}"].append(_mean([x[3] for x in H]))
            cols[f"sota10_{side}"].append(_mean([x[4] for x in H]))
            cols[f"sh10_{side}"].append(_mean([x[5] for x in H]))
            cols[f"att_{side}"].append(att[k])
            cols[f"def_{side}"].append(dfn[k])
            cols[f"rest_{side}"].append(min((date - last_date[k]).days, 21) if k in last_date else 14)
            cols[f"played_{side}"].append(min(n_season[k], 10))
        # head to head: last 6 meetings, from the home team's point of view
        M = meetings[(country, frozenset((kh[1], ka[1])))]
        h2h = [(g1 - g2) if host == kh[1] else (g2 - g1) for host, g1, g2 in M]
        cols["h2h_n"].append(float(len(h2h)))
        cols["h2h_gd"].append(_mean(h2h))
        cols["h2h_pts"].append(_mean([3.0 if x > 0 else 1.0 if x == 0 else 0.0 for x in h2h]))
        cols["vpts5_h"].append(_mean(list(home_pts[kh])))
        cols["vpts5_a"].append(_mean(list(away_pts[ka])))
        cols["lg_hw"].append(lg_hw)
        cols["lg_dr"].append(lg_dr)
        cols["lg_hg"].append(lg_hg)
        cols["lg_ag"].append(lg_ag)
        # expected goals from attack / defence strength
        cols["xg_h"].append(lg_hg * (att[kh] / lg_g) * (dfn[ka] / lg_g))
        cols["xg_a"].append(lg_ag * (att[ka] / lg_g) * (dfn[kh] / lg_g))

        # ---- update after the match (only if it has been played)
        res = get["res"][i]
        if res in LABEL:
            gh, ga = int(get["hg"][i]), int(get["ag"][i])
            r = LABEL[res]
            exp_h = 1.0 / (1.0 + 10 ** (-(eh + HFA - ea) / 400.0))
            score = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
            mult = np.log(abs(gh - ga) + 1) + 1 if gh != ga else 1.0
            d = K * mult * (score - exp_h)
            elo[kh], elo[ka] = eh + d, ea - d
            ph, pa = (3, 0) if r == 0 else (1, 1) if r == 1 else (0, 3)
            hs, as_, hst, ast = get["hs"][i], get["as_"][i], get["hst"][i], get["ast"][i]
            hist[kh].append((ph, gh, ga, hst, ast, hs, as_))
            hist[ka].append((pa, ga, gh, ast, hst, as_, hs))
            home_pts[kh].append(ph)
            away_pts[ka].append(pa)
            att[kh] = (1 - ALPHA) * att[kh] + ALPHA * gh
            dfn[kh] = (1 - ALPHA) * dfn[kh] + ALPHA * ga
            att[ka] = (1 - ALPHA) * att[ka] + ALPHA * ga
            dfn[ka] = (1 - ALPHA) * dfn[ka] + ALPHA * gh
            last_date[kh] = last_date[ka] = date
            n_season[kh] += 1
            n_season[ka] += 1
            pending.append((league, (r, gh, ga)))
            M.append((kh[1], gh, ga))

    F = pd.DataFrame(cols, index=df.index)
    out = pd.concat([df, F], axis=1)
    out["elo_exp"] = 1.0 / (1.0 + 10 ** (-out["elo_diff"] / 400.0))
    for k in ("pts5", "pts10", "gd5", "sot10", "sota10", "sh10", "att", "def", "rest", "played"):
        out[f"{k}_diff"] = out[f"{k}_h"] - out[f"{k}_a"]
    out["gd10_diff"] = (out["gf10_h"] - out["ga10_h"]) - (out["gf10_a"] - out["ga10_a"])
    out["vpts_diff"] = out["vpts5_h"] - out["vpts5_a"]
    out["tier"] = out["tier"].astype(float)

    _poisson(out)
    _market(out)

    out["y"] = out["res"].map(LABEL)
    played = out["y"].notna()
    out["over25"] = np.where(played, (out["hg"] + out["ag"] > 2.5).astype(float), np.nan)
    out["btts"] = np.where(played, ((out["hg"] > 0) & (out["ag"] > 0)).astype(float), np.nan)
    return out


def _poisson(out: pd.DataFrame) -> None:
    g = np.arange(11)
    xh = np.clip(out["xg_h"].to_numpy(float), 0.1, 6)
    xa = np.clip(out["xg_a"].to_numpy(float), 0.1, 6)
    ph = poisson.pmf(g[None, :], xh[:, None])
    pa = poisson.pmf(g[None, :], xa[:, None])
    m = ph[:, :, None] * pa[:, None, :]                      # n x home goals x away goals
    i, j = np.meshgrid(g, g, indexing="ij")
    out["pois_h"] = (m * (i > j)).sum((1, 2))
    out["pois_d"] = (m * (i == j)).sum((1, 2))
    out["pois_a"] = (m * (i < j)).sum((1, 2))
    out["pois_o25"] = (m * (i + j > 2)).sum((1, 2))
    out["pois_btts"] = (1 - ph[:, 0]) * (1 - pa[:, 0])


def _market(out: pd.DataFrame) -> None:
    o = out[["o_h", "o_d", "o_a"]].to_numpy(float)
    inv = 1.0 / o
    s = inv.sum(1)
    ok = np.isfinite(s)
    imp = inv / s[:, None]
    out["mk_h"], out["mk_d"], out["mk_a"] = imp[:, 0], imp[:, 1], imp[:, 2]
    out["overround"] = np.where(ok, s, np.nan)
    out["has_mk"] = ok.astype(float)
    ou = out[["o_over", "o_under"]].to_numpy(float)
    inv2 = 1.0 / ou
    out["mk_over"] = inv2[:, 0] / inv2.sum(1)


BASE_FEATURES = [
    "elo_h", "elo_a", "elo_diff", "elo_exp",
    "pts5_h", "pts5_a", "pts10_h", "pts10_a", "gf10_h", "gf10_a", "ga10_h", "ga10_a",
    "gd5_h", "gd5_a", "sot10_h", "sot10_a", "sota10_h", "sota10_a",
    "att_h", "att_a", "def_h", "def_a", "rest_h", "rest_a", "played_h", "played_a",
    "vpts5_h", "vpts5_a", "lg_hw", "lg_dr", "lg_hg", "lg_ag", "tier",
    "xg_h", "xg_a", "pois_h", "pois_d", "pois_a", "pois_o25", "pois_btts",
    "pts5_diff", "pts10_diff", "gd5_diff", "gd10_diff", "sot10_diff", "sota10_diff",
    "att_diff", "def_diff", "rest_diff", "vpts_diff", "h2h_n", "h2h_gd", "h2h_pts",
]
MARKET_1X2 = ["mk_h", "mk_d", "mk_a", "overround"]
FEATURES_1X2 = BASE_FEATURES + MARKET_1X2
FEATURES_GOALS = BASE_FEATURES + ["mk_h", "mk_a", "mk_over"]
