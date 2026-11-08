"""Backtest scoring: accuracy, log-loss, calibration, significance, profit, accas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import MAIN_LEAGUES
from .betting import acca_pool, build_my_picks, leg_table, leg_won, pick_name, slips_from_legs, value_bets

EPS = 1e-12


def _arr(df):
    P = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    M = df[["mk_h", "mk_d", "mk_a"]].to_numpy(float)
    y = df["y"].to_numpy(int)
    return P, M, y


def _ll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean())


def _brier(P, y):
    Y = np.eye(3)[y]
    return float(((P - Y) ** 2).sum(1).mean())


def summary(pred: pd.DataFrame, conf=0.50, strong=0.65) -> dict:
    d = pred[pred["has_mk"] == 1]
    P, M, y = _arr(d)
    top = P.max(1)
    hit = P.argmax(1) == y
    mhit = M.argmax(1) == y
    c, s = top >= conf, top >= strong
    return {
        "matches": len(d),
        "accuracy": hit.mean(), "accuracy_bookie": mhit.mean(),
        "logloss": _ll(P, y), "logloss_bookie": _ll(M, y),
        "brier": _brier(P, y), "brier_bookie": _brier(M, y),
        "conf_n": int(c.sum()), "conf_share": c.mean(), "conf_acc": hit[c].mean(),
        "strong_n": int(s.sum()), "strong_share": s.mean(), "strong_acc": hit[s].mean(),
        "bookie_strong_n": int((M.max(1) >= strong).sum()),
        "bookie_strong_acc": mhit[M.max(1) >= strong].mean(),
    }


def by_group(pred: pd.DataFrame, col: str, conf=0.50, strong=0.65) -> pd.DataFrame:
    rows = []
    for k, g in pred[pred["has_mk"] == 1].groupby(col):
        sm = summary(g, conf, strong)
        rows.append({col: k, **sm})
    return pd.DataFrame(rows)


def threshold_curve(pred: pd.DataFrame, steps=np.arange(0.35, 0.86, 0.05)) -> pd.DataFrame:
    P, _, y = _arr(pred[pred["has_mk"] == 1])
    top, hit = P.max(1), P.argmax(1) == y
    rows = []
    for t in steps:
        m = top >= t
        if m.sum() < 20:
            break
        rows.append({"threshold": round(float(t), 2), "share": m.mean(), "matches": int(m.sum()),
                     "accuracy": hit[m].mean()})
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame, bins=np.arange(0.0, 1.01, 0.1)) -> pd.DataFrame:
    """For every predicted outcome: how often did it really happen?"""
    P, _, y = _arr(pred[pred["has_mk"] == 1])
    p = P.ravel()
    o = np.eye(3)[y].ravel()
    idx = np.digitize(p, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 50:
            rows.append({"bin": f"{bins[b]:.0%}-{bins[b + 1]:.0%}", "predicted": p[m].mean(),
                         "actual": o[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def significance(pred: pd.DataFrame, n_boot=2000, seed=42) -> dict:
    """Is the model's log-loss really better than the bookmaker's? Paired bootstrap."""
    d = pred[pred["has_mk"] == 1]
    P, M, y = _arr(d)
    lm = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))
    lb = -np.log(np.clip(M[np.arange(len(y)), y], EPS, 1))
    diff = lb - lm                                     # > 0 means model better
    rng = np.random.default_rng(seed)
    # resample whole match days so that same-day matches stay together
    days = pd.factorize(d["date"])[0]
    day_sum = np.bincount(days, diff)
    day_n = np.bincount(days)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(day_sum), len(day_sum))
        boots[b] = day_sum[pick].sum() / day_n[pick].sum()
    hit_m = P.argmax(1) == y
    hit_b = M.argmax(1) == y
    only_m, only_b = int((hit_m & ~hit_b).sum()), int((~hit_m & hit_b).sum())
    mc = binomtest(only_m, only_m + only_b, 0.5, alternative="greater").pvalue if only_m + only_b else 1.0
    return {
        "logloss_gain": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
        "p_value": float((boots <= 0).mean()),
        "acc_only_model": only_m, "acc_only_bookie": only_b, "acc_p_value": float(mc),
    }


def roi_range(profit, dates, n_boot=2000, seed=42) -> tuple[float, float]:
    """95% range of a return on stake. Whole match days are resampled together."""
    d = pd.factorize(pd.Series(dates))[0]
    p, c = np.bincount(d, np.asarray(profit, float)), np.bincount(d)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        out[b] = p[i].sum() / c[i].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def goals_summary(pred: pd.DataFrame) -> dict:
    d = pred[pred["over25"].notna() & pred["p_over"].notna()]
    o = d["over25"].to_numpy(int)
    po = d["p_over"].to_numpy(float)
    b = d["btts"].to_numpy(int)
    pb = d["p_btts"].to_numpy(float)
    res = {
        "ou_matches": len(d), "ou_acc": ((po >= 0.5) == o).mean(), "ou_base": max(o.mean(), 1 - o.mean()),
        "btts_acc": ((pb >= 0.5) == b).mean(), "btts_base": max(b.mean(), 1 - b.mean()),
        "ou_ll": float(-np.mean(o * np.log(po) + (1 - o) * np.log(1 - po))),
        "btts_ll": float(-np.mean(b * np.log(pb) + (1 - b) * np.log(1 - pb))),
    }
    mk = d["mk_over"].notna().to_numpy()
    if mk.any():
        pm = d["mk_over"].to_numpy(float)[mk]
        res.update({"ou_mk_matches": int(mk.sum()),
                    "ou_acc_on_mk": ((po[mk] >= 0.5) == o[mk]).mean(),
                    "ou_acc_bookie": ((pm >= 0.5) == o[mk]).mean(),
                    "ou_ll_on_mk": float(-np.mean(o[mk] * np.log(po[mk]) + (1 - o[mk]) * np.log(1 - po[mk]))),
                    "ou_ll_bookie": float(-np.mean(o[mk] * np.log(pm) + (1 - o[mk]) * np.log(1 - pm)))})
    for name, p, t in (("ou", po, o), ("btts", pb, b)):
        conf = np.maximum(p, 1 - p) >= 0.65
        res[f"{name}_strong_n"] = int(conf.sum())
        res[f"{name}_strong_acc"] = float(((p[conf] >= 0.5) == t[conf]).mean()) if conf.any() else np.nan
    return res


def value_backtest(pred: pd.DataFrame, s: dict) -> tuple[dict, pd.DataFrame]:
    """Flat 1-unit stakes on every value bet, at average bookmaker odds."""
    d = pred[pred["has_mk"] == 1]
    vb = value_bets(d, s)
    if vb.empty:
        return {"bets": 0}, vb
    vb = vb.join(d[["res", "fold", "league"]])
    vb["won"] = vb["bet"] == vb["res"]
    vb["profit"] = np.where(vb["won"], vb["odds"] - 1, -1.0)
    kelly_unit = vb["stake"] / s["bankroll"]
    vb["kelly_profit"] = kelly_unit * vb["profit"]
    out = {"bets": len(vb), "won": int(vb["won"].sum()), "hit": vb["won"].mean(),
           "avg_odds": vb["odds"].mean(), "profit": vb["profit"].sum(), "roi": vb["profit"].mean(),
           "kelly_growth": float(vb["kelly_profit"].sum())}
    return out, vb


def edge_scan(pred: pd.DataFrame, s: dict, edges=(0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    rows = []
    for e in edges:
        r, _ = value_backtest(pred, {**s, "value_min_edge": e})
        rows.append({"min_edge": e, **{k: r.get(k) for k in ("bets", "hit", "avg_odds", "roi")}})
    return pd.DataFrame(rows)


def pick_roi(pred: pd.DataFrame, threshold: float) -> dict:
    """Flat stake on every pick at or above a confidence level, average odds."""
    d = pred[pred["has_mk"] == 1]
    P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
    O = d[["o_h", "o_d", "o_a"]].to_numpy(float)
    k = P.argmax(1)
    m = P.max(1) >= threshold
    odds = O[np.arange(len(O)), k][m]
    won = (k == d["y"].to_numpy(int))[m]
    prof = np.where(won, odds - 1, -1)
    return {"bets": int(m.sum()), "hit": won.mean(), "avg_odds": float(np.nanmean(odds)), "roi": float(np.nanmean(prof))}


def acca_backtest(pred: pd.DataFrame, s: dict, price: str = "avg") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Every match day: build the slips exactly as the live system does, then settle them.
    price = "avg", "bet365" or "best"."""
    d = acca_pool(pred[pred["has_mk"] == 1], s)
    t = leg_table(d, s, price)
    t["day"] = d["date"].dt.normalize().reindex(t.index).to_numpy()
    hg, ag = d["hg"].to_dict(), d["ag"].to_dict()
    slips, days, legs = [], [], []
    for day, g in t.groupby("day"):
        built = slips_from_legs(g, s)
        wins = 0
        for n, sl in enumerate(built, 1):
            hits = [leg_won(pick, hg[i], ag[i]) for i, pick, _, _ in sl["legs"]]
            won = all(hits)
            wins += won
            legs += [(pick, prob, odds, hit) for (_, pick, prob, odds), hit in zip(sl["legs"], hits)]
            slips.append({"date": day, "slip": n, "legs": len(sl["legs"]), "odds": sl["odds"],
                          "win_prob": sl["win_prob"], "won": won,
                          "profit": sl["odds"] - 1 if won else -1.0})
        if built:
            days.append({"date": day, "slips": len(built), "wins": wins})
    S, D = pd.DataFrame(slips), pd.DataFrame(days)
    if S.empty:
        return {"slips": 0}, S, D
    L = pd.DataFrame(legs, columns=["pick", "prob", "odds", "hit"])
    types = L.groupby("pick").agg(legs=("hit", "size"), said=("prob", "mean"), real=("hit", "mean"),
                                  avg_odds=("odds", "mean")).sort_values("legs", ascending=False).reset_index()
    types["pick"] = types["pick"].map(pick_name)
    full = D[D["slips"] == s["acca_slips"]]
    out = {
        "slips": len(S), "won": int(S["won"].sum()), "hit": S["won"].mean(),
        "avg_odds": S["odds"].mean(), "break_even": 1 / S["odds"].mean(),
        "avg_legs": S["legs"].mean(), "roi": S["profit"].mean(),
        "expected_hit": S["win_prob"].mean(), "types": types,
        "days": len(D), "full_days": len(full),
        "days_1plus": (full["wins"] >= 1).mean() if len(full) else np.nan,
        "days_2plus": (full["wins"] >= 2).mean() if len(full) else np.nan,
        "days_5": (full["wins"] >= 5).mean() if len(full) else np.nan,
    }
    return out, S, D


def combination_test(pred: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Every combination of the bet types with old odds and of leg strength, at three prices.
    Older and newer seasons apart: a real winner should hold up in both."""
    import itertools
    rows = []
    for n in (1, 2, 3):
        for combo in itertools.combinations(("win", "double_chance", "over_under"), n):
            for floor in (0.50, 0.55, 0.60, 0.65, 0.70):
                s2 = {**s, "acca_markets": list(combo), "acca_leg_min_prob": floor}
                a, sl, dd = acca_backtest(pred, s2)
                if a.get("slips", 0) < 100:
                    continue
                b, _, _ = acca_backtest(pred, s2, "bet365")
                c, _, _ = acca_backtest(pred, s2, "best")
                lo, hi = roi_range(sl["profit"], sl["date"])
                y = sl["date"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
                per = sl.groupby(y)["profit"].mean()
                names = {"win": "Win", "double_chance": "Double chance", "over_under": "Over/under"}
                rows.append({"Bets": " + ".join(names[x] for x in combo), "Legs from": f"{floor:.0%}", "slips": a["slips"],
                             "Slips a day": a["slips"] / a["days"], "hit": a["hit"], "roi": a["roi"],
                             "95% range": f"{lo:+.1%} to {hi:+.1%}", "Bet365": b["roi"], "Best odds": c["roi"],
                             "2018-21": sl.loc[y <= 2021, "profit"].mean(), "2022 on": sl.loc[y >= 2022, "profit"].mean(),
                             "Seasons with profit": f"{int((per > 0).sum())} of {len(per)}"})
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)


def strength_table(pred: pd.DataFrame, s: dict) -> pd.DataFrame:
    """The strength groups: how often right, at what price, in all, main and other leagues."""
    d = pred[pred["has_mk"] == 1]
    P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
    k, top = P.argmax(1), P.max(1)
    odds = d[["o_h", "o_d", "o_a"]].to_numpy(float)[np.arange(len(d)), k]
    right = k == d["y"].to_numpy(int)
    main = d["league"].isin(MAIN_LEAGUES).to_numpy()
    fold = d["fold"].to_numpy()
    rows = []
    for scope, mask in (("All 38 leagues", np.ones(len(d), bool)), ("Main 22", main), ("Other 16", ~main)):
        if not mask.any():
            continue
        cuts = ((f"Banger {s['banger_threshold']:.0%}+", s["banger_threshold"], 2),
                ("Strong", s["strong_threshold"], s["banger_threshold"]),
                ("Good", s["confident_threshold"], s["strong_threshold"]), ("Weak", 0, s["confident_threshold"]))
        for name, lo, hi in cuts:
            m = mask & (top >= lo) & (top < hi)
            if not m.any():
                continue
            per = pd.Series(right[m]).groupby(fold[m]).agg(["mean", "size"])
            rows.append({"Leagues": scope, "Strength": name, "matches": int(m.sum()), "share": m.sum() / mask.sum(),
                         "Model said": top[m].mean(), "Right": right[m].mean(), "Lowest season": per.loc[per["size"] >= 50, "mean"].min(),
                         "avg_odds": odds[m].mean(), "roi": (right[m] * odds[m] - 1).mean()})
    return pd.DataFrame(rows)


def favourite_acca_test(pred: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Favourites-only accas: stronger legs against total odds."""
    rows = []
    for floor in (0.65, 0.70, 0.75):
        for lo, hi in ((4.5, 7.0), (3.0, 4.5), (2.0, 3.0)):
            s2 = {**s, "acca_markets": ["win"], "acca_leg_min_prob": floor, "acca_min_odds": lo, "acca_max_odds": hi}
            a, sl, _ = acca_backtest(pred, s2)
            if a.get("slips", 0) < 30:
                continue
            b, _, _ = acca_backtest(pred, s2, "bet365")
            c, _, _ = acca_backtest(pred, s2, "best")
            l, h = roi_range(sl["profit"], sl["date"])
            y = sl["date"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
            per = sl.groupby(y)["profit"].mean()
            rows.append({"Legs from": f"{floor:.0%}", "Total odds": f"{lo:g}-{hi:g}", "slips": a["slips"], "Legs": a["avg_legs"], "avg_odds": a["avg_odds"],
                         "hit": a["hit"], "break_even": a["break_even"], "roi": a["roi"], "95% range": f"{l:+.1%} to {h:+.1%}", "Bet365": b["roi"], "Best odds": c["roi"],
                         "2018-21": sl.loc[y <= 2021, "profit"].mean(), "2022 on": sl.loc[y >= 2022, "profit"].mean(),
                         "Seasons with profit": f"{int((per > 0).sum())} of {len(per)}"})
    return pd.DataFrame(rows)


def my_picks_test(pred: pd.DataFrame, s: dict) -> dict:
    """The My picks rule on the past, every day, built exactly like the live tab."""
    d = acca_pool(pred[pred["has_mk"] == 1], s)
    hg, ag = d["hg"].to_dict(), d["ag"].to_dict()
    out = {}
    for price in ("avg", "bet365"):
        rows = []
        for day, g in d.groupby(d["date"].dt.normalize()):
            for sl in build_my_picks(g, s, price):
                won = all(leg_won(k, hg[i], ag[i]) for i, k, _, _ in sl["legs"])
                rows.append({"date": day, "set": sl["set"], "odds": sl["odds"], "legs": len(sl["legs"]), "won": won,
                             "profit": sl["odds"] - 1 if won else -1.0})
        out[price] = pd.DataFrame(rows)
    a, b = out["avg"], out["bet365"]
    table = []
    for name, x, y in [(n, a[a["set"] == n], b[b["set"] == n]) for n in ("Long acca: banger", "Long acca: strong", "Safer acca")] + [("All my picks", a, b)]:
        lo, hi = roi_range(x["profit"], x["date"])
        table.append({"Set": name, "slips": len(x), "Legs": x["legs"].mean(), "avg_odds": x["odds"].mean(), "hit": x["won"].mean(),
                      "break_even": 1 / x["odds"].mean(), "roi": x["profit"].mean(), "95% range": f"{lo:+.1%} to {hi:+.1%}", "Bet365": y["profit"].mean()})

    def worst_run(x):
        x = x.sort_values("date")
        return int((~x["won"]).astype(int).groupby(x["won"].cumsum()).sum().max())

    return {"table": pd.DataFrame(table), "days": a["date"].nunique(), "all_days": d["date"].dt.normalize().nunique(),
            "run_long": worst_run(a[a["set"] != "Safer acca"]), "run_safer": worst_run(a[a["set"] == "Safer acca"])}
