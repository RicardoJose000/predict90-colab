"""More bets: goal lines, team to score, handicap, corners, cards and shots.

Goals bets are yes/no stacks, like over 2.5 and BTTS. Corners, cards and shots are counts: a model
predicts the expected number, and the spread of the real numbers around it gives the chance of
going over any line. Every number is built only from matches played before kickoff.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .config import MAIN_LEAGUES, SEED
from .features import FEATURES_GOALS
from .models import StackModel

STATS = {"corners": ("hc", "ac"), "cards": ("hk", "ak"), "shots": ("hs", "as_")}
SIDES = ("total", "home", "away")
GOAL_BETS = ["over15", "over35", "h_scores", "a_scores", "h_by2", "a_by2", "fav_or_over"]
LEAGUE_WINDOW = 400
COUNT_PARAMS = dict(objective="poisson", n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED,
                    n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True)
COMMON = ["elo_diff", "elo_exp", "xg_h", "xg_a", "mk_h", "mk_a", "mk_over", "tier", "rest_h", "rest_a"]


def _mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else np.nan


def add_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Each team's last 10 corners, cards and shots, for and against, and the league's averages."""
    d = feat.copy()
    d["hk"], d["ak"] = d["hy"] + d["hr"], d["ay"] + d["ar"]
    hist = defaultdict(lambda: deque(maxlen=10))
    lg = defaultdict(lambda: deque(maxlen=LEAGUE_WINDOW))
    pending, current = [], None
    cols = defaultdict(list)
    get = {c: d[c].tolist() for c in ("date", "league", "country", "home", "away", "res") + tuple(x for p in STATS.values() for x in p)}
    for i in range(len(d)):
        if get["date"][i] != current:
            for key, item in pending:
                lg[key].append(item)
            pending, current = [], get["date"][i]
        kh, ka = (get["country"][i], get["home"][i]), (get["country"][i], get["away"][i])
        for st, (hc, ac) in STATS.items():
            L = lg[(get["league"][i], st)]
            cols[f"{st}_lg_h"].append(_mean([x[0] for x in L]))
            cols[f"{st}_lg_a"].append(_mean([x[1] for x in L]))
            for side, k in (("h", kh), ("a", ka)):
                H = hist[(k, st)]
                cols[f"{st}_f_{side}"].append(_mean([x[0] for x in H]))
                cols[f"{st}_a_{side}"].append(_mean([x[1] for x in H]))
            x, y = get[hc][i], get[ac][i]
            if get["res"][i] in ("H", "D", "A") and x == x and y == y:
                hist[(kh, st)].append((x, y))
                hist[(ka, st)].append((y, x))
                pending.append(((get["league"][i], st), (x, y)))
    F = pd.DataFrame(cols, index=d.index)
    d = pd.concat([d, F], axis=1)
    played = d["y"].notna()
    goals = d["hg"] + d["ag"]
    margin = d["hg"] - d["ag"]
    fav_home = d["mk_h"] >= d["mk_a"]

    def yes(x):
        return np.where(played, x.astype(float), np.nan)

    d["over15"], d["over35"] = yes(goals > 1.5), yes(goals > 3.5)
    d["h_scores"], d["a_scores"] = yes(d["hg"] > 0), yes(d["ag"] > 0)
    d["h_by2"], d["a_by2"] = yes(margin >= 2), yes(margin <= -2)
    fav_won = (margin > 0).where(fav_home, margin < 0)
    d["fav_or_over"] = np.where(played & d["mk_h"].notna(), (fav_won | (goals > 2.5)).astype(float), np.nan)
    for st, (hc, ac) in STATS.items():
        d[f"{st}_home"], d[f"{st}_away"], d[f"{st}_total"] = d[hc], d[ac], d[hc] + d[ac]
    return d


def count_features(st: str) -> list[str]:
    return COMMON + [f"{st}_{x}" for x in ("lg_h", "lg_a", "f_h", "a_h", "f_a", "a_a")]


class CountModel:
    """Expected number + spread. The spread is measured on the newest 15%, never on the matches predicted."""

    def __init__(self, features):
        self.features = features

    def fit(self, d, target):
        d = d[d[target].notna()].sort_values("date", kind="mergesort")
        X, y = d[self.features].to_numpy(np.float32), d[target].to_numpy(float)
        cut = int(len(d) * 0.85)
        mu = lgb.LGBMRegressor(**COUNT_PARAMS).fit(X[:cut], y[:cut]).predict(X[cut:])
        self.alpha = max(float((((y[cut:] - mu) ** 2 - mu) / mu ** 2).mean()), 0.0)
        self.model = lgb.LGBMRegressor(**COUNT_PARAMS).fit(X, y)
        return self

    def predict(self, d):
        return self.model.predict(d[self.features].to_numpy(np.float32))


def p_over(mu, alpha, line):
    """Chance the count ends above a half line like 9.5."""
    k = math.floor(line)
    if alpha < 1e-3:
        return poisson.sf(k, mu)
    return nbinom.sf(k, 1 / alpha, 1 / (1 + alpha * mu))


def fit(train: pd.DataFrame) -> dict:
    d = train[train["y"].notna()]
    out = {"goals": {}, "counts": {}}
    for b in GOAL_BETS:
        out["goals"][b] = StackModel(FEATURES_GOALS, 2, bases=("lr", "lgbm")).fit(d[d[b].notna()], b)
    for st in STATS:
        for side in SIDES:
            out["counts"][f"{st}_{side}"] = CountModel(count_features(st)).fit(d, f"{st}_{side}")
    return out


def predict(models: dict, d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for b, m in models["goals"].items():
        out[f"p_{b}"] = m.predict_proba(d)[:, 1]
    for name, m in models["counts"].items():
        out[f"mu_{name}"] = m.predict(d)
        out[f"alpha_{name}"] = m.alpha
    return out


def corner_1x2(mu_h, a_h, mu_a, a_a) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chance of more corners for home / level / more for away, from the two expected numbers."""
    k = np.arange(41)

    def pmf(mu, a):
        mu = np.asarray(mu, float)[:, None]
        return poisson.pmf(k[None, :], mu) if a < 1e-3 else nbinom.pmf(k[None, :], 1 / a, 1 / (1 + a * mu))

    P = pmf(mu_h, a_h)[:, :, None] * pmf(mu_a, a_a)[:, None, :]
    i, j = np.meshgrid(k, k, indexing="ij")
    return (P * (i > j)).sum((1, 2)), (P * (i == j)).sum((1, 2)), (P * (i < j)).sum((1, 2))


def walk_forward(feat: pd.DataFrame, test_from: int, test_to: int, verbose=True) -> pd.DataFrame:
    """Each season: train only on the seasons before it, then predict that season."""
    keep = ["league", "date", "home", "away", "season", "hg", "ag"] + GOAL_BETS + \
        [f"{st}_{sd}" for st in STATS for sd in SIDES]
    played = feat[feat["y"].notna()]
    out = []
    for y in range(test_from, test_to + 1):
        start, end = pd.Timestamp(y, 7, 1), pd.Timestamp(y + 1, 7, 1)
        test = played[(played["date"] >= start) & (played["date"] < end)]
        if test.empty:
            continue
        p = predict(fit(played[played["date"] < start]), test)
        p = pd.concat([test[keep], p], axis=1)
        p["fold"] = f"{y}/{(y + 1) % 100:02d}"
        out.append(p)
        if verbose:
            print(f"  {p['fold'].iat[0]}: {len(p):,} matches", flush=True)
    return pd.concat(out, ignore_index=True)


def save_backtest(more: pd.DataFrame, path) -> None:
    """Main leagues only (the others have no corners, cards or shots), 4 decimals: a much smaller file."""
    d = more[more["league"].isin(MAIN_LEAGUES)]
    d.round(4).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


TEST_LINES = {"corners_total": (8.5, 9.5, 10.5, 11.5), "corners_home": (3.5, 4.5, 5.5, 6.5), "corners_away": (3.5, 4.5, 5.5),
              "cards_total": (3.5, 4.5, 5.5), "cards_home": (1.5, 2.5), "cards_away": (1.5, 2.5),
              "shots_total": (22.5, 24.5, 26.5), "shots_home": (11.5, 13.5), "shots_away": (9.5, 11.5)}
GOAL_NAMES = {"over15": "Over 1.5", "over35": "Over 3.5", "h_scores": "Home team to score", "a_scores": "Away team to score",
              "h_by2": "Home wins by 2+", "a_by2": "Away wins by 2+", "fav_or_over": "Favourite or over 2.5"}
NICE = {"corners": "Corners", "cards": "Cards", "shots": "Shots"}


def _accuracy_row(name, chance, happened, folds):
    """Bets where the model is 60%+ sure on one side: how often right, against just always picking the likelier side."""
    ok = ~np.isnan(happened)
    q, y, fold = chance[ok], happened[ok], folds[ok]
    sure = (q >= 0.6) | (q <= 0.4)
    right = (q >= 0.5) == (y == 1)
    per = pd.Series(right[sure]).groupby(fold[sure]).mean() if sure.any() else pd.Series(dtype=float)
    per = per[pd.Series(sure).groupby(fold).sum().reindex(per.index) >= 100]
    return {"Bet": name, "Matches": int(ok.sum()), "Happened": y.mean(), "Model said": q.mean(),
            "60%+ picks": int(sure.sum()), "Share": sure.mean(), "Right": right[sure].mean() if sure.any() else np.nan,
            "Model said (60%+)": np.maximum(q, 1 - q)[sure].mean() if sure.any() else np.nan,
            "Always likelier side": max(y.mean(), 1 - y.mean()),
            "Lowest season": per.min() if len(per) else np.nan}


def accuracy_table(bp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    folds = bp["fold"].to_numpy()
    for b in GOAL_BETS:
        rows.append(_accuracy_row(GOAL_NAMES[b], bp[f"p_{b}"].to_numpy(float), bp[b].to_numpy(float), folds))
    for name, lines in TEST_LINES.items():
        st, side = name.split("_")
        label = NICE[st] if side == "total" else f"{side.title()} {st}"
        mu, y = bp[f"mu_{name}"].to_numpy(float), bp[name].to_numpy(float)
        alpha = bp[f"alpha_{name}"].to_numpy(float)
        for line in lines:
            q = np.array([float(p_over(m, a, line)) for m, a in zip(mu, alpha)])
            rows.append(_accuracy_row(f"{label} over {line}", q, np.where(np.isnan(y), np.nan, y > line), folds))
    return pd.DataFrame(rows)


def corner_1x2_table(bp: pd.DataFrame) -> pd.DataFrame:
    d = bp[bp["corners_home"].notna()]
    rows = []
    # the spread is one number per season, so each season is worked out on its own
    for _, g in d.groupby("fold"):
        h, l, a = corner_1x2(g["mu_corners_home"], g["alpha_corners_home"].iat[0], g["mu_corners_away"], g["alpha_corners_away"].iat[0])
        rows.append(pd.DataFrame({"h": h, "l": l, "a": a, "y": np.sign(g["corners_home"] - g["corners_away"]).to_numpy()}))
    t = pd.concat(rows)
    return pd.DataFrame([{"Result": n, "Model said": t[c].mean(), "Happened": (t["y"] == v).mean()}
                         for n, c, v in (("More home corners", "h", 1), ("Level", "l", 0), ("More away corners", "a", -1))])


def _price(more, *path):
    """A real price from the odds of one match, or None."""
    v = more if isinstance(more, dict) else {}
    for k in path:
        if isinstance(v, dict):
            v = v.get(k)
        elif isinstance(v, (tuple, list)):
            v = v[k] if v else None
        else:
            return None
    return v


def _balanced_line(lines: dict):
    """The bookmaker's main line: where over and under are closest to each other."""
    return min(lines, key=lambda l: abs(1 / lines[l][0] - 1 / lines[l][1]))


def _count_tab(common, d, stat, label):
    """Expected number, chance at the bookmaker's main line (or the half line nearest the expected number)."""
    t = common.copy()
    have = d[f"{stat}_f_h"].notna() & d[f"{stat}_f_a"].notna()
    for side, word in (("total", label), ("home", f"Home {label.lower()}"), ("away", f"Away {label.lower()}")):
        rows = []
        for mu, alpha, more in zip(d[f"mu_{stat}_{side}"], d[f"alpha_{stat}_{side}"], d.get("more_odds", pd.Series(None, index=d.index))):
            lines = _price(more, f"{stat}_{side}") or {}
            line = _balanced_line(lines) if lines else math.floor(mu) + 0.5
            over = float(p_over(mu, alpha, line))
            rows.append((mu, line, over, 1 - over) + (tuple(lines[line]) if lines else (None, None)))
        x = pd.DataFrame(rows, index=d.index, columns=[f"{word} expected", f"{word} line", f"{word} over %", f"{word} under %",
                                                       f"{word} over odds", f"{word} under odds"])
        if side != "total":
            x = x[[f"{word} expected", f"{word} line", f"{word} over %"]]
        t = pd.concat([t, x], axis=1)
    return t[have].sort_values(f"{label} expected", ascending=False)


def market_tables(pred: pd.DataFrame, league_label) -> dict[str, pd.DataFrame]:
    """One sheet per kind of bet, one row per match, also where no odds exist. The chance is the model's own."""
    d = pred
    more = d["more_odds"] if "more_odds" in d else pd.Series(None, index=d.index)
    common = pd.DataFrame({
        "Date": pd.to_datetime(d["date_local"]).dt.strftime("%a %d %b"), "Time": d["time_local"],
        "League": d["league"].map(league_label),
        "League type": d["league"].map(lambda c: "Main" if c in MAIN_LEAGUES else "Other"),
        "Home": d["home"], "Away": d["away"]})
    priced = d["mk_h"].notna()
    fav_home = d["mk_h"] >= d["mk_a"]
    num = lambda col: pd.to_numeric(d[col], errors="coerce") if col in d else pd.Series(np.nan, index=d.index)
    prices = lambda *path: [_price(m, *path) for m in more]
    out = {}

    over = d["p_over"]
    out["Over-Under 2.5"] = common.assign(**{
        "Pick": np.where(over >= 0.5, "Over 2.5", "Under 2.5"), "Pick %": np.maximum(over, 1 - over),
        "Over %": over, "Under %": 1 - over, "Bookmaker over %": num("mk_over"),
        "Over odds": num("o_over"), "Under odds": num("o_under")}).sort_values("Pick %", ascending=False)
    btts = d["p_btts"]
    out["BTTS"] = common.assign(**{
        "Pick": np.where(btts >= 0.5, "BTTS yes", "BTTS no"), "Pick %": np.maximum(btts, 1 - btts),
        "Yes %": btts, "No %": 1 - btts, "Yes odds": num("b_btts_yes"), "No odds": num("b_btts_no")}).sort_values("Pick %", ascending=False)
    out["Goal lines"] = common.assign(**{
        "Over 1.5 %": d["p_over15"], "Over 1.5 odds": prices("goals", 1.5, 0),
        "Under 3.5 %": 1 - d["p_over35"], "Under 3.5 odds": prices("goals", 3.5, 1),
        "Favourite": np.where(~priced, "", np.where(fav_home, d["home"], d["away"])),
        "Favourite or over 2.5 %": d["p_fav_or_over"].where(priced)}).sort_values("Over 1.5 %", ascending=False)
    fav_scores = np.where(fav_home, d["p_h_scores"], d["p_a_scores"])
    out["Team to score"] = common.assign(**{
        "Favourite": np.where(~priced, "", np.where(fav_home, d["home"], d["away"])),
        "Favourite to score %": pd.Series(fav_scores, index=d.index).where(priced),
        "Favourite to score odds": [_price(m, "cs_away" if h else "cs_home", 1) for m, h in zip(more, fav_home)],
        "Home to score %": d["p_h_scores"], "Away to score %": d["p_a_scores"]}).sort_values("Favourite to score %", ascending=False)
    dog = common.assign(**{
        "Underdog +1.5": np.where(fav_home, d["away"], d["home"]),
        "Underdog +1.5 %": np.where(fav_home, 1 - d["p_h_by2"], 1 - d["p_a_by2"]),
        "Underdog +1.5 odds": [_price(m, "away_plus15" if h else "home_plus15", 0) for m, h in zip(more, fav_home)]})
    out["Underdog +1.5"] = dog[priced].sort_values("Underdog +1.5 %", ascending=False)

    corners = _count_tab(common, d, "corners", "Corners")
    known = d.loc[corners.index]
    if len(known):
        h, l, a = corner_1x2(known["mu_corners_home"], known["alpha_corners_home"].iat[0], known["mu_corners_away"], known["alpha_corners_away"].iat[0])
        corners["More corners: home %"], corners["Level %"], corners["More corners: away %"] = h, l, a
    out["Corners"] = corners
    out["Cards"] = _count_tab(common, d, "cards", "Cards")
    out["Shots"] = _count_tab(common, d, "shots", "Shots")
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def top_per_league(table: pd.DataFrame, by: str, n: int, keep: list[str]) -> pd.DataFrame:
    """The n highest of each league, best first inside each league."""
    d = table[table[by].notna()].sort_values(["League", by], ascending=[True, False])
    d = d.groupby("League", sort=True).head(n)
    d.insert(3, "Rank", d.groupby("League").cumcount() + 1)
    first = ["Date", "Time", "League", "League type", "Rank", "Home", "Away"]
    return d[first + [c for c in keep if c in d]].reset_index(drop=True)
