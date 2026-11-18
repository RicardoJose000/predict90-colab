"""Bangers by market: the safest pick of each kind for every match, and one short acca per market.

Rules (all in settings.json):
- a pick needs bangers_min_prob (75%) or more, or its own level in bangers_min_prob_by_market
  (Over 2.5 and Over 8.5 corners: 70%, tested below)
- 1X2: the likeliest result. Over 2.5: half our chance, half the bookmaker's when there are odds
  (in the test that was clearly better than the model alone)
- a match in Over 2.5 is not shown again in Over 1.5
- corners and cards: one fixed line (bangers_lines: over 8.5 corners, over 3.5 cards), the lines the
  bookmakers really sell. Shots (and corners/cards without a fixed line): the highest line that is
  still 75%+, never below bangers_min_line, so every match shows once, at its best line
- the page, My picks and the Mixed acca use the main markets (1X2, Over 2.5, corners, cards).
  Over 1.5 has no acca: its odds are too short
- BTTS is not here: in 100,000 test matches the model never reached 75% for it (highest 71%)
- national teams: competitive matches only, 1X2 and Over 1.5 (their Over 2.5 chances were too high in the test)
- cups: domestic cups 1X2, Over 2.5 and Over 1.5; European cups Over 1.5 only (their 75%+ 1X2 picks were right
  only 71% in the test, and their Over 2.5 bangers 65%)
- each acca uses one market only, so the same match is never twice in a slip

The same code runs on the backtest (accuracy_table, acca_table), so the numbers on the page are the
numbers of exactly this rule. Part 5c of the notebook shows them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .betting import EST_MARGIN, slips_from_legs
from .config import CUP_LEAGUES, INTL_LEAGUES

STATS = {"corners": ("C", "Corners"), "cards": ("K", "Cards"), "shots": ("S", "Shots")}
MIN_LINE = {"corners": 6.5, "cards": 2.5, "shots": 15.5}     # lower lines are hardly offered by bookmakers
LINES = {"corners": 8.5, "cards": 3.5}                       # fixed lines (bangers_lines in settings.json)
MIN_PROB = {"over25": 0.70, "corners": 0.70}                 # own levels (bangers_min_prob_by_market)
SAFER = {"corners": 7.5}                                     # also shown: the chance at the line most bookies sell
MAIN = ("win", "over25", "corners", "cards")                 # first on the page, in My picks and the Mixed acca
NO_ACCA = {"over15"}                                         # odds too short for an acca


def lines(s: dict) -> dict:
    """The fixed line of corners and cards from the settings. 0 or null = the highest 75%+ line."""
    return {k: float(v) for k, v in {**LINES, **(s.get("bangers_lines") or {})}.items() if v}


def markets(s: dict | None = None) -> dict[str, str]:
    """{tab name: market}, the main markets first: 1X2, Over 2.5 goals, Over 8.5 corners, Over 3.5 cards,
    then Over 1.5 goals and Shots over."""
    ln = lines(s or {})
    name = {"win": "1X2", "over25": "Over 2.5 goals", "over15": "Over 1.5 goals", "shots": "Shots over",
            **{st: f"Over {ln[st]:g} {st}" if st in ln else f"{st.capitalize()} over" for st in ("corners", "cards")}}
    return {name[m]: m for m in (*MAIN, "over15", "shots")}


MARKETS = markets()
SKIP = {"INT_FRIENDLY", "INT_OTHER"}                          # friendlies: squads change too much
INTL_MARKETS = {"win", "over15"}
EUROPEAN_CUPS = {"CUP_UCL", "CUP_UEL", "CUP_UECL"}
CUP_MARKETS = {"win", "over15", "over25"}                    # domestic cups
EUROPE_MARKETS = {"over15"}
ACCA_TAB = "Banger accas"
MIXED_TAB = "Mixed acca"


def _sf(k, mu, alpha):
    """P(count > k) for many matches at once: negative binomial, Poisson where the spread is ~0."""
    mu, alpha = np.asarray(mu, float), np.asarray(alpha, float)
    a = np.maximum(alpha, 1e-6)
    nb = nbinom.sf(k, 1 / a, 1 / (1 + a * mu))
    return np.where(alpha < 1e-3, poisson.sf(k, mu), nb)


def safest_line(mu, alpha, min_prob: float, min_line: float):
    """The highest half line that is still min_prob or more, for every match. (line, chance), NaN if none."""
    mu, alpha = np.asarray(mu, float), np.asarray(alpha, float)
    line, chance = np.full(len(mu), np.nan), np.full(len(mu), np.nan)
    ok = ~np.isnan(mu) & ~np.isnan(alpha)
    if not ok.any():
        return line, chance
    top = np.floor(np.where(ok, mu, 0)) + 0.5
    for step in range(0, 40):                          # from the expected number down to the lowest line
        L = top - step
        todo = ok & np.isnan(line) & (L >= min_line)
        if not todo.any():
            break
        q = _sf(np.floor(L[todo]), mu[todo], alpha[todo])
        hit = q >= min_prob
        idx = np.flatnonzero(todo)[hit]
        line[idx], chance[idx] = L[todo][hit], q[hit]
    return line, chance


def _col(d, c):
    return pd.to_numeric(d[c], errors="coerce").to_numpy(float) if c in d else np.full(len(d), np.nan)


def _more_price(more, *path):
    v = more if isinstance(more, dict) else {}
    for k in path:
        if isinstance(v, dict):
            v = v.get(k)
        elif isinstance(v, (tuple, list)):
            v = v[k] if v else None
        else:
            return None
    return v


def _real_price(more, stat: str, line: float) -> float:
    """Bet365 price of over `line`. Corners: the normal over/under market, else the 3-way corners market,
    which has every line (so the low banger lines like 7.5 get a real price too). A price of 1.00 means the
    bookmaker hardly pays anything: it stays real (1.01), so the leg is too short for an acca, not an estimate."""
    p = _more_price(more, f"{stat}_total", line, 0)
    if not p and stat == "corners":
        p = _more_price(more, "corners_3way", line, 0)
    if not p:
        return np.nan
    return max(float(p), 1.01)


def _guess(prob, guess):
    """Estimated price: worked out from Bet365's nearest line when there is one, else our chance + a normal margin."""
    ours = 1 / (prob * EST_MARGIN)
    return ours if guess is None else np.where(np.isnan(guess), ours, guess)


def _from_market(more, stat: str, line: float, alpha: float) -> float:
    """Price of over `line` worked out from Bet365's own over/under line nearest to it, for when Bet365 has
    no price at our line (cards and shots: Bet365 mostly offers higher lines than our safe banger lines).
    The bookmaker's over/under price says how many it expects; with our spread that gives its chance at
    our line, plus its own margin. Still marked as an estimate, but it is the bookmaker's view, not ours."""
    lines = _more_price(more, f"{stat}_total")
    if not isinstance(lines, dict) or not lines or not alpha == alpha:
        return np.nan
    near = min(lines, key=lambda x: abs(float(x) - line))
    over, under = (lines[near] + (None, None))[:2] if isinstance(lines[near], (tuple, list)) else (None, None)
    if not over or not under or float(over) <= 1 or float(under) <= 1:
        return np.nan
    book = 1 / float(over) + 1 / float(under)
    p_over = (1 / float(over)) / book
    lo, hi = 0.01, 300.0                                   # the expected number that gives the bookmaker's chance
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(_sf(math.floor(float(near)), mid, alpha)) < p_over:
            lo = mid
        else:
            hi = mid
    q = float(_sf(math.floor(line), (lo + hi) / 2, alpha))
    return max(1 / (q * book), 1.01) if q > 0 else np.nan


def legs(d: pd.DataFrame, s: dict, odds=("odds_h", "odds_d", "odds_a"), over_odds="o_over") -> pd.DataFrame:
    """Every banger, one row per match and market: market, pick, prob, odds, est (made-up price), line, expected.

    d is the predictions table (or the backtest in the same columns). The index is kept, so slips point
    back to the matches like every other acca."""
    t = float(s.get("bangers_min_prob", 0.75))
    level = {**MIN_PROB, **(s.get("bangers_min_prob_by_market") or {})}
    min_line = {**MIN_LINE, **(s.get("bangers_min_line") or {})}
    fixed = lines(s)
    lg = d["league"].astype(str)
    intl = lg.isin(INTL_LEAGUES).to_numpy()
    euro = lg.isin(EUROPEAN_CUPS).to_numpy()
    cup = lg.isin(CUP_LEAGUES).to_numpy() & ~euro
    usable = ~lg.isin(SKIP).to_numpy()
    more = d["more_odds"] if "more_odds" in d else pd.Series(None, index=d.index)
    out = []

    def add(market, ok, pick, prob, price, line=np.nan, expected=np.nan, guess=None, safer=np.nan, safer_odds=np.nan):
        ok = ok & usable & ~np.isnan(prob) & (prob >= float(level.get(market, t)))
        if market not in INTL_MARKETS:
            ok &= ~intl
        if market not in CUP_MARKETS:
            ok &= ~cup
        if market not in EUROPE_MARKETS:
            ok &= ~euro
        if not ok.any():
            return
        price = np.where(np.isnan(price) | (price <= 1), np.nan, price)
        est = np.isnan(price)
        out.append(pd.DataFrame({
            "market": market, "pick": np.asarray(pick, dtype=object)[ok] if np.ndim(pick) else pick,
            "prob": prob[ok], "odds": np.where(est, _guess(prob, guess), price)[ok], "est": est[ok],
            "line": np.broadcast_to(line, len(d))[ok], "expected": np.broadcast_to(expected, len(d))[ok],
            "safer": np.broadcast_to(safer, len(d))[ok], "safer_odds": np.broadcast_to(safer_odds, len(d))[ok]},
            index=d.index[ok]))

    P = np.column_stack([_col(d, c) for c in ("p_h", "p_d", "p_a")])
    has = ~np.isnan(P).any(1)
    k = np.where(has, np.nan_to_num(P).argmax(1), 0)
    O = np.column_stack([_col(d, c) for c in odds])
    rows = np.arange(len(d))
    add("win", has, np.array(list("HDA"))[k], np.where(has, P[rows, k], np.nan), O[rows, k])

    over, mk = _col(d, "p_over"), _col(d, "mk_over")
    over = np.where(np.isnan(mk), over, (over + mk) / 2)
    in_over25 = usable & ~intl & ~euro & (over >= float(level.get("over25", t)))
    add("over25", np.ones(len(d), bool), "O", over, _col(d, over_odds))
    o15 = np.array([_more_price(m, "goals", 1.5, 0) or np.nan for m in more], float)
    add("over15", ~in_over25, "O15", _col(d, "p_over15"), o15)

    for stat, (code, _) in STATS.items():
        mu, alpha = _col(d, f"mu_{stat}_total"), _col(d, f"alpha_{stat}_total")
        known = np.ones(len(d), bool)
        for c in (f"{stat}_f_h", f"{stat}_f_a"):       # no history of this number: could never be settled
            if c in d:
                known &= d[c].notna().to_numpy()
        mu = np.where(known, mu, np.nan)
        ok = ~np.isnan(mu) & ~np.isnan(alpha)
        if stat in fixed:                               # one line for every match, the one bookies sell
            line = np.where(ok, fixed[stat], np.nan)
            chance = np.where(ok, _sf(math.floor(fixed[stat]), np.nan_to_num(mu, nan=1.0), np.nan_to_num(alpha, nan=0.1)), np.nan)
        else:
            line, chance = safest_line(mu, alpha, float(level.get(stat, t)), float(min_line[stat]))
        price = np.array([_real_price(m, stat, ln) if ln == ln else np.nan for m, ln in zip(more, line)], float)
        guess = np.array([_from_market(m, stat, ln, a) if ln == ln and p != p else np.nan
                          for m, ln, a, p in zip(more, line, alpha, price)], float)
        picks = np.array([f"{code}:T:O:{ln}" if ln == ln else "" for ln in line], dtype=object)
        safer, safer_odds = np.nan, np.nan
        if stat in SAFER and stat in fixed and SAFER[stat] < fixed[stat]:
            L = SAFER[stat]
            safer = np.where(ok, _sf(math.floor(L), np.nan_to_num(mu, nan=1.0), np.nan_to_num(alpha, nan=0.1)), np.nan)
            safer_odds = np.array([_real_price(m, stat, L) for m in more], float)
        add(stat, ~np.isnan(line), picks, chance, price, line, mu, guess, safer, safer_odds)

    if not out:
        return pd.DataFrame(columns=["market", "pick", "prob", "odds", "est", "line", "expected", "safer", "safer_odds"])
    return pd.concat(out)


def tables(pred: pd.DataFrame, s: dict, league_label, league_type, pick_name) -> dict[str, pd.DataFrame]:
    """One sheet per market, safest first."""
    b = legs(pred, s)
    out = {}
    for tab, market in markets(s).items():
        x = b[b["market"] == market]
        r = pred.loc[x.index]
        t = pd.DataFrame({
            "Date": pd.to_datetime(r["date_local"]).dt.strftime("%a %d %b"), "Time": r["time_local"],
            "League": r["league"].map(league_label), "League type": r["league"].map(league_type),
            "Home": r["home"], "Away": r["away"], "Pick": [pick_name(p) for p in x["pick"]],
            "Pick %": x["prob"], "Odds": x["odds"], "Price": np.where(x["est"], "estimate", "")})
        if market in STATS:
            t.insert(t.columns.get_loc("Pick %"), "Expected", x["expected"])
        if market in SAFER and x["safer"].notna().any():        # corners: also the over 7.5 most bookies sell
            L = SAFER[market]
            t.insert(t.columns.get_loc("Price"), f"Over {L:g} %", x["safer"].to_numpy(float))
            t.insert(t.columns.get_loc("Price"), f"Odds {L:g}", x["safer_odds"].to_numpy(float))
        if market == "win":
            t["Pick"] = [f"{h} win" if p == "H" else f"{a} win" if p == "A" else "Draw"
                         for p, h, a in zip(x["pick"], r["home"], r["away"])]
        out[tab] = t.sort_values(["Pick %", "Date", "Time"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def market_slip(t: pd.DataFrame, s: dict) -> list[dict]:
    """One acca from one market's bangers of one day, total odds bangers_acca_min_odds to _max_odds."""
    t = t[t["odds"] >= float(s.get("acca_leg_min_odds", 1.15))]
    if t.empty:
        return []
    t = t.assign(cost=-np.log(t["prob"]) / np.log(t["odds"])).sort_values(["cost", "prob"], ascending=[True, False])
    return slips_from_legs(t, {**s, "acca_markets": [t["market"].iat[0]], "acca_slips": 1,
                               "acca_min_odds": float(s.get("bangers_acca_min_odds", 2.0)),
                               "acca_max_odds": float(s.get("bangers_acca_max_odds", 3.0)),
                               "acca_max_legs": int(s.get("acca_max_legs", 7))})


def accas(pred: pd.DataFrame, s: dict, used: set | None = None) -> dict[str, dict[str, list]]:
    """{tab name: {day: [slip]}}, one slip per market and day at most.
    used (optional): matches already in another acca. They are skipped, and this function adds its own."""
    b = legs(pred, s)
    if b.empty:
        return {}
    day = pd.to_datetime(pred.loc[b.index, "date_local"]).dt.strftime("%Y-%m-%d")
    out = {}
    for tab, market in markets(s).items():
        if market in NO_ACCA:
            continue
        for d, g in b[b["market"] == market].groupby(day[b["market"] == market]):
            if used is not None:
                g = g[~g.index.isin(used)]
            sl = market_slip(g, s) if len(g) else []
            if sl:
                out.setdefault(tab, {})[d] = [{**x, "set": f"{tab} acca"} for x in sl]
                if used is not None:
                    used.update(i for x in sl for i, *_ in x["legs"])
    return out


def mixed_slip(b: pd.DataFrame, s: dict) -> list[dict]:
    """One acca from one day's bangers of the main kinds (1X2, Over 2.5, corners, cards): the top
    mixed_acca_per_market (2) of each, safest first. A match is used once: if it is top in two markets, it goes where it is safest and the
    other market takes its next banger."""
    n = int(s.get("mixed_acca_per_market", 2))
    b = b[b["market"].isin(MAIN) & (b["odds"] >= float(s.get("acca_leg_min_odds", 1.15)))]
    b = b.sort_values("prob", ascending=False, kind="stable")
    used, count, legs_ = set(), {}, []
    for i, r in zip(b.index, b.itertuples(index=False)):
        if i in used or count.get(r.market, 0) >= n:
            continue
        used.add(i)
        count[r.market] = count.get(r.market, 0) + 1
        legs_.append((i, r.pick, float(r.prob), float(r.odds), bool(r.est), r.market))
    if len(legs_) < 2:
        return []
    order = {m: k for k, m in enumerate(MAIN)}
    legs_.sort(key=lambda x: (order[x[5]], -x[2]))                 # grouped by kind on the slip
    return [{"legs": [x[:4] for x in legs_], "odds": float(np.prod([x[3] for x in legs_])),
             "win_prob": float(np.prod([x[2] for x in legs_])), "est": [x[0] for x in legs_ if x[4]],
             "markets": [x[5] for x in legs_], "set": MIXED_TAB}]


def mixed_accas(pred: pd.DataFrame, s: dict) -> dict[str, list]:
    """{day: [the mixed acca]}"""
    b = legs(pred, s)
    if b.empty:
        return {}
    day = pd.to_datetime(pred.loc[b.index, "date_local"]).dt.strftime("%Y-%m-%d")
    out = {}
    for d, g in b.groupby(day.to_numpy()):
        sl = mixed_slip(g, s)
        if sl:
            out[d] = sl
    return out


# ------------------------------------------------------------------ test
def _backtest_frame(bp: pd.DataFrame, more: pd.DataFrame | None) -> pd.DataFrame:
    """The 1X2/goals backtest and the corners/cards/shots backtest as one table in the live columns."""
    d = bp.rename(columns={"o_h": "odds_h", "o_d": "odds_d", "o_a": "odds_a"}).copy()
    d["over15"] = (d["hg"] + d["ag"] > 1.5).astype(float)
    if more is not None:
        keep = ["league", "date", "home", "away", "p_over15"] + [c for c in more if c.startswith(("mu_", "alpha_"))] + \
            [f"{st}_total" for st in STATS]
        d = d.drop(columns=["p_over15"], errors="ignore").merge(more[keep], on=["league", "date", "home", "away"], how="left")
    d["date_local"] = d["date"]
    return d


def _won(d: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    r = d.loc[b.index]
    tot = (r["hg"] + r["ag"]).to_numpy(float)
    res = r["y"].to_numpy(float)
    won = np.full(len(b), np.nan)
    m = b["market"].to_numpy()
    p = b["pick"].to_numpy()
    w = m == "win"
    won[w] = (np.array([{"H": 0, "D": 1, "A": 2}[x] for x in p[w]]) == res[w]).astype(float)
    won[m == "over25"] = (tot[m == "over25"] > 2.5).astype(float)
    won[m == "over15"] = (tot[m == "over15"] > 1.5).astype(float)
    for stat in STATS:
        x = m == stat
        if x.any() and f"{stat}_total" in r:
            v = r[f"{stat}_total"].to_numpy(float)[x]
            won[x] = np.where(np.isnan(v), np.nan, (v > b["line"].to_numpy(float)[x]).astype(float))
    return won


def accuracy_table(bp: pd.DataFrame, more: pd.DataFrame | None, s: dict, intl_bt: pd.DataFrame | None = None) -> pd.DataFrame:
    """How often each banger list was right in the walk-forward test (clubs), plus national teams."""
    rows = []

    def row(name, b, won, folds, matches):
        ok = ~np.isnan(won)
        per = pd.Series(won[ok]).groupby(np.asarray(folds)[ok]).agg(["mean", "size"])
        per = per[per["size"] >= 30]
        rows.append({"Bangers": name, "Picks": int(ok.sum()), "Share of matches": ok.sum() / max(matches, 1),
                     "Per season": ok.sum() / max(len(per), 1), "Right": won[ok].mean() if ok.any() else np.nan,
                     "Model said": b["prob"].to_numpy()[ok].mean() if ok.any() else np.nan,
                     "Lowest season": per["mean"].min() if len(per) else np.nan})

    # the matches a list could have come from: the ones with that chance worked out
    need = {"win": "p_h", "over25": "p_over", "over15": "p_over15", **{st: f"mu_{st}_total" for st in STATS}}

    def run(frame, prefix):
        b = legs(frame, s)
        won = _won(frame, b)
        folds = frame.loc[b.index, "fold"].to_numpy()
        usable = ~frame["league"].isin(SKIP)
        for tab, market in markets(s).items():
            m = (b["market"] == market).to_numpy()
            if m.any():
                row(prefix + tab, b[m], won[m], folds[m], int((usable & frame[need[market]].notna()).sum()))

    run(_backtest_frame(bp, more), "")
    if intl_bt is not None and len(intl_bt):
        run(intl_bt.assign(date_local=intl_bt["date"]), "National teams: ")
    return pd.DataFrame(rows)


def acca_table(bp: pd.DataFrame, more: pd.DataFrame | None, s: dict) -> pd.DataFrame:
    """The daily acca of each market in the test (clubs): how many, how often all legs won.
    Profit only where old odds exist (1X2 at average odds, Over 2.5): the other prices are estimates."""
    d = _backtest_frame(bp, more)
    b = legs(d, s)
    b["won"] = _won(d, b)
    b["day"] = pd.to_datetime(d.loc[b.index, "date"]).dt.strftime("%Y-%m-%d").to_numpy()
    rows = []
    for tab, market in markets(s).items():
        if market in NO_ACCA:
            continue
        g = b[(b["market"] == market) & b["won"].notna()]
        res = []
        for _, x in g.groupby("day"):
            for sl in market_slip(x, s):
                ids = [i for i, *_ in sl["legs"]]
                res.append((all(x.loc[ids, "won"] == 1), sl["odds"], sl["win_prob"], len(ids), bool(sl.get("est"))))
        if not res:
            continue
        r = pd.DataFrame(res, columns=["won", "odds", "chance", "legs", "est"])
        real = ~r["est"]
        rows.append({"Acca": f"{tab} acca", "Slips": len(r), "Legs (avg)": r["legs"].mean(), "Odds (avg)": r["odds"].mean(),
                     "Model said": r["chance"].mean(), "Won": r["won"].mean(),
                     "Profit per slip": (np.where(r.loc[real, "won"], r.loc[real, "odds"] - 1, -1)).mean() if real.sum() >= 30 else np.nan,
                     "Real odds slips": int(real.sum())})
    return pd.DataFrame(rows)


def mixed_table(bp: pd.DataFrame, more: pd.DataFrame | None, s: dict, per_market=(1, 2)) -> pd.DataFrame:
    """The daily mixed acca in the test (clubs): how many legs, total odds, how often every leg won."""
    d = _backtest_frame(bp, more)
    b = legs(d, s)
    b["won"] = _won(d, b)
    b = b[b["won"].notna()]
    won = {(i, m): w for i, m, w in zip(b.index, b["market"], b["won"])}      # a match has one row per market
    day = pd.to_datetime(d.loc[b.index, "date"]).dt.strftime("%Y-%m-%d").to_numpy()
    rows = []
    for n in per_market:
        res = []
        for _, x in b.groupby(day):
            for sl in mixed_slip(x, {**s, "mixed_acca_per_market": n}):
                w = [won[(leg[0], m)] for leg, m in zip(sl["legs"], sl["markets"])]
                res.append((all(v == 1 for v in w), sl["odds"], sl["win_prob"], len(w), len(sl["est"]) == 0,
                            sum(v == 0 for v in w)))
        if not res:
            continue
        r = pd.DataFrame(res, columns=["won", "odds", "chance", "legs", "real", "lost_legs"])
        rows.append({"Mixed acca": f"top {n} of each kind", "Slips": len(r), "Legs (avg)": r["legs"].mean(),
                     "Odds (avg)": r["odds"].mean(), "Model said": r["chance"].mean(), "Won": r["won"].mean(),
                     "Only 1 leg lost": (r["lost_legs"] == 1).mean(), "Legs right": 1 - r["lost_legs"].sum() / r["legs"].sum()})
    return pd.DataFrame(rows)
