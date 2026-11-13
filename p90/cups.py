"""Cups: Champions League, Europa League, Conference League and the big domestic cups.

The clubs are the same clubs as in the leagues, so their strength comes from the league data: Elo, recent
points and goals, attack and defence, read from BEFORE kickoff (cup matches never change the league model).
What the cups add:
- domestic cups: clubs from different levels meet. A club below our lowest league (for example a 4th tier
  Spanish club in the Copa del Rey) is "small club": the model knows only that it plays below our leagues.
- European cups: clubs from different countries meet, so each country gets a strength offset, learned from
  the European results (England stronger than Greece, and so on).
- finals (and FA Cup semi-finals) are on neutral ground.
Bets are settled after 90 minutes, so the training uses the 90 minute score (not extra time or penalties).
A European match with a club from a country we have no league for is skipped: no blind guesses.

Data: API-Football (free plan has the 2022/23-2024/25 seasons, shipped in data_snapshot), and the newest
results from the saved daily fixture lists. Model: logistic regression (1X2), Poisson goals.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .config import API_CACHE, LEAGUES, OUTPUT, ROOT
from .features import tier_base

SNAPSHOT = ROOT / "data_snapshot" / "cup_results.csv.gz"
ID_SNAPSHOT = ROOT / "data_snapshot" / "club_ids.csv.gz"
BT_FILE = OUTPUT / "backtest_cups.csv.gz"

# API-Football id: (code, name, country of a domestic cup / None for the European cups)
CUPS = {
    2: ("CUP_UCL", "Champions League", None),
    3: ("CUP_UEL", "Europa League", None),
    848: ("CUP_UECL", "Conference League", None),
    45: ("CUP_FA", "FA Cup", "England"),
    48: ("CUP_EFL", "League Cup (Carabao)", "England"),
    143: ("CUP_CDR", "Copa del Rey", "Spain"),
    81: ("CUP_DFB", "DFB Pokal", "Germany"),
    137: ("CUP_CI", "Coppa Italia", "Italy"),
    66: ("CUP_CDF", "Coupe de France", "France"),
    181: ("CUP_SCO", "Scottish Cup", "Scotland"),
    185: ("CUP_SLC", "Scottish League Cup", "Scotland"),
    90: ("CUP_KNVB", "KNVB Beker", "Netherlands"),
    96: ("CUP_TACA", "Taca de Portugal", "Portugal"),
}
CUP_NAMES = {code: name for code, name, _ in CUPS.values()}
EUROPE = {"CUP_UCL", "CUP_UEL", "CUP_UECL"}
COUNTRY_OF = {code: c for code, _, c in CUPS.values()}
# countries whose clubs can meet in the cups above (the club list of each comes from API-Football once)
COUNTRIES = ["England", "Scotland", "Germany", "Italy", "Spain", "France", "Netherlands", "Belgium", "Portugal",
             "Turkey", "Greece", "Austria", "Denmark", "Finland", "Ireland", "Norway", "Poland", "Romania",
             "Sweden", "Switzerland"]
CLUB_COUNTRY_FIX = {"Monaco": "France"}            # plays in the French league
DONE = ("FT", "AET", "PEN")
OUR_COUNTRIES = {v[0] for v in LEAGUES.values()}
LOWEST_TIER = {c: max(t for cc, _, t, _ in LEAGUES.values() if cc == c) for c in OUR_COUNTRIES}
MAX_IDLE_DAYS = 150                                 # no league match for longer (summer and Nordic winter breaks are shorter): the club left our leagues

FEATURES = ["dr", "neutral", "euro", "tier_h", "tier_a", "tier_gap", "known_h", "known_a",
            "pts5_h", "pts5_a", "gd5_h", "gd5_a", "gf10_h", "ga10_h", "gf10_a", "ga10_a",
            "att_h", "def_h", "att_a", "def_a"]
GOAL_FEATURES = FEATURES + ["lg_g_h", "lg_g_a"]
FILL = {"pts5": 1.35, "pts10": 1.35, "gd5": 0.0, "gf10": 1.3, "ga10": 1.3, "att": 1.3, "def": 1.3, "lg_g": 1.3}


def neutral_round(round_name: str, lid: int) -> int:
    r = str(round_name).lower()
    final = r == "final" or r.endswith(" final") or r.startswith("final")
    return int(final or (lid == 45 and r.startswith("semi")))       # FA Cup semi-finals are at Wembley


def parse_fixtures(response: list, done_only=True) -> list:
    """API-Football fixtures -> rows. hg / ag are the score after 90 minutes (bets are settled on it)."""
    rows = []
    for fx in response:
        lid = fx["league"]["id"]
        if lid not in CUPS:
            continue
        st = fx["fixture"]["status"]["short"]
        ft = (fx.get("score") or {}).get("fulltime") or {}
        if done_only and (st not in DONE or ft.get("home") is None or ft.get("away") is None):
            continue
        rows.append({"fixture_id": fx["fixture"]["id"], "league": CUPS[lid][0], "season": fx["league"].get("season"),
                     "round": fx["league"].get("round", ""), "date": fx["fixture"]["date"][:10],
                     "kickoff": fx["fixture"]["date"], "status": st,
                     "home_id": fx["teams"]["home"]["id"], "away_id": fx["teams"]["away"]["id"],
                     "api_home": fx["teams"]["home"]["name"], "api_away": fx["teams"]["away"]["name"],
                     "hg": ft.get("home"), "ag": ft.get("away"),
                     "neutral": neutral_round(fx["league"].get("round", ""), lid)})
    return rows


# ------------------------------------------------------------------ data
def load_history(today=None) -> pd.DataFrame:
    """Finished cup matches: the shipped seasons plus everything in the saved daily fixture lists."""
    import json
    parts = [pd.read_csv(SNAPSHOT)] if SNAPSHOT.exists() else []
    stop = (today or datetime.now().date()).isoformat()
    new = []
    for f in sorted(API_CACHE.glob("fixtures_date-*.json")):
        if f.stem.split("-", 1)[1] > stop:
            continue
        try:
            new += parse_fixtures(json.loads(f.read_text(encoding="utf-8")).get("response", []))
        except Exception:
            continue
    if new:
        parts.append(pd.DataFrame(new))
    if not parts:
        return pd.DataFrame()
    h = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["fixture_id"], keep="last")
    h["date"] = pd.to_datetime(h["date"])
    h["hg"], h["ag"] = h["hg"].astype(int), h["ag"].astype(int)
    return h.sort_values(["date", "fixture_id"], kind="stable").reset_index(drop=True)


def upcoming_fixtures(api, days) -> pd.DataFrame:
    """Cup matches today/tomorrow from the same saved fixture lists the leagues use (no extra request)."""
    rows = []
    for d in days:
        data = api.get("fixtures", {"date": d.isoformat()}, max_age_hours=2)
        rows += [r for r in parse_fixtures((data or {}).get("response", []), done_only=False) if r["status"] in ("NS", "TBD")]
    if not rows:
        return pd.DataFrame()
    u = pd.DataFrame(rows).drop_duplicates(subset=["fixture_id"])
    u["date"] = pd.to_datetime(u["date"])
    u["season"] = u["season"].fillna(0).astype(int)
    return u


class Names:
    """API-Football club -> (country, our club name).

    By club id only, never by a guess on the name: a cup has many small clubs with names like league clubs
    ("Chichester City", "Coventry Sphinx"). The ids come from the league fixtures (shipped list for 2024/25,
    plus every league match in the saved daily fixture lists, matched by the league's own name list).
    A small club whose name looks like a league club's is flagged, and such a match is skipped."""

    def __init__(self, results: pd.DataFrame):
        import json
        from . import live
        from .config import API_TO_CODE
        from .data import fix_name
        self.live = live
        r = results[results["date"] >= results["date"].max() - pd.Timedelta(days=12 * 365)]
        self.teams = {c: sorted(set(g["home"]) | set(g["away"])) for c, g in r.groupby("country")}
        ids = pd.read_csv(ID_SNAPSHOT) if ID_SNAPSHOT.exists() else pd.DataFrame(columns=["id", "name", "country", "our_name"])
        if "our_name" not in ids:
            ids["our_name"] = None
        self.country_of_id = {int(i): CLUB_COUNTRY_FIX.get(n, c) for i, n, c in zip(ids["id"], ids["name"], ids["country"])}
        self.ours = {int(i): (c, fix_name(c, o)) for i, c, o in zip(ids["id"], ids["country"], ids["our_name"]) if isinstance(o, str)}
        # newer seasons: league matches in the saved fixture lists, with the names the league matcher already confirmed
        aliases = live.load_aliases()
        for f in sorted(API_CACHE.glob("fixtures_date-*.json")):
            try:
                resp = json.loads(f.read_text(encoding="utf-8")).get("response", [])
            except Exception:
                continue
            for fx in resp:
                code = API_TO_CODE.get(fx["league"]["id"])
                if code is None:
                    continue
                country = LEAGUES[code][0]
                for side in ("home", "away"):
                    t = fx["teams"][side]
                    our = aliases.get((code, t["name"]))
                    if our:
                        self.ours[int(t["id"])] = (country, fix_name(country, our))
                        self.country_of_id.setdefault(int(t["id"]), country)
        self.memo: dict = {}

    def country(self, league: str, team_id) -> str | None:
        c = COUNTRY_OF.get(league)
        if c:
            return c
        if pd.isna(team_id):
            return None
        tid = int(team_id)
        return self.ours[tid][0] if tid in self.ours else self.country_of_id.get(tid)

    def match(self, country: str | None, team_id, api_name: str) -> str | None:
        """Our name for this club, None = not in our leagues (a small club, or a country we don't have)."""
        if pd.isna(team_id):
            return None
        hit = self.ours.get(int(team_id))
        return hit[1] if hit and hit[0] == country else None

    def looks_like_league_club(self, country: str | None, api_name: str) -> bool:
        """True when an unknown club's name is close to a club in our leagues: then we can't be sure who it is."""
        if country not in self.teams:
            return False
        key = (country, api_name)
        if key not in self.memo:
            sc = max((self.live._score(api_name, c) for c in self.teams[country]), default=0.0)
            self.memo[key] = sc >= 0.75
        return self.memo[key]


def assemble(rows: pd.DataFrame, names: Names) -> pd.DataFrame:
    """Adds country_h/a, team_h/a (our name, None = not in our leagues) and unsure_h/a."""
    r = rows.copy()
    for side, col in (("h", "home"), ("a", "away")):
        r[f"country_{side}"] = [names.country(lg, i) for lg, i in zip(r["league"], r[f"{col}_id"])]
        r[f"team_{side}"] = [names.match(c, i, n) for c, i, n in zip(r[f"country_{side}"], r[f"{col}_id"], r[f"api_{col}"])]
        r[f"unsure_{side}"] = [(not isinstance(t, str)) and names.looks_like_league_club(c, n)
                               for t, c, n in zip(r[f"team_{side}"], r[f"country_{side}"], r[f"api_{col}"])]
    return r


def probes(rows: pd.DataFrame) -> pd.DataFrame:
    p = rows[["date", "country_h", "team_h", "country_a", "team_a"]].copy()
    return p.fillna({"team_h": "", "team_a": "", "country_h": "", "country_a": ""})


# -------------------------------------------------------------- features
def make_features(rows: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """One row per cup match: both clubs' league state, small clubs filled in, and which rows can be used."""
    d = pd.concat([rows.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
    d["euro"] = d["league"].isin(EUROPE).astype(int)
    for s in ("h", "a"):
        known = d[f"team_{s}"].notna() & d[f"elo_{s}"].notna() & (d[f"idle_{s}"] <= MAX_IDLE_DAYS)
        d[f"known_{s}"] = known.astype(int)
        low = d[f"country_{s}"].map(LOWEST_TIER)
        small_tier = low + 1
        d[f"tier_{s}"] = np.where(known, d[f"tier_{s}"], small_tier)
        d[f"elo_{s}"] = np.where(known, d[f"elo_{s}"], [tier_base(t) - 40 if t == t else np.nan for t in small_tier])
        for c, v in FILL.items():
            d[f"{c}_{s}"] = np.where(known, d[f"{c}_{s}"], v)
            d[f"{c}_{s}"] = d[f"{c}_{s}"].fillna(v)
    d["tier_gap"] = d["tier_a"] - d["tier_h"]
    # usable: European = both clubs known; domestic = at least one known (the other is a small club)
    dom = d["euro"] == 0
    both = (d["known_h"] == 1) & (d["known_a"] == 1)
    one = (d["known_h"] == 1) | (d["known_a"] == 1)
    # a small club whose name looks like a league club's: we can't be sure who it is, so the match is skipped
    unsure = ((d["known_h"] == 0) & d["unsure_h"].astype(bool)) | ((d["known_a"] == 0) & d["unsure_a"].astype(bool))
    d["usable"] = np.where(dom, one & d["country_h"].isin(OUR_COUNTRIES) & ~unsure, both)
    d["y"] = np.select([d["hg"] > d["ag"], d["hg"] == d["ag"]], [0, 1], 2) if "hg" in d else np.nan
    if "hg" in d:
        d.loc[d["hg"].isna(), "y"] = np.nan
    return d


def fit_offsets(train: pd.DataFrame, ridge: float = 8.0) -> dict:
    """Country strength in Elo points, from European matches: goal difference ~ Elo gap + home + countries."""
    e = train[(train["euro"] == 1) & train["usable"]]
    countries = sorted(set(e["country_h"]) | set(e["country_a"]))
    if len(e) < 50 or len(countries) < 2:
        return {}
    idx = {c: i for i, c in enumerate(countries)}
    X = np.zeros((len(e), len(countries) + 2))
    X[:, 0] = (e["elo_h"] - e["elo_a"]).to_numpy() / 100
    X[:, 1] = 1 - e["neutral"].to_numpy()
    for r, (ch, ca) in enumerate(zip(e["country_h"], e["country_a"])):
        X[r, 2 + idx[ch]] += 1
        X[r, 2 + idx[ca]] -= 1
    y = (e["hg"] - e["ag"]).to_numpy(float)
    pen = np.r_[0.0, 0.0, np.full(len(countries), ridge)]           # the countries shrink to "average"
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    per100 = max(beta[0], 0.2)                                         # goals per 100 Elo points
    return {c: float(beta[2 + i] / per100 * 100) for c, i in idx.items()}


def apply_offsets(d: pd.DataFrame, off: dict) -> pd.DataFrame:
    d = d.copy()
    oh = d["country_h"].map(off).fillna(0.0)
    oa = d["country_a"].map(off).fillna(0.0)
    d["dr"] = (d["elo_h"] + oh) - (d["elo_a"] + oa)
    return d


# ---------------------------------------------------------------- model
class Model:
    """1X2: logistic regression with strong regularisation (in the test it beat LR + LightGBM: the cup data
    is small, and the tree model only added noise). Goals: two Poisson regressions."""
    C = 0.05

    def fit(self, train: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression, PoissonRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.off = fit_offsets(train)
        t = apply_offsets(train[train["usable"] & train["y"].notna()], self.off)
        X, y = t[FEATURES].to_numpy(float), t["y"].to_numpy(int)
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000)).fit(X, y)
        G = t[GOAL_FEATURES].to_numpy(float)
        self.gh = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)).fit(G, t["hg"].clip(upper=10))
        self.ga = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)).fit(G, t["ag"].clip(upper=10))
        self.n = len(t)
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        from .intl import goal_markets
        d = apply_offsets(d, self.off)
        X = d[FEATURES].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = self.lr.predict_proba(X)
        G = d[GOAL_FEATURES].to_numpy(float)
        lh, la = self.gh.predict(G), self.ga.predict(G)
        out = pd.DataFrame({"p_h": P[:, 0], "p_d": P[:, 1], "p_a": P[:, 2], "xg_h": lh, "xg_a": la, "dr": d["dr"].to_numpy()},
                           index=d.index)
        return pd.concat([out, goal_markets(lh, la, d.index)], axis=1)


# ------------------------------------------------------------- backtest
def build(results: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Cup rows (finished and/or upcoming) -> features, reading the club state from the league results."""
    from .features import build_features
    names = Names(results)
    r = assemble(rows, names)
    _, state = build_features(results, probes=probes(r))
    return make_features(r, state)


def backtest(results: pd.DataFrame, test_seasons=(2023, 2024), verbose=True, built: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each season predicted by a model (and country offsets) trained only on the seasons before it."""
    d = build(results, load_history()) if built is None else built
    out = []
    for season in test_seasons:
        train, test = d[d["season"] < season], d[(d["season"] == season) & d["usable"]]
        if test.empty:
            continue
        m = Model().fit(train)
        p = pd.concat([test[["date", "season", "league", "round", "api_home", "api_away", "team_h", "team_a",
                             "country_h", "country_a", "known_h", "known_a", "tier_h", "tier_a", "neutral", "euro",
                             "hg", "ag", "y"]], m.predict(test)], axis=1)
        p["fold"] = f"{season}/{str(season + 1)[2:]}"
        out.append(p)
        if verbose:
            acc = (p[["p_h", "p_d", "p_a"]].to_numpy().argmax(1) == p["y"]).mean()
            print(f"  {p['fold'].iloc[0]}: tested {len(p):,} cup matches (trained on {m.n:,}), accuracy {acc:.1%}", flush=True)
    bt = pd.concat(out, ignore_index=True)
    bt.to_csv(BT_FILE, index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.4f")
    return bt


def load_backtest() -> pd.DataFrame | None:
    return pd.read_csv(BT_FILE, parse_dates=["date"]) if BT_FILE.exists() else None


def _groups(d):
    yield "All cups", d
    yield "European cups", d[d["euro"] == 1]
    yield "Domestic cups", d[d["euro"] == 0]
    yield "Domestic, both clubs in our leagues", d[(d["euro"] == 0) & (d["known_h"] == 1) & (d["known_a"] == 1)]
    yield "Domestic, against a small club", d[(d["euro"] == 0) & ((d["known_h"] == 0) | (d["known_a"] == 0))]


def accuracy_table(bt: pd.DataFrame) -> pd.DataFrame:
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    d = bt.assign(top=P.max(1), right=(P.argmax(1) == bt["y"].to_numpy()),
                  ll=-np.log(np.clip(P[np.arange(len(P)), bt["y"].to_numpy(int)], 1e-9, 1)),
                  stronger=np.where(bt["dr"] + np.where(bt["neutral"] == 1, 0, 60) >= 0, 0, 2) == bt["y"].to_numpy())
    rows = []
    for name, g in _groups(d):
        if g.empty:
            continue
        def at(p):
            m = g["top"] >= p
            return (g.loc[m, "right"].mean() if m.sum() >= 20 else np.nan), m.mean(), int(m.sum())
        a50, s50, _ = at(0.5)
        a65, s65, _ = at(0.65)
        a75, s75, n75 = at(0.75)
        rows.append({"Group": name, "Matches": len(g), "Accuracy": g["right"].mean(), "Stronger club wins": g["stronger"].mean(),
                     "Picks 50%+": a50, "Share 50%+": s50, "Picks 65%+": a65, "Share 65%+": s65,
                     "Picks 75%+": a75, "75%+ picks": n75, "Log-loss": g["ll"].mean()})
    return pd.DataFrame(rows)


def season_table(bt: pd.DataFrame) -> pd.DataFrame:
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    d = bt.assign(top=P.max(1), right=(P.argmax(1) == bt["y"].to_numpy()))
    rows = []
    for f, g in d.groupby("fold"):
        rows.append({"Season": f, "Matches": len(g), "Accuracy": g["right"].mean(),
                     "Picks 65%+": g.loc[g["top"] >= .65, "right"].mean(), "Picks 75%+": g.loc[g["top"] >= .75, "right"].mean(),
                     "75%+ picks": int((g["top"] >= .75).sum())})
    return pd.DataFrame(rows)


def competition_table(bt: pd.DataFrame) -> pd.DataFrame:
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    d = bt.assign(top=P.max(1), right=(P.argmax(1) == bt["y"].to_numpy()))
    rows = []
    for code, g in d.groupby("league"):
        rows.append({"Cup": CUP_NAMES.get(code, code), "Matches": len(g), "Accuracy": g["right"].mean(),
                     "Picks 65%+": g.loc[g["top"] >= .65, "right"].mean() if (g["top"] >= .65).sum() >= 10 else np.nan,
                     "65%+ picks": int((g["top"] >= .65).sum())})
    return pd.DataFrame(rows).sort_values("Matches", ascending=False)


def calibration_table(bt: pd.DataFrame) -> pd.DataFrame:
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    top, right = P.max(1), P.argmax(1) == bt["y"].to_numpy()
    rows = []
    for lo, hi in ((.4, .5), (.5, .6), (.6, .65), (.65, .7), (.7, .75), (.75, .8), (.8, .9), (.9, 1.01)):
        m = (top >= lo) & (top < hi)
        if m.sum():
            rows.append({"Range": f"{lo:.0%}-{min(hi, 1):.0%}", "Matches": int(m.sum()), "Model said": top[m].mean(), "Really happened": right[m].mean()})
    return pd.DataFrame(rows)


def goals_table(bt: pd.DataFrame) -> pd.DataFrame:
    from .intl import goals_table as gt
    return gt(bt)


# ---------------------------------------------------------------- live
def prepare(api, days, results: pd.DataFrame) -> pd.DataFrame | None:
    """All cup rows the daily run needs (finished ones to train on, today's/tomorrow's to predict), with
    country and club names. None when there is nothing to predict. The club state is read in the same
    pass as the league features (features.build_features(..., probes=probes(rows)))."""
    up = upcoming_fixtures(api, days)
    if up.empty:
        return None
    hist = load_history(days[0])
    rows = pd.concat([hist.assign(_up=0), up.assign(_up=1, hg=np.nan, ag=np.nan)], ignore_index=True)
    return assemble(rows, Names(results))


def finish(rows: pd.DataFrame, state: pd.DataFrame, utc_offset: int, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions for today's/tomorrow's cup matches, newest finished cup matches for the tracker)."""
    d = make_features(rows, state)
    train, fx = d[d["_up"] == 0], d[(d["_up"] == 1) & d["usable"]].copy()
    skipped = int(((d["_up"] == 1) & ~d["usable"]).sum())
    if verbose and skipped:
        print(f"  cups: {skipped} matches skipped (a club from a league we don't have)")
    if fx.empty:
        return pd.DataFrame(), _for_tracker(train)
    m = Model().fit(train)
    p = m.predict(fx)
    local = pd.to_datetime(fx["kickoff"].str[:19]) + pd.Timedelta(hours=int(utc_offset))
    out = pd.DataFrame({
        "league": fx["league"].to_numpy(), "home": display(fx, "h"), "away": display(fx, "a"),
        "api_home": fx["api_home"].to_numpy(), "api_away": fx["api_away"].to_numpy(),
        "date_local": local.dt.normalize().to_numpy(), "time_local": local.dt.strftime("%H:%M").to_numpy(),
        "fixture_id": fx["fixture_id"].to_numpy(),
        "competition": [CUP_NAMES[c] + (f" - {r}" if r else "") for c, r in zip(fx["league"], fx["round"])],
        "known_h": fx["known_h"].to_numpy(), "known_a": fx["known_a"].to_numpy(),
    })
    return pd.concat([out, p.reset_index(drop=True).drop(columns="dr")], axis=1), _for_tracker(train)


def display(d: pd.DataFrame, side: str) -> np.ndarray:
    """Our club name when we know the club (same as in the league sheets), else API-Football's name."""
    col = "home" if side == "h" else "away"
    return np.where(d[f"team_{side}"].notna() & (d[f"known_{side}"] == 1), d[f"team_{side}"], d[f"api_{col}"])


def _for_tracker(b: pd.DataFrame) -> pd.DataFrame:
    """Finished cup matches of the last 60 days in the names the predictions use (90 minute score)."""
    if b is None or b.empty:
        return pd.DataFrame()
    b = b[b["date"] >= b["date"].max() - pd.Timedelta(days=60)]
    return pd.DataFrame({"league": b["league"].to_numpy(), "date": b["date"].to_numpy(),
                         "home": display(b, "h"), "away": display(b, "a"), "hg": b["hg"].to_numpy(), "ag": b["ag"].to_numpy()})
