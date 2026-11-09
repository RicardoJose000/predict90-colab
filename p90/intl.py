"""National teams: World Cup and Euro qualifiers, Nations League, AFCON, Copa America, Gold Cup, Asian Cup, friendlies.

Their own model, because the club models learned from club leagues and club odds and know nothing about countries.
Data: every international result since 1872 (github.com/martj42/international_results, free, updated after each window).
Ratings: Elo, the same idea as the World Football Elo ratings: a win against a strong team counts more,
big wins count a bit more, World Cup matches count more than friendlies, and the home team gets a bonus.
Model: Elo + recent goals -> logistic regression and LightGBM (home / draw / away), Poisson goals (over/under, BTTS).
"""
from __future__ import annotations

import re
import time
import unicodedata
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import CACHE, OUTPUT, ROOT, SEED

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE_FILE = CACHE / "intl_results.csv"
SNAPSHOT = ROOT / "data_snapshot" / "intl_results.csv.gz"
BT_FILE = OUTPUT / "backtest_internationals.csv.gz"

# competition groups: code -> (name, Elo weight)
COMPS = {
    "INT_WC": ("World Cup", 60), "INT_WCQ": ("World Cup qualifiers", 40),
    "INT_EURO": ("Euro", 50), "INT_EUROQ": ("Euro qualifiers", 40), "INT_UNL": ("UEFA Nations League", 40),
    "INT_AFCON": ("Africa Cup of Nations", 50), "INT_AFCONQ": ("AFCON qualifiers", 40),
    "INT_COPA": ("Copa America", 50), "INT_GOLD": ("Gold Cup", 50), "INT_CNL": ("CONCACAF Nations League", 40),
    "INT_ASIAN": ("Asian Cup", 50), "INT_ASIANQ": ("Asian Cup qualifiers", 40),
    "INT_FRIENDLY": ("Friendlies", 20), "INT_OTHER": ("Other tournaments", 30),
}
FINALS = {"INT_WC", "INT_EURO", "INT_AFCON", "INT_COPA", "INT_GOLD", "INT_ASIAN"}       # mostly neutral grounds
QUALIFIERS = {"INT_WCQ", "INT_EUROQ", "INT_AFCONQ", "INT_ASIANQ"}
NATIONS_LEAGUES = {"INT_UNL", "INT_CNL"}
NOT_FIFA = re.compile(r"conifa|island games|viva world|muratti|wild cup|elf cup|fifi wild", re.I)

# API-Football name -> name in the results data (the rest match by spelling)
COUNTRY_ALIAS = {
    "USA": "United States", "Korea Republic": "South Korea", "Korea DPR": "North Korea", "China PR": "China",
    "Cape Verde Islands": "Cape Verde", "Chinese Taipei": "Taiwan", "Congo DR": "DR Congo", "Czechia": "Czech Republic",
    "FYR Macedonia": "North Macedonia", "French Guyana": "French Guiana", "Rep. Of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland", "St. Kitts and Nevis": "Saint Kitts and Nevis", "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines", "Türkiye": "Turkey", "Cote D'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast", "Swaziland": "Eswatini", "Macao": "Macau", "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei", "East Timor": "Timor-Leste", "US Virgin Islands": "United States Virgin Islands",
    "IR Iran": "Iran", "Sao Tome and Principe": "São Tomé and Príncipe", "Congo Republic": "Congo",
}

FEATURES = ["dr", "neutral", "elo_h", "elo_a", "gf_h", "ga_h", "gf_a", "ga_a", "friendly", "finals", "qualifier", "nations_league"]
GOAL_FEATURES = FEATURES


def comp_code(tournament: str) -> str | None:
    """Results-data tournament name -> group code. None = not a FIFA national team match."""
    t = str(tournament)
    if NOT_FIFA.search(t):
        return None
    low = t.lower()
    q = "qualification" in low
    if low.startswith("fifa world cup"):
        return "INT_WCQ" if q else "INT_WC"
    if low.startswith("uefa euro"):
        return "INT_EUROQ" if q else "INT_EURO"
    if low == "uefa nations league":
        return "INT_UNL"
    if low.startswith("african cup of nations"):
        return "INT_AFCONQ" if q else "INT_AFCON"
    if low.startswith("copa am"):
        return "INT_OTHER" if q else "INT_COPA"
    if low.startswith("gold cup"):
        return "INT_OTHER" if q else "INT_GOLD"
    if low.startswith("concacaf nations league"):
        return "INT_OTHER" if q else "INT_CNL"
    if low.startswith("afc asian cup"):
        return "INT_ASIANQ" if q else "INT_ASIAN"
    if low == "friendly":
        return "INT_FRIENDLY"
    return "INT_OTHER"


def api_comp_code(name: str) -> str | None:
    """API-Football competition name -> group code. None = women, youth or club football."""
    low = name.lower()
    if re.search(r"women|\bu-?\d\d\b|olympic|club|youth|futsal|beach", low):
        return None
    if "world cup" in low:
        return "INT_WCQ" if "qualification" in low else "INT_WC"
    if "euro championship" in low:
        return "INT_EUROQ" if "qualification" in low else "INT_EURO"
    if "nations league" in low:
        return "INT_UNL" if "uefa" in low else "INT_CNL" if "concacaf" in low else "INT_OTHER"
    if "africa cup of nations" in low:
        return "INT_AFCONQ" if "qualification" in low else "INT_AFCON"
    if "copa america" in low:
        return "INT_COPA"
    if "gold cup" in low:
        return "INT_GOLD" if "qualification" not in low else "INT_OTHER"
    if "asian cup" in low:
        return "INT_ASIANQ" if "qualification" in low else "INT_ASIAN"
    if low == "friendlies":
        return "INT_FRIENDLY"
    return "INT_OTHER"


def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


# ------------------------------------------------------------------ data
def load_results(refresh_hours: float = 6, use_snapshot: bool = False, verbose=True) -> pd.DataFrame:
    """All international results, newest from the internet (saved copy if offline)."""
    src = None
    if use_snapshot and SNAPSHOT.exists():
        src = SNAPSHOT
    else:
        fresh = CACHE_FILE.exists() and (time.time() - CACHE_FILE.stat().st_mtime) / 3600 < refresh_hours
        if not fresh:
            try:
                import requests
                r = requests.get(URL, timeout=60)
                r.raise_for_status()
                CACHE_FILE.write_bytes(r.content)
            except Exception as e:
                if verbose:
                    print(f"  ! international results could not be downloaded ({type(e).__name__}), using the saved copy")
        src = CACHE_FILE if CACHE_FILE.exists() else SNAPSHOT
    d = pd.read_csv(src, encoding="utf-8")
    d = d[d["home_score"].notna() & d["away_score"].notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["neutral"] = d["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    d["league"] = d["tournament"].map(comp_code)
    d = d.rename(columns={"home_team": "home", "away_team": "away", "home_score": "hg", "away_score": "ag"})
    d["hg"], d["ag"] = d["hg"].astype(int), d["ag"].astype(int)
    return d.sort_values("date", kind="stable").reset_index(drop=True)


def save_snapshot(verbose=True) -> None:
    df = pd.read_csv(CACHE_FILE, encoding="utf-8") if CACHE_FILE.exists() else None
    if df is not None:
        SNAPSHOT.parent.mkdir(exist_ok=True)
        df.to_csv(SNAPSHOT, index=False, compression={"method": "gzip", "mtime": 0})
        if verbose:
            print(f"International snapshot saved: {len(df):,} matches")


# -------------------------------------------------------------- ratings
class Ratings:
    """Elo for every national team, plus recent goals for and against, updated match by match."""
    HOME = 100.0
    ALPHA = 0.15          # weight of the newest match in the recent goals average

    def __init__(self):
        self.elo: dict = {}
        self.gf: dict = {}
        self.ga: dict = {}
        self.n: dict = {}

    def state(self, team):
        return self.elo.get(team, 1500.0), self.gf.get(team, 1.3), self.ga.get(team, 1.3), self.n.get(team, 0)

    def row(self, home, away, neutral, code) -> dict:
        eh, gfh, gah, nh = self.state(home)
        ea, gfa, gaa, na = self.state(away)
        return {"elo_h": eh, "elo_a": ea, "dr": eh - ea + (0 if neutral else self.HOME), "neutral": int(neutral),
                "gf_h": gfh, "ga_h": gah, "gf_a": gfa, "ga_a": gaa, "n_h": nh, "n_a": na,
                "friendly": int(code == "INT_FRIENDLY"), "finals": int(code in FINALS),
                "qualifier": int(code in QUALIFIERS), "nations_league": int(code in NATIONS_LEAGUES)}

    def update(self, home, away, hg, ag, neutral, code):
        eh, gfh, gah, nh = self.state(home)
        ea, gfa, gaa, na = self.state(away)
        dr = eh - ea + (0 if neutral else self.HOME)
        we = 1 / (10 ** (-dr / 400) + 1)
        w = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        k = COMPS.get(code, ("", 30))[1] * mult
        self.elo[home], self.elo[away] = eh + k * (w - we), ea - k * (w - we)
        a = self.ALPHA
        self.gf[home], self.ga[home] = (1 - a) * gfh + a * hg, (1 - a) * gah + a * ag
        self.gf[away], self.ga[away] = (1 - a) * gfa + a * ag, (1 - a) * gaa + a * hg
        self.n[home], self.n[away] = nh + 1, na + 1


def build_features(res: pd.DataFrame) -> tuple[pd.DataFrame, Ratings]:
    """One row per match with the ratings from BEFORE kickoff, and the ratings after the last match."""
    r = Ratings()
    rows = []
    for m in res.itertuples(index=False):
        code = m.league or "INT_OTHER"
        rows.append(r.row(m.home, m.away, m.neutral, code))
        r.update(m.home, m.away, m.hg, m.ag, m.neutral, code)
    feat = pd.concat([res.reset_index(drop=True), pd.DataFrame(rows).drop(columns="neutral")], axis=1)
    feat["y"] = np.select([feat["hg"] > feat["ag"], feat["hg"] == feat["ag"]], [0, 1], 2)
    return feat, r


# ---------------------------------------------------------------- model
class Model:
    """1X2: average of logistic regression and LightGBM. Goals: two Poisson regressions."""

    def fit(self, train: pd.DataFrame):
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression, PoissonRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = train[FEATURES].to_numpy(float), train["y"].to_numpy(int)
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
        self.gb = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                 verbose=-1).fit(pd.DataFrame(X, columns=FEATURES), y)
        G = train[GOAL_FEATURES].to_numpy(float)
        self.gh = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)).fit(G, train["hg"].clip(upper=10))
        self.ga = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)).fit(G, train["ag"].clip(upper=10))
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        X = d[FEATURES].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = 0.5 * self.lr.predict_proba(X) + 0.5 * self.gb.predict_proba(pd.DataFrame(X, columns=FEATURES))
        G = d[GOAL_FEATURES].to_numpy(float)
        lh, la = self.gh.predict(G), self.ga.predict(G)
        out = pd.DataFrame(index=d.index)
        out["p_h"], out["p_d"], out["p_a"] = P[:, 0], P[:, 1], P[:, 2]
        out["xg_h"], out["xg_a"] = lh, la
        out = pd.concat([out, goal_markets(lh, la, d.index)], axis=1)
        return out


def goal_markets(lh, la, index) -> pd.DataFrame:
    """Over/under, BTTS, team to score and win-by-2 chances from the two expected goal numbers."""
    k = np.arange(11)
    ph = poisson.pmf(k[None, :], np.asarray(lh)[:, None])
    pa = poisson.pmf(k[None, :], np.asarray(la)[:, None])
    J = ph[:, :, None] * pa[:, None, :]
    i, j = np.meshgrid(k, k, indexing="ij")
    tot = i + j
    s = lambda m: (J * m).sum((1, 2))
    return pd.DataFrame({"p_over": s(tot >= 3), "p_over15": s(tot >= 2), "p_over35": s(tot >= 4),
                         "p_btts": s((i > 0) & (j > 0)), "p_h_scores": 1 - ph[:, 0], "p_a_scores": 1 - pa[:, 0],
                         "p_h_by2": s(i - j >= 2), "p_a_by2": s(j - i >= 2)}, index=index)


def training_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """FIFA matches since 1990 where both teams already have a rating history."""
    return feat[feat["league"].notna() & (feat["date"] >= "1990-01-01") & (feat["n_h"] >= 5) & (feat["n_a"] >= 5)]


# ------------------------------------------------------------- backtest
def backtest(test_from: int = 2016, use_snapshot: bool = True, verbose=True) -> pd.DataFrame:
    """Each year predicted by a model trained only on the years before it. Ratings are always from before kickoff."""
    res = load_results(use_snapshot=use_snapshot, verbose=verbose)
    feat, _ = build_features(res)
    rows = training_rows(feat)
    out = []
    for y in range(test_from, int(rows["date"].dt.year.max()) + 1):
        test = rows[rows["date"].dt.year == y]
        if test.empty:
            continue
        m = Model().fit(rows[rows["date"] < pd.Timestamp(y, 1, 1)])
        p = pd.concat([test[["date", "league", "home", "away", "hg", "ag", "y", "dr", "neutral"]], m.predict(test)], axis=1)
        p["fold"] = str(y)
        out.append(p)
        if verbose:
            acc = (p[["p_h", "p_d", "p_a"]].to_numpy().argmax(1) == p["y"]).mean()
            print(f"  {y}: tested {len(p):,} matches, accuracy {acc:.1%}", flush=True)
    bt = pd.concat(out, ignore_index=True)
    bt.to_csv(BT_FILE, index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.4f")
    return bt


def load_backtest() -> pd.DataFrame | None:
    return pd.read_csv(BT_FILE, parse_dates=["date"]) if BT_FILE.exists() else None


def accuracy_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Per competition group: how often the pick was right, against simply picking the higher rated team."""
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    d = bt.assign(top=P.max(1), right=(P.argmax(1) == bt["y"].to_numpy()),
                  elo_right=np.where(bt["dr"] >= 0, 0, 2) == bt["y"].to_numpy(),
                  ll=-np.log(np.clip(P[np.arange(len(P)), bt["y"].to_numpy(int)], 1e-9, 1)))
    groups = [("All", d), ("Competitive (no friendlies)", d[d["league"] != "INT_FRIENDLY"]),
              ("Friendlies", d[d["league"] == "INT_FRIENDLY"])]
    groups += [(COMPS[c][0], d[d["league"] == c]) for c in COMPS if c not in ("INT_FRIENDLY",) and (d["league"] == c).sum() >= 80]
    rows = []
    for name, g in groups:
        def at(p):
            m = g["top"] >= p
            return g.loc[m, "right"].mean() if m.sum() >= 20 else np.nan, m.mean()
        a50, s50 = at(0.5)
        a65, s65 = at(0.65)
        a75, s75 = at(0.75)
        rows.append({"Matches": len(g), "Group": name, "Accuracy": g["right"].mean(), "Higher rated wins": g["elo_right"].mean(),
                     "Picks 50%+": a50, "Share 50%+": s50, "Picks 65%+": a65, "Share 65%+": s65,
                     "Picks 75%+": a75, "Share 75%+": s75, "Log-loss": g["ll"].mean()})
    return pd.DataFrame(rows)[["Group", "Matches", "Accuracy", "Higher rated wins", "Picks 50%+", "Share 50%+",
                               "Picks 65%+", "Share 65%+", "Picks 75%+", "Share 75%+", "Log-loss"]]


def season_table(bt: pd.DataFrame, competitive=True) -> pd.DataFrame:
    d = bt[bt["league"] != "INT_FRIENDLY"] if competitive else bt
    P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
    d = d.assign(top=P.max(1), right=(P.argmax(1) == d["y"].to_numpy()))
    out = []
    for y, g in d.groupby("fold"):
        out.append({"Year": y, "Matches": len(g), "Accuracy": g["right"].mean(),
                    "Picks 50%+": g.loc[g["top"] >= .5, "right"].mean(), "Picks 65%+": g.loc[g["top"] >= .65, "right"].mean(),
                    "65%+ picks": int((g["top"] >= .65).sum())})
    return pd.DataFrame(out)


def calibration_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Competitive matches: when the pick's chance was in this range, how often did it win?"""
    d = bt[bt["league"] != "INT_FRIENDLY"]
    P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
    top, right = P.max(1), P.argmax(1) == d["y"].to_numpy()
    rows = []
    for lo, hi in ((.4, .5), (.5, .6), (.6, .65), (.65, .7), (.7, .75), (.75, .8), (.8, .9), (.9, 1.01)):
        m = (top >= lo) & (top < hi)
        if m.sum():
            rows.append({"Range": f"{lo:.0%}-{min(hi, 1):.0%}", "Matches": int(m.sum()), "Model said": top[m].mean(), "Really happened": right[m].mean()})
    return pd.DataFrame(rows)


def goals_table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hg, ag = bt["hg"].to_numpy(), bt["ag"].to_numpy()
    for name, p, real in (("Over 2.5", bt["p_over"], hg + ag >= 3), ("Over 1.5", bt["p_over15"], hg + ag >= 2),
                          ("Under 3.5", 1 - bt["p_over35"], hg + ag <= 3), ("BTTS yes", bt["p_btts"], (hg > 0) & (ag > 0))):
        p = p.to_numpy()
        pick = p >= 0.5
        sure = np.maximum(p, 1 - p) >= 0.6
        rows.append({"Bet": name, "Happened": real.mean(), "Model said": p.mean(),
                     "Right": (pick == real).mean(), "Always likelier side": max(real.mean(), 1 - real.mean()),
                     "Model said (60%+)": (pick == real)[sure].mean() if sure.any() else np.nan, "Share 60%+": sure.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- live
def upcoming(api, days, utc_offset: int, ratings: Ratings, names: set, verbose=True) -> pd.DataFrame:
    """Senior national team matches today/tomorrow from API-Football (the same saved fixture lists as the clubs)."""
    index = {_key(n): n for n in names}
    rows, skipped = [], set()
    for d in days:
        data = api.get("fixtures", {"date": d.isoformat()}, max_age_hours=2)
        for fx in (data or {}).get("response", []):
            lg = fx["league"]
            if lg.get("country") != "World" or fx["fixture"]["status"]["short"] not in ("NS", "TBD"):
                continue
            code = api_comp_code(lg["name"])
            if code is None:
                continue
            teams = []
            for side in ("home", "away"):
                n = fx["teams"][side]["name"]
                teams.append(index.get(_key(COUNTRY_ALIAS.get(n, n))))
                if teams[-1] is None and not re.search(r"\bu-?\d\d\b| w$|women", n.lower()):
                    skipped.add(n)
            h, a = teams
            if not h or not a or h == a:
                continue
            ko = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
            local = ko.replace(tzinfo=None) + timedelta(hours=utc_offset)
            neutral = code in FINALS
            rows.append({"league": code, "home": h, "away": a, "api_home": fx["teams"]["home"]["name"],
                         "api_away": fx["teams"]["away"]["name"], "date_local": pd.Timestamp(local.date()),
                         "time_local": local.strftime("%H:%M"), "fixture_id": fx["fixture"]["id"],
                         "competition": lg["name"], **ratings.row(h, a, neutral, code)})
    if skipped and verbose:
        print(f"  (internationals: {len(skipped)} team names not known: {', '.join(sorted(skipped)[:6])})")
    return pd.DataFrame(rows).drop_duplicates(subset=["league", "home", "away"]) if rows else pd.DataFrame()


def predict_upcoming(api, days, s: dict, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions for the national team matches, all international results for the tracker)."""
    res = load_results(float(s.get("refresh_hours", 6)), verbose=verbose)
    feat, ratings = build_features(res)
    names = set(res.loc[res["date"] >= res["date"].max() - pd.Timedelta(days=4 * 365), "home"]) | \
        set(res.loc[res["date"] >= res["date"].max() - pd.Timedelta(days=4 * 365), "away"])
    fx = upcoming(api, days, int(s.get("utc_offset", 1)), ratings, names, verbose)
    if fx.empty:
        return fx, res
    model = Model().fit(training_rows(feat))
    pred = pd.concat([fx, model.predict(fx)], axis=1)
    return pred, res
