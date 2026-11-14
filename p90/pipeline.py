"""The whole system in a few calls.

    from p90 import pipeline
    out = pipeline.daily()          # data -> features -> models -> predictions -> Excel
    bt  = pipeline.backtest()       # walk-forward test of every season (slow, run weekly at most)
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from . import bangers, betting, cups, data, evaluate, intl, live, props, report, tracker, webpage
from .backtest import fit_all, past_warmup, predict_all, walk_forward
from .config import CUP_LEAGUES, DATA, INTL_LEAGUES, LEAGUES, MAIN_LEAGUES, MODELS, OUTPUT, ROOT, current_season_start, league_label, load_settings
from .config import league_type as _league_type
from .features import build_features

MODEL_FILE = MODELS / "models.joblib"
BT_FILE = OUTPUT / "backtest_predictions.csv.gz"
BT_PROPS_FILE = OUTPUT / "backtest_more_bets.csv.gz"
SNAPSHOT = ROOT / "data_snapshot" / "results_snapshot.csv.gz"
NAMES = {"H": "Home", "D": "Draw", "A": "Away"}


def log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


# ------------------------------------------------------------------ upcoming
def upcoming_matches(results: pd.DataFrame, s: dict, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    utc = int(s.get("utc_offset", 1))
    fd = data.load_fixtures_fd(s["leagues"], results, s["days_ahead"])
    if not fd.empty:
        loc = [live.uk_to_local(d, t, utc) for d, t in zip(fd["date"], fd["time"])]
        fd["date_local"] = [x[0] for x in loc]
        fd["time_local"] = [x[1] for x in loc]
    log(f"  football-data.co.uk: {len(fd)} upcoming matches with odds", verbose)

    api = live.ApiFootball(verbose=verbose)
    inj = pd.DataFrame(columns=["fixture_id", "team", "player", "type", "reason"])
    info = {"api": "off"}
    if s.get("use_api_football") and api.ok:
        today = date.today()
        days = [today, today + timedelta(days=1)]
        matcher = live.TeamMatcher(results)
        ax = live.api_fixtures(api, days, s["leagues"], matcher, utc)
        log(f"  API-Football: {len(ax)} matches today/tomorrow in your leagues", verbose)
        if len(matcher.unmatched):
            log(f"  (skipped {len(matcher.unmatched)} teams with unknown names - see team_aliases.csv)", verbose)
        if not ax.empty:
            if fd.empty:
                fx = ax.assign(source="API-Football")
            else:
                fx = fd.merge(ax[["league", "home", "away", "fixture_id", "api_home", "api_away",
                                  "date_local", "time_local"]],
                              on=["league", "home", "away"], how="outer", suffixes=("", "_api"))
                for c in ("date_local", "time_local"):
                    fx[c] = fx[f"{c}_api"].where(fx[f"{c}_api"].notna(), fx[c])
                fx = fx.drop(columns=[f"date_local_api", f"time_local_api"])
                fx["source"] = fx["source"].fillna("API-Football")
            need = fx["o_h"].isna() if "o_h" in fx else pd.Series(True, index=fx.index)
            # most wanted first: matches with no odds at all, then main-league matches, today first,
            # because football-data has no double chance or BTTS odds for the slips
            order = fx.assign(_no=~need, _other=~fx["league"].isin(MAIN_LEAGUES),
                              _d=pd.to_datetime(fx["date_local"])).sort_values(["_no", "_other", "_d"])
            order = order[~order["_no"] | ~order["_other"]]
            ids = list(dict.fromkeys(order["fixture_id"].dropna().astype(int)))
            odds = live.api_odds(api, days, ids, per_match_budget=int(s.get("api_odds_budget", 30)))
            for col in ("b_h", "b_d", "b_a", "b_over", "b_under", "b_btts_yes", "b_btts_no", "b_1x", "b_x2"):
                vals = fx["fixture_id"].map(lambda f: odds.get(int(f), {}).get(col) if pd.notna(f) else None)
                if col in fx:
                    fx[col] = fx[col].where(fx[col].notna(), vals)
                else:
                    fx[col] = vals
            fx = fx.infer_objects()
            fx["more_odds"] = fx["fixture_id"].map(lambda f: odds.get(int(f), {}).get("more") if pd.notna(f) else None)
            for a, b in (("o_h", "b_h"), ("o_d", "b_d"), ("o_a", "b_a"), ("o_over", "b_over"), ("o_under", "b_under")):
                if a not in fx:
                    fx[a] = np.nan
                fx[a] = pd.to_numeric(fx[a], errors="coerce").fillna(pd.to_numeric(fx[b], errors="coerce"))
            inj = live.api_injuries(api, days)
        else:
            fx = fd
        info = {"api": "on", "requests_used": api.used, "requests_left": api.remaining}
        log(f"  API-Football requests used now: {api.used}" + (f", daily limit reached, {api.stale_used} older saved answers used" if api.limit_reached else f", left today: {api.remaining}" if api.remaining else " (all from saved copies)"), verbose)
    else:
        fx = fd
        if s.get("use_api_football") and not api.ok:
            log("  API-Football key not set - injuries and extra fixtures skipped (see README)", verbose)
    if fx.empty:
        return fx, inj, info
    for c in data.COLUMNS:
        if c not in fx:
            fx[c] = np.nan
    for c in ("o_h", "o_d", "o_a", "m_h", "m_d", "m_a", "b_h", "b_d", "b_a", "o_over", "o_under",
              "m_over", "m_under", "b_over", "b_under", "hc", "ac", "hy", "ay", "hr", "ar", "hs", "as_", "hst", "ast", "hg", "ag"):
        fx[c] = pd.to_numeric(fx[c], errors="coerce")

    fx["country"] = fx["league"].map(lambda code: LEAGUES[code][0])
    fx["tier"] = fx["league"].map(lambda code: LEAGUES[code][2])
    fx["date"] = pd.to_datetime(fx["date"]).fillna(pd.to_datetime(fx["date_local"]))
    fx["time"] = fx["time"].fillna("") if "time" in fx else ""
    fx = data.fill_season_labels(fx, results)
    # drop matches that already kicked off (local time)
    now_local = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=utc)
    ko = pd.to_datetime(fx["date_local"].astype(str).str[:10] + " " + fx["time_local"].fillna("").astype(str).str[:5],
                        errors="coerce")
    ko = ko.fillna(pd.to_datetime(fx["date_local"]) + pd.Timedelta(hours=23, minutes=59))
    fx = fx[ko >= pd.Timestamp(now_local) - pd.Timedelta(minutes=10)]
    fx = fx.drop_duplicates(subset=["league", "home", "away"]).reset_index(drop=True)
    return fx, inj, info


# -------------------------------------------------------------------- models
def get_models(feat: pd.DataFrame, s: dict, force=False, verbose=True) -> dict:
    # saved models only work with the library versions that made them, so new versions mean new training
    import lightgbm, sklearn, xgboost
    sig = {"leagues": sorted(s["leagues"]), "first_season": s["first_season"], "version": 3,
           "libs": [sklearn.__version__, lightgbm.__version__, xgboost.__version__]}
    if MODEL_FILE.exists() and not force:
        try:
            with warnings.catch_warnings():         # models from other versions warn here, then get retrained below
                warnings.simplefilter("ignore")
                saved = joblib.load(MODEL_FILE)
            age = (datetime.now() - saved["trained_at"]).days
            if saved.get("sig") == sig and age < s["retrain_days"]:
                log(f"  using models trained {saved['trained_at']:%d %b %Y %H:%M} "
                    f"(retrain every {s['retrain_days']} days)", verbose)
                return saved["models"]
        except Exception as e:                      # e.g. saved with other library versions
            log(f"  saved models could not be used ({type(e).__name__}) - training new ones", verbose)
    t = time.time()
    played = past_warmup(feat[feat["y"].notna()])
    log(f"  training the stacked models on {len(played):,} matches (takes a few minutes)...", verbose)
    models = fit_all(played)
    models["props"] = props.fit(played)
    joblib.dump({"models": models, "trained_at": datetime.now(), "sig": sig}, MODEL_FILE)
    log(f"  models trained and saved ({time.time() - t:.0f}s)", verbose)
    return models


# --------------------------------------------------------------------- daily
def tier_label(p, s):
    return np.select([p >= s["banger_threshold"], p >= s["strong_threshold"], p >= s["confident_threshold"]],
                     ["BANGER", "STRONG", "GOOD"], "WEAK")


def league_type(league) -> pd.Series:
    return pd.Series(league).map(_league_type)


INTL_ODDS = ("b_h", "b_d", "b_a", "b_over", "b_under", "b_btts_yes", "b_btts_no", "b_1x", "b_x2")


def internationals(s: dict, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """National team matches today/tomorrow with their own model (p90/intl.py), in the same columns as the clubs."""
    if not s.get("internationals", True):
        return pd.DataFrame(), None
    api = live.ApiFootball(verbose=verbose)
    if not (s.get("use_api_football") and api.ok):
        log("  internationals: need the API-Football key (football-data.co.uk has no national teams)", verbose)
        return pd.DataFrame(), None
    days = [date.today(), date.today() + timedelta(days=1)]
    try:
        p, res = intl.predict_upcoming(api, days, s, verbose)
    except Exception as e:                       # never let them stop the club predictions
        log(f"  ! internationals skipped this time: {type(e).__name__}: {e}", verbose)
        return pd.DataFrame(), None
    log(f"  internationals: {len(p)} national team matches today/tomorrow", verbose)
    if p.empty:
        return p, res
    p = _attach_odds(p, api, days, int(s.get("intl_odds_budget", 20)))
    p["model"] = "national teams"
    return p, res


def _attach_odds(p: pd.DataFrame, api, days, budget: int) -> pd.DataFrame:
    """Bet365 odds for national team or cup matches, in the same columns as the clubs."""
    odds = live.api_odds(api, days, [int(f) for f in p["fixture_id"]], per_match_budget=budget)
    for c in INTL_ODDS:
        p[c] = pd.to_numeric(p["fixture_id"].map(lambda f: odds.get(int(f), {}).get(c)), errors="coerce")
    p["more_odds"] = p["fixture_id"].map(lambda f: odds.get(int(f), {}).get("more"))
    for a, b in (("o_h", "b_h"), ("o_d", "b_d"), ("o_a", "b_a"), ("o_over", "b_over"), ("o_under", "b_under")):
        p[a] = p[b]
    inv = 1 / p[["b_h", "b_d", "b_a"]].to_numpy(float)
    p["mk_h"], p["mk_d"], p["mk_a"] = (inv / inv.sum(1, keepdims=True)).T
    inv2 = 1 / p[["b_over", "b_under"]].to_numpy(float)
    p["mk_over"] = inv2[:, 0] / inv2.sum(1)
    p["date"] = pd.to_datetime(p["date_local"])
    p["source"] = "API-Football"
    return p


def cup_rows(results: pd.DataFrame, s: dict, verbose=True) -> pd.DataFrame | None:
    """Cup matches to predict today/tomorrow, plus the finished ones to train on (p90/cups.py)."""
    if not s.get("cups", True):
        return None
    api = live.ApiFootball(verbose=verbose)
    if not (s.get("use_api_football") and api.ok):
        log("  cups: need the API-Football key (football-data.co.uk has no cups)", verbose)
        return None
    try:
        return cups.prepare(api, [date.today(), date.today() + timedelta(days=1)], results)
    except Exception as e:                       # never let them stop the league predictions
        log(f"  ! cups skipped this time: {type(e).__name__}: {e}", verbose)
        return None


def cup_matches(rows, state, s: dict, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows is None or state is None:
        return pd.DataFrame(), pd.DataFrame()
    try:
        p, res = cups.finish(rows, state, int(s.get("utc_offset", 1)), verbose)
    except Exception as e:
        log(f"  ! cups skipped this time: {type(e).__name__}: {e}", verbose)
        return pd.DataFrame(), pd.DataFrame()
    log(f"  cups: {len(p)} cup matches today/tomorrow", verbose)
    if p.empty:
        return p, res
    days = [date.today(), date.today() + timedelta(days=1)]
    p = _attach_odds(p, live.ApiFootball(verbose=verbose), days, int(s.get("cups_odds_budget", 15)))
    p["model"] = "cups"
    return p, res


def intl_results_for_tracker(res: pd.DataFrame | None) -> pd.DataFrame:
    if res is None or res.empty:
        return pd.DataFrame()
    r = res[res["league"].notna() & (res["date"] >= res["date"].max() - pd.Timedelta(days=60))]
    return r[["league", "date", "home", "away", "hg", "ag"]]


def adjust_for_injuries(pred: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Small shift for missing players, at the size measured on 6,248 matches (2022-2025, 7 top
    leagues): about 0.3 percentage points per missing player. The odds already price most of it."""
    gap = (pred["miss_h"] - pred["miss_a"]).clip(-8, 8).fillna(0).to_numpy(float)
    P = np.clip(pred[["p_h", "p_d", "p_a"]].to_numpy(float), 1e-4, 1)
    Z = np.log(P) + np.c_[-beta * gap, np.zeros(len(gap)), beta * gap]
    Z = np.exp(Z)
    Z = Z / Z.sum(1, keepdims=True)
    pred = pred.copy()
    pred["p_h"], pred["p_d"], pred["p_a"] = Z[:, 0], Z[:, 1], Z[:, 2]
    return pred


def _update_tracker(entries: list, results: pd.DataFrame, s: dict, verbose=True):
    today = (datetime.now(timezone.utc).replace(tzinfo=None)
             + timedelta(hours=int(s.get("utc_offset", 1)))).strftime("%Y-%m-%d")
    try:
        tr = tracker.settle(tracker.add(tracker.load(), entries, today), results)
        tracker.save(tr)
        return tr
    except PermissionError:
        log("  ! bet_tracker.xlsx is open in Excel, so it was not updated. Close it and run again.", verbose)
        return None


def daily(force_retrain=False, verbose=True, write_excel=True) -> dict:
    t0 = time.time()
    s = load_settings()
    log(f"1/5 Results and odds for {len(s['leagues'])} leagues", verbose)
    results = data.load_results(s["leagues"], s["first_season"], s["refresh_hours"], verbose=verbose)
    log("2/5 Upcoming matches", verbose)
    fx, inj, info = upcoming_matches(results, s, verbose)
    ip, ires = internationals(s, verbose)
    crows = cup_rows(results, s, verbose)
    log("3/5 Building features", verbose)
    both = pd.concat([results, fx], ignore_index=True)
    cstate = None
    if crows is not None:                        # the cup matches read the clubs' league state in the same pass
        base, cstate = build_features(both, probes=cups.probes(crows))
    else:
        base = build_features(both)
    feat = props.add_features(base)
    cp, cres = cup_matches(crows, cstate, s, verbose)
    log("4/5 Models", verbose)
    models = get_models(feat, s, force=force_retrain, verbose=verbose)
    up = feat[feat["y"].isna()].copy()
    settle_with = pd.concat([results, intl_results_for_tracker(ires), cres], ignore_index=True)
    if up.empty and ip.empty and cp.empty:
        log("No upcoming matches found for the next days.", verbose)
        _update_tracker([], settle_with, s, verbose)
        return {"predictions": pd.DataFrame(), "results": results, "settings": s, "feat": feat,
                "slips": {}, "strong_slips": {}, "banger_slips": {}, "other_slips": {}, "my_picks": {}, "intl_slips": {}, "market_accas": {},
                "mixed_accas": {}, "cup_slips": {},
                "value": pd.DataFrame(), "injuries": inj, "info": info}

    log("5/5 Predictions", verbose)
    pred = pd.DataFrame()
    if not up.empty:
        extra = [c for c in ("date_local", "time_local", "fixture_id", "api_home", "api_away", "source",
                             "b_btts_yes", "b_btts_no", "b_1x", "b_x2", "more_odds") if c in up.columns]
        extra += [c for c in up.columns if c.endswith(("_f_h", "_f_a"))]
        pred = predict_all(models, up)
        pred[extra] = up[extra]
        pred = pd.concat([pred, props.predict(models["props"], up)], axis=1)
    for extra in (ip, cp):                       # national teams and cups, each with its own model
        if extra.empty:
            continue
        keep = [c for c in extra.columns if c in pred.columns or c.startswith(("p_", "b_", "o_", "mk_")) or c in
                ("league", "home", "away", "date", "date_local", "time_local", "fixture_id", "api_home", "api_away",
                 "source", "model", "more_odds", "xg_h", "xg_a", "competition")]
        pred = pd.concat([pred, extra[keep]], ignore_index=True) if len(pred) else extra[keep].reset_index(drop=True)
    # a day with only national teams (or only clubs) still needs every column the sheets use
    club_only = [f"p_{b}" for b in models["props"]["goals"]] + \
        [f"{k}_{n}" for n in models["props"]["counts"] for k in ("mu", "alpha")] + \
        [c for c in feat.columns if c.endswith(("_f_h", "_f_a"))]
    for c in club_only + ["o_h", "o_d", "o_a", "b_h", "b_d", "b_a", "m_h", "m_d", "m_a", "mk_h", "mk_d", "mk_a", "mk_over",
                          "o_over", "o_under", "b_btts_yes", "b_btts_no", "b_1x", "b_x2", "more_odds", "p_over", "p_btts"]:
        if c not in pred:
            pred[c] = np.nan
    pred["odds_h"] = pred["o_h"].fillna(pred["b_h"])
    pred["odds_d"] = pred["o_d"].fillna(pred["b_d"])
    pred["odds_a"] = pred["o_a"].fillna(pred["b_a"])
    pred = pred.sort_values(["date_local", "time_local", "league"]).copy()

    # injuries per team
    pred["miss_h"], pred["miss_a"] = np.nan, np.nan
    if not inj.empty and "fixture_id" in pred:
        inj = inj.drop_duplicates(subset=["fixture_id", "team", "player"])
        miss = inj[inj["type"].str.contains("Missing", na=False)]
        doubt = inj[~inj["type"].str.contains("Missing", na=False)]
        count = miss.groupby(["fixture_id", "team"]).size().to_dict()
        pred["miss_h"] = [count.get((f, t), 0) if pd.notna(f) else np.nan
                          for f, t in zip(pred["fixture_id"], pred.get("api_home", pd.Series("", index=pred.index)))]
        pred["miss_a"] = [count.get((f, t), 0) if pd.notna(f) else np.nan
                          for f, t in zip(pred["fixture_id"], pred.get("api_away", pd.Series("", index=pred.index)))]
        if s.get("injury_adjust", True):
            pred = adjust_for_injuries(pred, float(s.get("injury_beta", 0.0095)))
        def names(df, fid, team):
            m = df[(df["fixture_id"] == fid) & (df["team"] == team)]
            return ", ".join(m["player"].astype(str).head(8))
        for side in ("home", "away"):
            pred[f"out_{side}"] = [names(miss, f, t) if pd.notna(f) else "" for f, t in zip(pred["fixture_id"], pred.get(f"api_{side}", ""))]
            pred[f"doubt_{side}"] = [names(doubt, f, t) if pd.notna(f) else "" for f, t in zip(pred["fixture_id"], pred.get(f"api_{side}", ""))]

    P = pred[["p_h", "p_d", "p_a"]].to_numpy(float)
    pred["pick"] = np.array(list("HDA"))[P.argmax(1)]
    pred["top"] = P.max(1)
    pred["confidence"] = tier_label(pred["top"], s)
    is_intl = pred["league"].isin(INTL_LEAGUES)
    is_cup = pred["league"].isin(CUP_LEAGUES)
    pred.loc[pred["league"] == "INT_FRIENDLY", "confidence"] = "WEAK"     # squads change a lot in friendlies
    # European cups: their 75%+ picks were right only 71% in the test, so never BANGER
    pred.loc[pred["league"].isin(bangers.EUROPEAN_CUPS) & (pred["confidence"] == "BANGER"), "confidence"] = "STRONG"

    # value bets need a model that knows the market: the national team and cup models have no old odds to learn from
    value = betting.value_bets(pred[pred["odds_h"].notna() & ~is_intl & ~is_cup], s, ("odds_h", "odds_d", "odds_a"))
    slips, strong_slips, banger_slips, other_slips, my_picks, intl_slips, cup_slips = {}, {}, {}, {}, {}, {}, {}
    competitive = pred[is_intl & ~pred["league"].isin(["INT_FRIENDLY", "INT_OTHER"]) & pred["odds_h"].notna()]
    cup_fav = pred[is_cup & pred["odds_h"].notna()]
    pool = betting.acca_pool(pred[~is_intl & ~is_cup], s)
    # favourites to win: competitive national team matches (no friendlies) and cup matches can join
    fav_pool = pd.concat([pool] + ([competitive] if s.get("intl_in_picks", True) else [])
                         + ([cup_fav] if s.get("cups_in_picks", True) else []))

    # one_slip_per_match: a match goes into one acca only, across ALL the acca sheets, so one bad result
    # can only hurt one slip. The accas are built in this order, each from the matches still free:
    # My picks, Mixed acca, Banger accas, Cup acca, International acca, Strong, Banger, Main, Other.
    one = bool(s.get("one_slip_per_match", True))
    used: set = set()

    def free(df):
        return df[~df.index.isin(used)] if one else df

    def take(target, key, built):
        if built:
            target[key] = built
            if one:
                used.update(i for sl in built for i, *_ in sl["legs"])

    days = sorted(set(fav_pool["date_local"]) | set(pred["date_local"]))
    for day in days:
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        fav = fav_pool[(fav_pool["date_local"] == day) & fav_pool["odds_h"].notna()]
        take(my_picks, key, betting.build_my_picks(fav, s) if len(fav) else [])
    mixed_accas = bangers.mixed_accas(free(pred), s)          # {day: [slip]}
    for key, built in list(mixed_accas.items()):
        take(mixed_accas, key, built)
    market_accas = bangers.accas(pred, s, used if one else None)   # {market tab: {day: [slip]}}, fills `used`
    for target, frame, min_prob in ((cup_slips, cup_fav, s.get("cup_acca_min_prob", 0.70)),
                                    (intl_slips, competitive, s.get("intl_acca_min_prob", 0.70))):
        for day, g in free(frame).groupby("date_local"):
            take(target, pd.Timestamp(day).strftime("%Y-%m-%d"), betting.build_favourites(g, s, float(min_prob)))
    for day in days:
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        fav = free(fav_pool[(fav_pool["date_local"] == day) & fav_pool["odds_h"].notna()])
        take(strong_slips, key, betting.build_favourites(fav, s, s["strong_slips_min_prob"]) if len(fav) else [])
        fav = free(fav)
        take(banger_slips, key, betting.build_favourites(fav, s, s["banger_slips_min_prob"]) if len(fav) else [])
        g = free(pool[pool["date_local"] == day])
        priced = g[g["odds_h"].notna()]
        take(slips, key, betting.build_accas(priced, s, "live") if len(priced) else [])
        g = free(g)
        take(other_slips, key, betting.build_other(g, s) if len(g) else [])

    out = {"predictions": pred, "value": value, "slips": slips, "strong_slips": strong_slips,
           "banger_slips": banger_slips, "other_slips": other_slips, "my_picks": my_picks, "intl_slips": intl_slips,
           "market_accas": market_accas, "mixed_accas": mixed_accas, "cup_slips": cup_slips, "injuries": inj, "results": results,
           "settings": s, "info": info, "feat": feat}
    if write_excel:
        path = export_excel(out)
        out["excel"] = path
        sets = (("Main acca", slips), ("Strong acca", strong_slips), ("Banger acca", banger_slips), ("Other acca", other_slips),
                ("Picks acca", my_picks), ("International acca", intl_slips), ("Cup acca", cup_slips), ("Mixed acca", mixed_accas),
                *((f"{tab} acca", by_day) for tab, by_day in market_accas.items()))
        out["tracker"] = _update_tracker(tracker.entries_from(pred, value, sets, s), settle_with, s, verbose)
        log(f"\nSaved: {path.name} and {path.stem}.html in the output folder", verbose)
    log(f"Done in {time.time() - t0:.0f}s: {len(pred)} matches, "
        f"{int((pred['confidence'] == 'BANGER').sum())} bangers, {int((pred['confidence'] == 'STRONG').sum())} strong, "
        f"{len(value)} value bets, accas: {sum(len(v) for v in slips.values())} main, {sum(len(v) for v in strong_slips.values())} strong, "
        f"{sum(len(v) for v in banger_slips.values())} banger, {sum(len(v) for v in other_slips.values())} other, "
        f"{sum(len(v) for v in my_picks.values())} my picks, "
        f"{sum(len(x) for v in market_accas.values() for x in v.values())} banger accas by market, "
        f"{sum(len(v) for v in mixed_accas.values())} mixed"
        + (f" | {int(is_intl.sum())} national team matches, {sum(len(v) for v in intl_slips.values())} international accas"
           if is_intl.any() else "")
        + (f" | {int(is_cup.sum())} cup matches, {sum(len(v) for v in cup_slips.values())} cup accas" if is_cup.any() else ""), verbose)
    return out


# --------------------------------------------------------------------- excel
def tables(out: dict) -> dict[str, pd.DataFrame]:
    pred, s = out["predictions"], out["settings"]
    base = pd.DataFrame({
        "Date": pd.to_datetime(pred["date_local"]).dt.strftime("%a %d %b"),
        "Time": pred["time_local"],
        "League": pred["league"].map(league_label),
        "League type": league_type(pred["league"]).to_numpy(),
        "Home": pred["home"], "Away": pred["away"],
        "Home %": pred["p_h"], "Draw %": pred["p_d"], "Away %": pred["p_a"],
        "Pick": pred["pick"].map(NAMES), "Pick %": pred["top"], "Strength": pred["confidence"],
        "Odds 1": pred["odds_h"], "Odds X": pred["odds_d"], "Odds 2": pred["odds_a"],
        "Over 2.5 %": pred["p_over"], "BTTS %": pred["p_btts"],
    })
    # best price on the market for the pick (football-data.co.uk max odds)
    k = pred["pick"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    M = pred[["m_h", "m_d", "m_a"]].to_numpy(float) if "m_h" in pred else np.full((len(pred), 3), np.nan)
    base.insert(base.columns.get_loc("Odds 2") + 1, "Best odds (pick)", M[np.arange(len(M)), k])
    v = out["value"]
    base["Value bet"] = [NAMES.get(v.loc[i, "bet"], "") if i in v.index else "" for i in pred.index]
    base["Edge %"] = [v.loc[i, "edge"] if i in v.index else np.nan for i in pred.index]
    base["Stake"] = [v.loc[i, "stake"] if i in v.index else np.nan for i in pred.index]
    if "out_home" in pred:
        base["Missing (H-A)"] = [f"{int(h)}-{int(a)}" if pd.notna(h) else "" for h, a in zip(pred["miss_h"], pred["miss_a"])]
        base["Out (home)"] = pred["out_home"].values
        base["Out (away)"] = pred["out_away"].values
        dh, da = pred["doubt_home"].fillna(""), pred["doubt_away"].fillna("")
        base["Doubtful"] = [" | ".join(x for x in (f"home: {h}" if h else "", f"away: {a}" if a else "") if x)
                            for h, a in zip(dh, da)]
    base["Model"] = pred["model"].values
    base = base.reset_index(drop=True)

    def group(name):
        return base[base["Strength"] == name].sort_values("Pick %", ascending=False)

    vb = base[base["Value bet"] != ""].sort_values("Edge %", ascending=False)[
        ["Date", "Time", "League", "League type", "Home", "Away", "Value bet", "Edge %", "Stake", "Home %", "Draw %", "Away %",
         "Odds 1", "Odds X", "Odds 2"]]

    def slip_sheet(by_day):
        rows = []
        for day, day_slips in by_day.items():
            for n, sl in enumerate(day_slips, 1):
                rows.append({"Row": "SLIP", "Set": sl.get("set", ""), "Day": day, "Slip": n, "Match": f"{len(sl['legs'])} legs",
                             "League": "", "League type": "", "Pick": "", "Pick %": sl["win_prob"], "Odds": sl["odds"],
                             "Price": "has estimates" if sl.get("est") else ""})
                for i, pick, prob, odds in sl["legs"]:
                    r = pred.loc[i]
                    rows.append({"Row": "", "Day": "", "Slip": n,
                                 "Match": f"{r['home']} v {r['away']}  ({r['time_local']})",
                                 "League": league_label(r["league"]), "League type": _league_type(r["league"]),
                                 "Pick": betting.pick_name(pick), "Pick %": prob,
                                 "Odds": odds, "Price": "estimate" if i in sl.get("est", ()) else ""})
        t = pd.DataFrame(rows, columns=["Row", "Set", "Day", "Slip", "Match", "League", "League type", "Pick", "Pick %", "Odds", "Price"])
        return t if t["Set"].fillna("").astype(bool).any() else t.drop(columns="Set")

    markets = props.market_tables(pred, league_label)
    picks = base.assign(**{"Pick odds": np.where(pred["pick"] == "H", pred["odds_h"], np.where(pred["pick"] == "D", pred["odds_d"], pred["odds_a"]))})
    goals = base.assign(**{"Under 2.5 %": 1 - base["Over 2.5 %"], "Over odds": pred["o_over"].to_numpy(), "Under odds": pred["o_under"].to_numpy(),
                           "Bookmaker over %": pred["mk_over"].to_numpy()})
    btts = markets["BTTS"].assign(**{"BTTS yes %": lambda x: x["Yes %"]})
    n = int(s.get("top_n_per_league", 3))
    nat = base[base["League type"] == "International"].sort_values("Pick %", ascending=False)
    cup = base[base["League type"] == "Cup"].sort_values("Pick %", ascending=False)
    by_market = out.get("market_accas", {})
    days = sorted({d for v in by_market.values() for d in v})
    market_slips = {d: [sl for v in by_market.values() for sl in v.get(d, [])] for d in days}
    # My picks also shows the banger acca of every other market (the same slips as the Banger accas sheet,
    # so they are not counted twice in the tracker): 1X2 accas first, then Over 2.5, Over 1.5, corners, cards, shots
    extra = [t for t in bangers.MARKETS if t != "1X2"] if s.get("my_picks_markets", True) else []
    mine = {d: list(out["my_picks"].get(d, [])) + [sl for t in extra for sl in by_market.get(t, {}).get(d, [])]
            for d in sorted(set(out["my_picks"]) | {d for t in extra for d in by_market.get(t, {})})}
    sheets = {"My picks": slip_sheet({d: v for d, v in mine.items() if v}),
              **bangers.tables(pred, s, league_label, _league_type, betting.pick_name),
              bangers.ACCA_TAB: slip_sheet(market_slips),
              bangers.MIXED_TAB: slip_sheet(out.get("mixed_accas", {})),
              "All predictions": base, "Bangers": group("BANGER"), "Strong": group("STRONG"),
              "Weak": group("WEAK"), "Cups": cup, "Internationals": nat, "Value bets": vb,
              "Main acca": slip_sheet(out["slips"]), "Strong acca": slip_sheet(out["strong_slips"]),
              "Banger acca": slip_sheet(out["banger_slips"]), "Other acca": slip_sheet(out["other_slips"]),
              "International acca": slip_sheet(out.get("intl_slips", {})),
              "Cup acca": slip_sheet(out.get("cup_slips", {})),
              "Top favourites": props.top_per_league(picks, "Pick %", n, ["Pick", "Pick %", "Strength", "Pick odds"]),
              "Top goals": props.top_per_league(goals, "Over 2.5 %", n, ["Over 2.5 %", "Bookmaker over %", "Over odds", "BTTS %"]),
              "Top BTTS": props.top_per_league(btts, "BTTS yes %", n, ["BTTS yes %", "Yes odds", "No odds"]),
              "Top corners": props.top_per_league(markets["Corners"], "Corners expected", n, ["Corners expected", "Corners line", "Corners over %", "Corners over odds"]),
              **markets}
    if "out_home" in pred:
        inj = base[(base["Out (home)"] != "") | (base["Out (away)"] != "")][
            ["Date", "League", "League type", "Home", "Away", "Out (home)", "Out (away)", "Doubtful"]]
        sheets["Injuries"] = inj
    return sheets


NOTES = [
    "PREDICT/90 - how to read this file",
    "",
    "PREDICTIONS (sheets All predictions, Bangers, Strong, Weak)",
    "Home % / Draw % / Away %: the model's chance for each result. The three always add up to 100%.",
    "Pick: the most likely result. Strength: BANGER = pick is 75%+, STRONG = 65-75%, GOOD = 50-65%, WEAK = below 50%. In the test they were right 81%, 70%, 57% and 42% of the time.",
    "League type: Main = the 22 main leagues, Other = the 16 other leagues, Cup = the cups, International = national teams. Every sheet has filter arrows at the top: filter by Strength, League type or anything else.",
    "All predictions has every match: what is not in a list or an acca is still there.",
    "Odds 1 / X / 2: average bookmaker odds (football-data.co.uk) or Bet365 (API-Football).",
    "Best odds (pick): the highest price on the market for the pick. Compare it with what your bookmaker pays.",
    "The price you take matters a lot for accas: see part 7 of the notebook.",
    "Over 2.5 % and BTTS %: chance of 3+ goals, and of both teams scoring.",
    "Model: 'with odds' uses the bookmaker market as one of its inputs. 'no odds' = odds were not available yet.",
    "",
    "VALUE BETS",
    "A value bet is a result where our chance x the odds is at least 1 + the minimum edge (5% by default).",
    "Edge % = how much better than fair the odds are, if our chance is right. Stake = quarter-Kelly stake on your bankroll (settings.json), max 3%.",
    "",
    "MY PICKS",
    "My own recommendation, the most disciplined way to use the model. Each day, only when the model has them: 1 long acca around 5 odds (banger, or strong if there is no banger) and up to 2 safer accas around 2.3 odds (favourites 70%+). No match is used twice.",
    "Same small stake on every slip, at most 1% of your bankroll. I would skip weak picks, the main acca, other leagues in accas, and the other acca until your tracker shows it works.",
    "This exact rule in the 9 season test: 949 slips on about 1 day in 3, 28.0% of slips won, +1.7% at average odds, -0.3% at Bet365, range -9.8% to +13.5%. So about break-even: not a sure profit. Expect losing runs: up to 18 long slips or 10 safer slips in a row.",
    "National teams: competitive matches (no friendlies) can be in My picks, the strong and the banger acca too. Their accuracy is tested (79% for 65%+ picks), their profit is not: no old odds for national teams.",
    "Other markets: under the 1X2 accas, My picks also shows one acca per market from the bangers (Over 2.5, Over 1.5, corners, cards, shots), total odds about 2 to 3. They are the same slips as the Banger accas sheet. In the test all legs won on 30-37% of slips. Corners, cards and shots odds are mostly estimates: check the real price (my_picks_markets in settings.json).",
    "",
    "BANGERS (the safest pick of each kind)",
    "1X2, Over 2.5, Over 1.5, Corners over, Cards over, Shots over: only picks that are 75%+ (bangers_min_prob in settings.json).",
    "Over 2.5 uses half our chance, half the bookmaker's. A match in Over 2.5 is not shown again in Over 1.5.",
    "Corners, cards and shots: every match once, at its highest line that is still 75%+ (never below 6.5 corners, 2.5 cards, 15.5 shots).",
    "Test on 9 seasons, each predicted only with the seasons before it: right 1X2 81%, Over 2.5 81%, Over 1.5 80%, corners 79%, cards 82%, shots 78%. Lowest season 73-79%. National teams: 1X2 86%, Over 1.5 81% (competitive matches only, no Over 2.5).",
    "No BTTS bangers: in 100,000 test matches the model never reached 75% for BTTS (highest 71%).",
    "Cups: domestic cups in 1X2 (82% in the test), Over 2.5 and Over 1.5. European cups only in Over 1.5: their 75%+ 1X2 picks were right only 71% in the test.",
    "BANGER ACCAS: one acca per market and day, from that market's bangers only, total odds about 2 to 3 (bangers_acca_min_odds / max_odds). One market per slip, so the same match is never twice in a slip.",
    "In the test all legs won in 42% (1X2), 33% (Over 1.5), 30% (corners), 37% (cards), 36% (shots) of slips. At real old odds the 1X2 acca lost about 8% per slip (the bookmaker margin); the others have no old odds, so profit is not tested. Part 5c of the notebook has the tables.",
    "Odds = Bet365 when API-Football has that line, otherwise an estimate (our chance + a normal margin): check the price at your bookmaker.",
    "The page shows My picks, Bangers, Mixed acca and All predictions first. Press More for everything else (simple_page: false shows all at once). The Excel file always has every sheet.",
    "",
    "MIXED ACCA",
    "One acca a day from the bangers of every kind: the top 2 of 1X2, Over 2.5, Over 1.5, corners, cards and shots (mixed_acca_per_market in settings.json: 1 = shorter and safer). Safest first. Every match only once: a match that is top in two kinds goes where it is safest, and the other kind takes its next banger.",
    "Test on 9 seasons (clubs): 2,075 days, on average 7.6 legs at about 3.8 total odds. Every leg won on 22% of days (the model said 22%), on another 32% only one leg lost. Top 1 of each kind: 4.4 legs, about 2.1 odds, won 38%. Most corners, cards and shots prices are estimates, so profit is not tested."
    "",
    "ONE SLIP PER MATCH",
    "A match goes into one acca only, across all acca sheets (one_slip_per_match in settings.json). So one bad result, like a favourite that draws, can only hurt one slip. The order: My picks first, then Mixed acca, Banger accas, Cup acca, International acca, Strong, Banger, Main, Other: each takes the best matches that are still free.",
    "The tests of each acca were made with each sheet on its own, so later sheets can now have other legs or fewer slips. Set it to false to let every sheet pick freely.",
    "",
    "THE 4 ACCA SHEETS",
    "All from the 22 main leagues by default (acca_leagues in settings.json), total odds about 4.5-7. Within one sheet every match is used once. A match can appear on more than one sheet. Part 7 of the notebook has the tests.",
    "MAIN ACCA: up to 5 a day. Win, double chance and over/under 2.5, legs 50%+, at most 2 of one type. These have old odds, so their profit is tested: at average odds they lost in the test.",
    "STRONG ACCA: favourites to win only, every leg 65%+. Only 1 or 2 a day and only on some days. A small profit in the test, not proven.",
    "BANGER ACCA: favourites to win only, every leg 70%+, not 75%+: those legs almost never reach 5 odds. Very few slips, about 40 a season. A small profit in the test, not proven: acca numbers move a lot with tiny changes (part 7).",
    "OTHER ACCA: up to 5 a day. BTTS, over 1.5, under 3.5, favourite to score, underdog +1.5, favourite or over 2.5, corners, corner 1X2, cards, shots, at most 2 of one type.",
    "These have no old odds, so accuracy is tested but profit is not (part 5 of the notebook).",
    "Price = estimate means no real odds were found: only bets 60%+ sure, at our chance plus a normal bookmaker margin. Check the real price at your bookmaker before you bet: total odds can differ.",
    "Double chance odds are real when API-Football has them, otherwise worked out from the win odds. Cards are counted as yellow + red: check the rules at your bookmaker.",
    "On goals bets (over/under, BTTS) the slip's chance is half our model, half the market. So it can differ a little from the Over 2.5 % and BTTS % columns.",
    "Busy days give all 5. Quiet days give fewer or none, because there are not enough safe matches: slips are never forced.",
    "Each slip stops when total odds reach about 4.5-7. 'Pick %' on the SLIP row = chance that every leg wins.",
    "",
    "NATIONAL TEAMS (sheets Internationals and International acca)",
    "World Cup and Euro qualifiers, Nations League, AFCON, Copa America, Gold Cup, Asian Cup and friendlies, today and tomorrow. Needs the API-Football key, and they only come in the international windows.",
    "Their own model: ratings for every country plus recent goals (part 5b of the notebook). Test on 11 years, competitive matches: picks 65%+ right 79% of the time, 75%+ 86%.",
    "Friendlies are always WEAK: squads change a lot there. League type = International, so you can filter them.",
    "INTERNATIONAL ACCA: favourites to win only, competitive matches only, every leg 70%+ (intl_acca_min_prob in settings.json). Accuracy tested, profit not: no old odds for national teams. No value bets on national teams for the same reason.",
    "",
    "CUPS (sheets Cups and Cup acca)",
    "Champions League, Europa League, Conference League, FA Cup, Carabao Cup, Copa del Rey, DFB Pokal, Coppa Italia, Coupe de France, Scottish Cup and League Cup, KNVB Beker, Taca de Portugal. Needs the API-Football key.",
    "Their own model on the clubs' league form: strength, form, level (a small club below our leagues), country strength in European matches, neutral finals. Bets settle after 90 minutes, so the model learned the 90 minute score.",
    "Test on 2 seasons (2023/24 and 2024/25), each predicted only with the seasons before it, 2,134 matches: all cups 57% right, picks 65%+ 75%, 75%+ 79%. Domestic cups 75%+: 82%. European cups 75%+: 71%, so a European pick is never BANGER.",
    "Skipped, no blind guesses: European matches with a club from a country we have no league for, and small clubs whose name looks like a league club's.",
    "CUP ACCA: cup favourites to win only, every leg 70%+ (cup_acca_min_prob). Cup matches can also be in My picks, the strong and the banger acca (cups_in_picks). No value bets on cups: no old cup odds, so profit is not tested.",
    "",
    "TOP BY LEAGUE (sheets Top favourites, Top goals, Top BTTS, Top corners)",
    "The top matches of each league, 3 by default: top_n_per_league in settings.json. A top list is still only chances: check the Strength and the odds.",
    "",
    "THE BET SHEETS (Over-Under 2.5, BTTS, Goal lines, Team to score, Underdog +1.5, Corners, Cards, Shots)",
    "Chances for every match, also where no odds exist. Odds are real Bet365 odds when API-Football has them, otherwise empty. The chance here is the model's own.",
    "Expected = the number the model expects. Line = the bookmaker's main line when there is one, otherwise the half line nearest the expected number. Corners, cards and shots only exist for the 22 main leagues (no corners or shots for England's National League).",
    "Accuracy is tested in part 5 of the notebook. Profit can't be tested for bets with no old odds.",
    "1up and 2up are not here: the free data has no goal minutes, so they can't be predicted or tested.",
    "",
    "INJURIES",
    "From API-Football for today and tomorrow (free plan). Out = confirmed missing, Doubtful = questionable.",
    "Missing (H-A) = confirmed missing players, home-away. Each one moves the chance by only about 0.3 points:",
    "that is the size measured on 6,248 past matches, because the odds already price most injuries.",
    "",
    "IMPORTANT",
    "These are probabilities, not guarantees. Bet only what you can afford to lose.",
]


def export_excel(out: dict):
    """The Excel file and the HTML page, from the same tables."""
    sheets = tables(out)
    title = str(out["settings"].get("page_title") or "PrincePredict")
    notes = [NOTES[0].replace("PREDICT/90", title)] + NOTES[1:]
    path = OUTPUT / f"predictions_{date.today():%Y-%m-%d}.xlsx"
    try:
        report.write_predictions(path, sheets, notes)
    except PermissionError:
        path = OUTPUT / f"predictions_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
        report.write_predictions(path, sheets, notes)
    out["page"] = path.with_suffix(".html")
    webpage.write(out["page"], sheets, notes, simple=bool(out["settings"].get("simple_page", True)), title=title)
    return path


# ------------------------------------------------------------------ backtest
def backtest(verbose=True, goals=True, use_snapshot=False) -> pd.DataFrame:
    """use_snapshot=True: same saved data, same seeds -> the same numbers as in the delivery report.
    use_snapshot=False: newest data from the internet, and the snapshot is refreshed."""
    s = load_settings()
    if use_snapshot and SNAPSHOT.exists():
        results = pd.read_csv(SNAPSHOT, parse_dates=["date"], keep_default_na=True,
                              dtype={"time": str, "season": str}).fillna({"time": "", "season": ""})
        log(f"Using the saved data snapshot: {len(results):,} matches up to {results['date'].max():%d %b %Y}", verbose)
    else:
        log("Loading results...", verbose)
        results = data.load_results(s["leagues"], s["first_season"], s["refresh_hours"], verbose=verbose)
        SNAPSHOT.parent.mkdir(exist_ok=True)
        results.to_csv(SNAPSHOT, index=False, compression={"method": "gzip", "mtime": 0})
    log("Building features...", verbose)
    feat = build_features(results)
    log("Walk-forward backtest (each season trained only on the seasons before it):", verbose)
    pred = walk_forward(feat, s["test_from"], current_season_start(), goals=goals, verbose=verbose)
    pred.to_csv(BT_FILE, index=False, compression="gzip")
    log(f"Saved {len(pred):,} backtest predictions to {BT_FILE.name}", verbose)
    log("Walk-forward backtest of the more bets (corners, cards, shots, goal lines):", verbose)
    more = props.walk_forward(past_warmup(props.add_features(feat)), s["test_from"], current_season_start(), verbose)
    props.save_backtest(more, BT_PROPS_FILE)
    return pred


def load_more_bets() -> pd.DataFrame | None:
    return pd.read_csv(BT_PROPS_FILE, parse_dates=["date"]) if BT_PROPS_FILE.exists() else None


def load_backtest() -> pd.DataFrame | None:
    if not BT_FILE.exists():
        return None
    return pd.read_csv(BT_FILE, parse_dates=["date"])
