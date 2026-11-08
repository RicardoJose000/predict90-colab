"""Download and clean free results + odds data from football-data.co.uk.

Every league ends up in one table with the same columns:
league, country, tier, season, date, time, home, away, hg, ag, res,
hs, as_, hst, ast              shots / shots on target (main leagues only)
o_h, o_d, o_a                  average bookmaker odds 1X2
m_h, m_d, m_a                  best (max) odds 1X2
b_h, b_d, b_a                  Bet365 odds 1X2
o_over, o_under                average odds over / under 2.5 goals (main leagues only)
m_over, m_under, b_over, b_under   best and Bet365 odds over / under 2.5 (main leagues only)
"""
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import (CACHE, DATA, EXTRA_LEAGUES, LEAGUES, MAIN_LEAGUES, COUNTRY_TO_EXTRA,
                     current_season_start)

BASE = "https://www.football-data.co.uk"
MAIN_URL = BASE + "/mmz4281/{s}/{c}.csv"
EXTRA_URL = BASE + "/new/{c}.csv"
FIX_MAIN_URL = BASE + "/fixtures.csv"
FIX_EXTRA_URL = BASE + "/new_league_fixtures.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (football predictor)"}

# Same club, different spelling or a renamed club. Without this the club would
# restart as a brand-new team with no history.
TEAM_FIX = {
    "Argentina": {"Colon Santa FE": "Colon Santa Fe"},
    "Belgium": {"KV Mechelen": "Mechelen", "Mouscron-Peruwelz": "Mouscron", "Waasland-Beveren": "Beveren"},
    "China": {"Guangzhou R&F": "Guangzhou City", "Guangzhou Evergrande": "Guangzhou FC",
              "Shanghai SIPG": "Shanghai Port", "Dalian Yifang F.C.": "Dalian Pro",
              "Shandong Luneng": "Shandong Taishan", "Hangzhou Greentown": "Zhejiang Professional",
              "Zhejiang Greentown": "Zhejiang Professional", "Tianjin Quanjian": "Tianjin Tianhai"},
    "England": {"Sheffield Utd": "Sheffield United", "Oxford Utd": "Oxford"},
    "Greece": {"OFI": "OFI Crete"},
    "Norway": {"Ham-Kam": "HamKam"},
    "Poland": {"Gornik Z.": "Gornik Zabrze", "Ruch": "Ruch Chorzow"},
    "Romania": {"Din. Bucuresti": "Dinamo Bucuresti", "U Craiova": "Univ. Craiova",
                "Viitorul Constanta": "Farul Constanta"},
    "Russia": {"T. Moscow": "Torpedo Moscow"},
    "Spain": {"Leonesa": "Cultural Leonesa"},
    "Sweden": {"Osters": "Oster"},
    "Turkey": {"Erzurum BB": "Erzurumspor"},
}
AUTO_FIX_FILE = DATA / "team_fixes_auto.json"


def _auto_fixes() -> dict:
    if AUTO_FIX_FILE.exists():
        try:
            return json.loads(AUTO_FIX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def fix_name(country: str, name: str) -> str:
    return {**TEAM_FIX.get(country, {}), **_auto_fixes().get(country, {})}.get(name, name)


def _apply_fixes(df: pd.DataFrame) -> pd.DataFrame:
    auto = _auto_fixes()
    for country in set(TEAM_FIX) | set(auto):
        fix = {**TEAM_FIX.get(country, {}), **auto.get(country, {})}
        m = df["country"] == country
        if m.any():
            df.loc[m, "home"] = df.loc[m, "home"].replace(fix)
            df.loc[m, "away"] = df.loc[m, "away"].replace(fix)
    return df


def detect_new_spellings(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """No club can join a league in the middle of a season. So a name that first appears
    4+ weeks into a season is a new spelling of the club that stopped playing just before."""
    from .live import _score
    auto = _auto_fixes()
    changed = False
    cut = df["date"].max() - pd.Timedelta(days=120)
    for (lg, season), g in df[df["date"] >= cut].groupby(["league", "season"]):
        start = df[(df["league"] == lg) & (df["season"] == season)]["date"].min()
        long = pd.concat([g[["home", "date"]].rename(columns={"home": "t"}),
                          g[["away", "date"]].rename(columns={"away": "t"})])
        st = long.groupby("t")["date"].agg(["min", "max", "size"])
        for new, r in st[st["min"] >= start + pd.Timedelta(days=28)].iterrows():
            cands = st[(st["max"] < r["min"]) & (st["size"] >= 5)]
            if cands.empty:
                continue
            best = max(cands.index, key=lambda c: _score(new, c))
            if _score(new, best) >= 0.6:
                country = g["country"].iat[0]
                auto.setdefault(country, {})[new] = best
                changed = True
                if verbose:
                    print(f"  new spelling found: {new} = {best} ({country})")
    if changed:
        AUTO_FIX_FILE.write_text(json.dumps(auto, indent=1), encoding="utf-8")
        df = _apply_fixes(df)
    return df


COLUMNS = ["league", "country", "tier", "season", "date", "time", "home", "away",
           "hg", "ag", "res", "hs", "as_", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
           "o_h", "o_d", "o_a", "m_h", "m_d", "m_a", "b_h", "b_d", "b_a",
           "o_over", "o_under", "m_over", "m_under", "b_over", "b_under"]


# ----------------------------------------------------------------- download
def fetch(url: str, path, max_age_hours: float | None, quiet=False) -> bytes | None:
    """Download with cache. max_age_hours=None means 'never refresh once cached'."""
    if path.exists():
        age = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age < max_age_hours:
            return path.read_bytes()
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if r.content[:200].lower().lstrip().startswith(b"<!doctype") or b"<html" in r.content[:200].lower():
                return None
            path.write_bytes(r.content)
            return r.content
        except Exception as e:
            if attempt == 2:
                if path.exists():
                    if not quiet:
                        print(f"  ! could not refresh {url} ({e}) - using saved copy")
                    return path.read_bytes()
                if not quiet:
                    print(f"  ! could not download {url} ({e})")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip", low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df


def _col(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index)


def _num_or_nan(s):
    return pd.to_numeric(s, errors="coerce")


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")


# ------------------------------------------------------------ normalisation
def normalise_main(df: pd.DataFrame, code: str, season: str) -> pd.DataFrame:
    country, _, tier, _ = MAIN_LEAGUES[code]
    home = df.get("HomeTeam", df.get("HT"))
    away = df.get("AwayTeam", df.get("AT"))
    if home is None or away is None:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame({
        "league": code, "country": country, "tier": tier, "season": season,
        "date": _dates(df["Date"]),
        "time": df["Time"].astype(str) if "Time" in df else "",
        "home": home.astype(str).str.strip(), "away": away.astype(str).str.strip(),
        "hg": _num_or_nan(df.get("FTHG", df.get("HG"))) if ("FTHG" in df or "HG" in df) else np.nan,
        "ag": _num_or_nan(df.get("FTAG", df.get("AG"))) if ("FTAG" in df or "AG" in df) else np.nan,
        "res": df["FTR"].astype(str).str.strip() if "FTR" in df else (df["Res"] if "Res" in df else np.nan),
        "hs": _col(df, "HS"), "as_": _col(df, "AS"), "hst": _col(df, "HST"), "ast": _col(df, "AST"),
        "hc": _col(df, "HC"), "ac": _col(df, "AC"), "hy": _col(df, "HY"), "ay": _col(df, "AY"),
        "hr": _col(df, "HR"), "ar": _col(df, "AR"),
        "o_h": _col(df, "AvgH", "BbAvH", "B365H", "PSH", "WHH"),
        "o_d": _col(df, "AvgD", "BbAvD", "B365D", "PSD", "WHD"),
        "o_a": _col(df, "AvgA", "BbAvA", "B365A", "PSA", "WHA"),
        "m_h": _col(df, "MaxH", "BbMxH", "PSH", "B365H"),
        "m_d": _col(df, "MaxD", "BbMxD", "PSD", "B365D"),
        "m_a": _col(df, "MaxA", "BbMxA", "PSA", "B365A"),
        "b_h": _col(df, "B365H"), "b_d": _col(df, "B365D"), "b_a": _col(df, "B365A"),
        "o_over": _col(df, "Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"),
        "o_under": _col(df, "Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"),
        "m_over": _col(df, "Max>2.5", "BbMx>2.5", "P>2.5", "B365>2.5"),
        "m_under": _col(df, "Max<2.5", "BbMx<2.5", "P<2.5", "B365<2.5"),
        "b_over": _col(df, "B365>2.5"), "b_under": _col(df, "B365<2.5"),
    })
    return out


def normalise_extra(df: pd.DataFrame, code: str, fixtures=False) -> pd.DataFrame:
    country, _, tier, _ = EXTRA_LEAGUES[code]
    c = "" if fixtures else "C"           # history file has closing odds (AvgCH ...)
    season = df["Season"].astype(str).str.strip() if "Season" in df else ""
    if isinstance(season, pd.Series):
        season = season.str.replace(r"^(\d{4})/(\d{2})(\d{2})$", r"\1/\3", regex=True)
    out = pd.DataFrame({
        "league": code, "country": country, "tier": tier, "season": season,
        "date": _dates(df["Date"]),
        "time": df["Time"].astype(str) if "Time" in df else "",
        "home": df["Home"].astype(str).str.strip(), "away": df["Away"].astype(str).str.strip(),
        "hg": _num_or_nan(df["HG"]) if "HG" in df else np.nan,
        "ag": _num_or_nan(df["AG"]) if "AG" in df else np.nan,
        "res": df["Res"].astype(str).str.strip() if "Res" in df else np.nan,
        "hs": np.nan, "as_": np.nan, "hst": np.nan, "ast": np.nan,
        "hc": np.nan, "ac": np.nan, "hy": np.nan, "ay": np.nan, "hr": np.nan, "ar": np.nan,
        "o_h": _col(df, f"Avg{c}H", f"PS{c}H", f"B365{c}H"),
        "o_d": _col(df, f"Avg{c}D", f"PS{c}D", f"B365{c}D"),
        "o_a": _col(df, f"Avg{c}A", f"PS{c}A", f"B365{c}A"),
        "m_h": _col(df, f"Max{c}H", f"PS{c}H"), "m_d": _col(df, f"Max{c}D", f"PS{c}D"),
        "m_a": _col(df, f"Max{c}A", f"PS{c}A"),
        "b_h": _col(df, f"B365{c}H"), "b_d": _col(df, f"B365{c}D"), "b_a": _col(df, f"B365{c}A"),
        "o_over": np.nan, "o_under": np.nan, "m_over": np.nan, "m_under": np.nan, "b_over": np.nan, "b_under": np.nan,
    })
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date"])
    df = df[(df["home"] != "") & (df["away"] != "") & (df["home"] != "nan") & (df["away"] != "nan")].copy()
    df = _apply_fixes(df)
    for col in ("o_h", "o_d", "o_a", "m_h", "m_d", "m_a", "b_h", "b_d", "b_a",
                "o_over", "o_under", "m_over", "m_under", "b_over", "b_under"):
        df.loc[~(df[col] > 1.0), col] = np.nan
    df["time"] = df["time"].replace({"nan": "", "None": ""}).fillna("")
    return df


# --------------------------------------------------------------------- load
def season_codes(first_year: int, last_year: int) -> list[str]:
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first_year, last_year + 1)]


def load_results(leagues: list[str], first_year: int = 2013, refresh_hours: float = 6,
                 verbose: bool = True) -> pd.DataFrame:
    """All finished matches for the chosen leagues."""
    cur = current_season_start()
    frames = []
    main = [c for c in leagues if c in MAIN_LEAGUES]
    extra = [c for c in leagues if c in EXTRA_LEAGUES]
    t0 = time.time()
    cur_code = season_codes(cur, cur)[0]
    jobs = []                                   # (code, season, url, path, max_age)
    for code in main:
        for s in season_codes(first_year, cur):
            jobs.append((code, s, MAIN_URL.format(s=s, c=code), CACHE / f"{code}_{s}.csv",
                         refresh_hours if s == cur_code else None))
    for code in extra:
        jobs.append((code, None, EXTRA_URL.format(c=code), CACHE / f"{code}_all.csv", refresh_hours))

    def get(job):
        code, s, url, path, age = job
        return job, fetch(url, path, age, quiet=(s is not None and s != cur_code))

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(get, jobs))
    for (code, s, *_), raw in got:
        if raw is None:
            continue
        df = read_csv_bytes(raw)
        if "Date" not in df:
            continue
        if s is None:
            d = normalise_extra(df, code)
            frames.append(d[d["date"] >= pd.Timestamp(first_year, 1, 1)])
        else:
            frames.append(normalise_main(df, code, f"20{s[:2]}/{s[2:]}"))
    if not frames:
        raise SystemExit("Could not download the football data. Check your internet connection and run again.")
    df = _clean(pd.concat(frames, ignore_index=True))
    df = df[df["res"].isin(["H", "D", "A"]) & df["hg"].notna() & df["ag"].notna()]
    df = detect_new_spellings(df, verbose)
    df = df.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    df = df.sort_values(["date", "league", "home"]).reset_index(drop=True)
    if verbose:
        print(f"  {len(df):,} finished matches, {df['league'].nunique()} leagues, "
              f"{df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y} "
              f"({time.time() - t0:.0f}s)")
    return df


def load_fixtures_fd(leagues: list[str], results: pd.DataFrame, days_ahead: int,
                     refresh_hours: float = 3) -> pd.DataFrame:
    """Upcoming matches published by football-data.co.uk (with odds)."""
    frames = []
    raw = fetch(FIX_MAIN_URL, CACHE / "fixtures.csv", refresh_hours)
    if raw is not None:
        fx = read_csv_bytes(raw)
        if "Div" in fx:
            for code, g in fx.groupby("Div"):
                if code in leagues and code in MAIN_LEAGUES:
                    frames.append(normalise_main(g, code, ""))
    raw = fetch(FIX_EXTRA_URL, CACHE / "new_league_fixtures.csv", refresh_hours)
    if raw is not None:
        fx = read_csv_bytes(raw)
        if "Country" in fx:
            for country, g in fx.groupby("Country"):
                code = COUNTRY_TO_EXTRA.get(str(country).strip())
                if code in leagues:
                    frames.append(normalise_extra(g, code, fixtures=True))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    fx = _clean(pd.concat(frames, ignore_index=True))
    today = pd.Timestamp(datetime.now().date())
    fx = fx[(fx["date"] >= today) & (fx["date"] <= today + timedelta(days=days_ahead))]
    fx = drop_already_played(fx, results)
    fx["res"], fx["hg"], fx["ag"] = np.nan, np.nan, np.nan
    fx["source"] = "football-data"
    return fx.reset_index(drop=True)


def drop_already_played(fx: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if fx.empty:
        return fx
    recent = results[results["date"] >= fx["date"].min() - timedelta(days=4)]
    played = {(r.league, r.home, r.away) for r in recent.itertuples()}
    keep = [(l, h, a) not in played for l, h, a in zip(fx["league"], fx["home"], fx["away"])]
    return fx[keep]


def fill_season_labels(fx: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Give upcoming matches the season label of the league's latest season."""
    if fx.empty:
        return fx
    last = results.sort_values("date").groupby("league")["season"].last()
    fx = fx.copy()
    fx["season"] = fx["league"].map(last).fillna("")
    return fx
