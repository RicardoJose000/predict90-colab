"""Upcoming matches: fixtures, odds, injuries and lineups from API-Football (free plan).

Free plan = 100 requests a day, and only yesterday / today / tomorrow.
Every answer is saved in data/api/ so running twice costs nothing extra.
A normal daily run uses about 6-40 requests.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import time
import unicodedata
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import API_CACHE, API_TO_CODE, LEAGUES, ROOT, api_key

API = "https://v3.football.api-sports.io/"
ALIAS_FILE = ROOT / "team_aliases.csv"
BET365 = 8


class ApiFootball:
    def __init__(self, key: str | None = None, verbose=True):
        self.key = key or api_key()
        self.verbose = verbose
        self.remaining = None
        self.used = 0
        self._last = 0.0
        self.limit_reached = False

    @property
    def ok(self) -> bool:
        return bool(self.key)

    def get(self, endpoint: str, params: dict, max_age_hours: float = 3.0) -> dict | None:
        name = endpoint.replace("/", "_") + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        path = API_CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]', '', name)}.json"
        if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_hours:
            return json.loads(path.read_text(encoding="utf-8"))
        if not self.ok or self.limit_reached:
            return None
        for attempt in range(2):
            wait = self._last + 6.5 - time.time()          # free plan: max 10 requests a minute
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(API + endpoint, params=params, headers={"x-apisports-key": self.key}, timeout=30)
                self._last = time.time()
                self.used += 1
                self.remaining = r.headers.get("x-ratelimit-requests-remaining", self.remaining)
                data = r.json()
            except Exception as e:
                if self.verbose:
                    print(f"  ! API-Football {endpoint}: {e}")
                return None
            errs = data.get("errors")
            if errs and "rateLimit" in str(errs) and "daily" in str(errs).lower():
                errs = {"requests": "daily limit"}
                break
            if errs and "rateLimit" in str(errs) and attempt == 0:
                time.sleep(61)
                continue
            break
        if errs:
            if isinstance(errs, dict) and "requests" in errs:
                self.limit_reached = True
                if self.verbose:
                    print("  ! API-Football daily limit reached: some odds and injuries are missing today")
                return None
            if self.verbose:
                msg = errs.get("requests") or errs.get("plan") or errs if isinstance(errs, dict) else errs
                print(f"  ! API-Football {endpoint}: {msg}")
            return None
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def status(self) -> dict | None:
        d = self.get("status", {}, max_age_hours=0)
        return d.get("response") if d else None


# ---------------------------------------------------------------- names
DROP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sv", "vfb", "vfl", "tsg", "rc", "cd", "ud", "sd",
        "club", "de", "fk", "sk", "if", "bk", "ff", "aik", "cfc", "fbc", "calcio", "us", "the", "and",
        "spvgg", "kv", "krc", "rsc", "nk", "hnk", "ca", "se", "ec", "cr", "ks", "gks", "cs",
        "1", "04", "05", "1899", "1846", "1860", "1909", "1910"}
SWAP = {"man": "manchester", "utd": "united", "nottm": "nottingham", "ath": "athletic", "st": "saint",
        "wolves": "wolverhampton", "spurs": "tottenham", "sheffield weds": "sheffield wednesday",
        "mgladbach": "monchengladbach", "ein": "eintracht", "psg": "paris saint germain",
        "inter": "internazionale", "sp": "sporting", "intl": "internacional"}


# names the fuzzy matcher can't guess (API-Football name -> football-data name)
MANUAL = {
    "Heart Of Midlothian": "Hearts", "Dagenham & Redbridge": "Dag and Red", "Sutton Utd": "Sutton",
    "Elgin City": "Elgin", "Ayr Utd": "Ayr", "Queen of the South": "Queen of Sth",
    "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Ath Madrid", "Paris Saint Germain": "Paris SG",
    "Borussia Monchengladbach": "M'gladbach", "Borussia Mönchengladbach": "M'gladbach",
    "Inter": "Inter", "AC Milan": "Milan", "Sporting CP": "Sp Lisbon", "Wolves": "Wolves",
    "Nottingham Forest": "Nott'm Forest", "Manchester United": "Man United", "Manchester City": "Man City",
    "Sheffield Wednesday": "Sheffield Weds", "Sheffield Utd": "Sheffield United",
    "Union St. Gilloise": "St. Gilloise", "Bayern München": "Bayern Munich", "Bayern Munich": "Bayern Munich",
    "1. FC Köln": "FC Koln", "1.FC Köln": "FC Koln", "Eintracht Frankfurt": "Ein Frankfurt",
    "Stade Brestois 29": "Brest", "OH Leuven": "Oud-Heverlee Leuven", "Vitória SC": "Guimaraes",
    "Vitoria SC": "Guimaraes", "Başakşehir": "Buyuksehyr", "Basaksehir": "Buyuksehyr",
    "Atletico Paranaense": "Athletico-PR", "Tianjin Teda": "Tianjin Jinmen Tiger",
    "Henan Jianye": "Henan Songshan Longmen", "Shandong Luneng": "Shandong Taishan",
    "Hangzhou Greentown": "Zhejiang Professional", "Qingdao Jonoon": "Qingdao Hainiu",
    "Qingdao Youth Island": "Qingdao West Coast", "Chengdu Better City": "Chengdu Rongcheng",
    "EIF": "Ekenas", "Tokyo Verdy": "Verdy", "Universitatea Craiova": "Univ. Craiova",
    "Politehnica Iasi": "Poli Iasi", "Universitatea Cluj": "U. Cluj", "Dynamo": "Dynamo Moscow",
    "Nizhny Novgorod": "Pari NN", "AIK Stockholm": "AIK", "WSG Wattens": "Tirol", "WSG Tirol": "Tirol",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("'", ""))
    words = [SWAP.get(w, w) for w in s.split()]
    words = [w for w in " ".join(words).split() if w not in DROP]
    return " ".join(words)


def _score(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    prefix = 0.85 if (na.startswith(nb) or nb.startswith(na)) and min(len(na), len(nb)) >= 4 else 0
    return max(r, 0.9 * overlap, prefix)


def load_aliases() -> dict:
    out = {}
    if ALIAS_FILE.exists():
        with ALIAS_FILE.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(row["league"], row["api_name"])] = row["our_name"]
    return out


def save_aliases(al: dict) -> None:
    with ALIAS_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["league", "api_name", "our_name"])
        for (lg, api), ours in sorted(al.items()):
            w.writerow([lg, api, ours])


class TeamMatcher:
    """Maps API-Football team names to football-data names, per league."""

    def __init__(self, results: pd.DataFrame):
        recent = results[results["date"] >= results["date"].max() - pd.Timedelta(days=500)]
        self.teams = {lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in recent.groupby("league")}
        self.aliases = load_aliases()
        self.new = 0
        self.unmatched: list = []

    def match(self, league: str, api_name: str) -> str | None:
        from .data import fix_name
        name = self._match(league, api_name)
        return fix_name(LEAGUES[league][0], name) if name else None

    def _match(self, league: str, api_name: str) -> str | None:
        if (league, api_name) in self.aliases:
            return self.aliases[(league, api_name)] or None
        cands = self.teams.get(league, [])
        if not cands:
            return None
        if api_name in MANUAL and MANUAL[api_name] in cands:
            return MANUAL[api_name]
        scored = sorted(((_score(api_name, c), c) for c in cands), reverse=True)
        best, name = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0
        if best >= 0.999 or (best >= 0.62 and best - second >= 0.04):
            self.aliases[(league, api_name)] = name
            self.new += 1
            return name
        self.unmatched.append((league, api_name, name, round(best, 2)))
        return None

    def save(self):
        if self.new:
            save_aliases(self.aliases)


# ---------------------------------------------------------------- times
def _uk_offset(d: datetime) -> int:
    """UK summer time: last Sunday of March to last Sunday of October."""
    def last_sunday(y, m):
        x = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
        return x - timedelta(days=(x.weekday() + 1) % 7)
    return 1 if last_sunday(d.year, 3) <= d.date() < last_sunday(d.year, 10) else 0


def uk_to_local(day: pd.Timestamp, hhmm: str, utc_offset: int) -> tuple[pd.Timestamp, str]:
    try:
        h, m = map(int, str(hhmm)[:5].split(":"))
    except Exception:
        return day, ""
    t = datetime(day.year, day.month, day.day, h, m)
    t = t - timedelta(hours=_uk_offset(t)) + timedelta(hours=utc_offset)
    return pd.Timestamp(t.date()), t.strftime("%H:%M")


# ------------------------------------------------------------- fixtures
def api_fixtures(api: ApiFootball, days: list[date], leagues: list[str], matcher: TeamMatcher,
                 utc_offset: int) -> pd.DataFrame:
    rows = []
    want = {LEAGUES[c][3]: c for c in leagues}
    for d in days:
        data = api.get("fixtures", {"date": d.isoformat()}, max_age_hours=2)
        if not data:
            continue
        for fx in data.get("response", []):
            lg = want.get(fx["league"]["id"])
            if lg is None or fx["fixture"]["status"]["short"] not in ("NS", "TBD"):
                continue
            h = matcher.match(lg, fx["teams"]["home"]["name"])
            a = matcher.match(lg, fx["teams"]["away"]["name"])
            if not h or not a or h == a:
                continue
            ko = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
            local = ko.replace(tzinfo=None) + timedelta(hours=utc_offset)
            rows.append({"league": lg, "home": h, "away": a,
                         "api_home": fx["teams"]["home"]["name"], "api_away": fx["teams"]["away"]["name"],
                         "date_local": pd.Timestamp(local.date()), "time_local": local.strftime("%H:%M"),
                         "fixture_id": fx["fixture"]["id"], "round": fx["league"].get("round", "")})
    matcher.save()
    return pd.DataFrame(rows)


def _parse_odds(r: dict) -> dict:
    o, more = {}, {}
    for bm in r.get("bookmakers", []):
        for bet in bm.get("bets", []):
            vals = {v["value"]: float(v["odd"]) for v in bet.get("values", []) if v.get("odd")}
            if bet["id"] == 1:
                o.update(b_h=vals.get("Home"), b_d=vals.get("Draw"), b_a=vals.get("Away"))
            elif bet["id"] == 5:
                o.update(b_over=vals.get("Over 2.5"), b_under=vals.get("Under 2.5"))
                more["goals"] = _half_lines(vals)
            elif bet["id"] == 8:
                o.update(b_btts_yes=vals.get("Yes"), b_btts_no=vals.get("No"))
            elif bet["id"] == 12:
                o.update(b_1x=vals.get("Home/Draw"), b_x2=vals.get("Draw/Away"))
            else:
                _parse_more(bet["id"], vals, more)
    if more:
        o["more"] = more
    return o


MORE_LINES = {45: "corners_total", 57: "corners_home", 58: "corners_away", 80: "cards_total", 82: "cards_home",
              83: "cards_away", 211: "shots_total"}


def _half_lines(vals: dict) -> dict:
    """{9.5: (over odds, under odds)}. Whole lines like 10 are skipped: they can end level and refund."""
    out = {}
    for k, over in vals.items():
        if k.startswith("Over "):
            line = float(k[5:])
            under = vals.get(f"Under {k[5:]}")
            if under and line % 1 == 0.5:
                out[line] = (over, under)
    return out


def _parse_more(bet_id: int, vals: dict, more: dict) -> None:
    if bet_id in MORE_LINES:
        more[MORE_LINES[bet_id]] = _half_lines(vals)
    elif bet_id == 27:                                 # clean sheet home: yes / no
        more["cs_home"] = (vals.get("Yes"), vals.get("No"))
    elif bet_id == 28:
        more["cs_away"] = (vals.get("Yes"), vals.get("No"))
    elif bet_id == 4:
        # API-Football writes the line from the home team's side, for both teams:
        # "Away -1.5" really is Away +1.5, the other side of "Home -1.5".
        # Checked on real odds: home favourite at 1.25, "Home -1.5" 1.68, "Away -1.5" 2.15.
        more["away_plus15"] = (vals.get("Away -1.5"), vals.get("Home -1.5"))     # (underdog away +1.5, other side)
        more["home_plus15"] = (vals.get("Home +1.5"), vals.get("Away +1.5"))     # (underdog home +1.5, other side)


def api_odds(api: ApiFootball, days: list[date], fixture_ids: list, per_match_budget: int = 30) -> dict:
    """Bet365 odds for the given fixture ids (most wanted first): 1X2, double chance, over/under 2.5, BTTS.

    Free plan: max 3 pages per date (30 matches), then one request per missing match.
    """
    out: dict = {}
    wanted = set(fixture_ids)
    for d in days:
        page, total = 1, 1
        while page <= min(total, 3):
            if wanted and wanted <= set(out):
                break
            data = api.get("odds", {"date": d.isoformat(), "bookmaker": BET365, "page": page}, max_age_hours=3)
            if not data:
                break
            total = data.get("paging", {}).get("total", 1)
            for r in data.get("response", []):
                out[r["fixture"]["id"]] = _parse_odds(r)
            page += 1
    missing = [f for f in fixture_ids if f not in out][:per_match_budget]
    for fid in missing:
        data = api.get("odds", {"fixture": int(fid), "bookmaker": BET365}, max_age_hours=3)
        if data and data.get("response"):
            out[fid] = _parse_odds(data["response"][0])
        elif data is not None:
            out[fid] = {}
    return out


def api_injuries(api: ApiFootball, days: list[date]) -> pd.DataFrame:
    rows = []
    for d in days:
        data = api.get("injuries", {"date": d.isoformat()}, max_age_hours=3)
        if not data:
            continue
        for r in data.get("response", []):
            rows.append({"fixture_id": r["fixture"]["id"], "team": r["team"]["name"],
                         "player": r["player"]["name"], "type": r["player"].get("type", ""),
                         "reason": r["player"].get("reason", "")})
    return pd.DataFrame(rows, columns=["fixture_id", "team", "player", "type", "reason"])


def lineups(api: ApiFootball, fixture_id: int) -> pd.DataFrame:
    """Starting XI, published about 20-40 minutes before kickoff (1 request per match)."""
    data = api.get("fixtures/lineups", {"fixture": int(fixture_id)}, max_age_hours=0.25)
    rows = []
    for t in (data or {}).get("response", []):
        for p in t.get("startXI", []):
            rows.append({"team": t["team"]["name"], "formation": t.get("formation"),
                         "player": p["player"]["name"], "pos": p["player"].get("pos")})
    return pd.DataFrame(rows)
