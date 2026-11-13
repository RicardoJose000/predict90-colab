"""Paths, leagues and user settings."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# your own files (downloads, models, results, settings, key): the project folder, or on Colab a Google Drive folder
FILES = Path(os.environ["P90_FILES"]) if os.environ.get("P90_FILES") else ROOT
DATA = FILES / "data"
CACHE = DATA / "cache"
API_CACHE = DATA / "api"
MODELS = FILES / "models"
OUTPUT = FILES / "output"
for _p in (DATA, CACHE, API_CACHE, MODELS, OUTPUT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# code: (country, league name, tier, API-Football league id)
MAIN_LEAGUES = {
    "E0": ("England", "Premier League", 1, 39),
    "E1": ("England", "Championship", 2, 40),
    "E2": ("England", "League One", 3, 41),
    "E3": ("England", "League Two", 4, 42),
    "EC": ("England", "National League", 5, 43),
    "SC0": ("Scotland", "Premiership", 1, 179),
    "SC1": ("Scotland", "Championship", 2, 180),
    "SC2": ("Scotland", "League One", 3, 183),
    "SC3": ("Scotland", "League Two", 4, 184),
    "D1": ("Germany", "Bundesliga", 1, 78),
    "D2": ("Germany", "2. Bundesliga", 2, 79),
    "I1": ("Italy", "Serie A", 1, 135),
    "I2": ("Italy", "Serie B", 2, 136),
    "SP1": ("Spain", "La Liga", 1, 140),
    "SP2": ("Spain", "Segunda Division", 2, 141),
    "F1": ("France", "Ligue 1", 1, 61),
    "F2": ("France", "Ligue 2", 2, 62),
    "N1": ("Netherlands", "Eredivisie", 1, 88),
    "B1": ("Belgium", "Pro League", 1, 144),
    "P1": ("Portugal", "Primeira Liga", 1, 94),
    "T1": ("Turkey", "Super Lig", 1, 203),
    "G1": ("Greece", "Super League", 1, 197),
}

EXTRA_LEAGUES = {
    "ARG": ("Argentina", "Liga Profesional", 1, 128),
    "AUT": ("Austria", "Bundesliga", 1, 218),
    "BRA": ("Brazil", "Serie A", 1, 71),
    "CHN": ("China", "Super League", 1, 169),
    "DNK": ("Denmark", "Superliga", 1, 119),
    "FIN": ("Finland", "Veikkausliiga", 1, 244),
    "IRL": ("Ireland", "Premier Division", 1, 357),
    "JPN": ("Japan", "J1 League", 1, 98),
    "MEX": ("Mexico", "Liga MX", 1, 262),
    "NOR": ("Norway", "Eliteserien", 1, 103),
    "POL": ("Poland", "Ekstraklasa", 1, 106),
    "ROU": ("Romania", "Liga 1", 1, 283),
    "RUS": ("Russia", "Premier League", 1, 235),
    "SWE": ("Sweden", "Allsvenskan", 1, 113),
    "SWZ": ("Switzerland", "Super League", 1, 207),
    "USA": ("USA", "MLS", 1, 253),
}

LEAGUES = {**MAIN_LEAGUES, **EXTRA_LEAGUES}
COUNTRY_TO_EXTRA = {v[0]: k for k, v in EXTRA_LEAGUES.items()}
API_TO_CODE = {v[3]: k for k, v in LEAGUES.items()}


# national team competitions (p90/intl.py), not downloaded with the club leagues
INTL_LEAGUES = {
    "INT_WC": "World Cup", "INT_WCQ": "World Cup qualifiers", "INT_EURO": "Euro", "INT_EUROQ": "Euro qualifiers",
    "INT_UNL": "UEFA Nations League", "INT_AFCON": "Africa Cup of Nations", "INT_AFCONQ": "AFCON qualifiers",
    "INT_COPA": "Copa America", "INT_GOLD": "Gold Cup", "INT_CNL": "CONCACAF Nations League", "INT_ASIAN": "Asian Cup",
    "INT_ASIANQ": "Asian Cup qualifiers", "INT_FRIENDLY": "Friendlies", "INT_OTHER": "Other tournaments",
}


# cups (p90/cups.py): European cups and the big domestic cups
CUP_LEAGUES = {
    "CUP_UCL": "Champions League", "CUP_UEL": "Europa League", "CUP_UECL": "Conference League",
    "CUP_FA": "England FA Cup", "CUP_EFL": "England League Cup (Carabao)", "CUP_CDR": "Spain Copa del Rey",
    "CUP_DFB": "Germany DFB Pokal", "CUP_CI": "Italy Coppa Italia", "CUP_CDF": "France Coupe de France",
    "CUP_SCO": "Scottish Cup", "CUP_SLC": "Scottish League Cup", "CUP_KNVB": "Netherlands KNVB Beker",
    "CUP_TACA": "Portugal Taca de Portugal",
}


def league_label(code: str) -> str:
    if code in INTL_LEAGUES:
        return f"International - {INTL_LEAGUES[code]}"
    if code in CUP_LEAGUES:
        return f"Cup - {CUP_LEAGUES[code]}"
    c, n, *_ = LEAGUES[code]
    return f"{c} - {n}"


def league_type(code: str) -> str:
    return "Main" if code in MAIN_LEAGUES else "International" if code in INTL_LEAGUES else "Cup" if code in CUP_LEAGUES else "Other"


DEFAULTS = {
    "leagues": "all",
    "first_season": 2005,
    "test_from": 2018,
    "days_ahead": 3,
    "confident_threshold": 0.50,
    "strong_threshold": 0.65,
    "value_min_edge": 0.05,
    "value_min_prob": 0.25,
    "value_max_odds": 8.0,
    "bankroll": 10000,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.03,
    "acca_slips": 5,
    "acca_min_odds": 4.5,
    "acca_max_odds": 7.0,
    "acca_leg_min_prob": 0.50,
    "acca_leg_min_odds": 1.15,
    "acca_max_legs": 7,
    "acca_leagues": "main",
    "acca_markets": ["win", "double_chance", "over_under"],
    "acca_max_per_type": 2,
    "banger_threshold": 0.75,
    "top_n_per_league": 3,
    "strong_slips_min_prob": 0.65,
    "banger_slips_min_prob": 0.70,
    "other_slips": 5,
    "other_markets": ["btts", "goal_lines", "team_to_score", "handicap", "fav_or_over", "corners", "corner_1x2", "cards", "shots"],
    "injury_adjust": True,
    "injury_beta": 0.0095,
    "internationals": True,
    "intl_acca_min_prob": 0.70,
    "intl_in_picks": True,
    "bangers_min_prob": 0.75,
    "bangers_min_line": {"corners": 6.5, "cards": 2.5, "shots": 15.5},
    "bangers_acca_min_odds": 2.0,
    "bangers_acca_max_odds": 3.0,
    "simple_page": True,
    "mixed_acca_per_market": 2,
    "one_slip_per_match": True,
    "cups": True,
    "cups_in_picks": True,
    "cup_acca_min_prob": 0.70,
    "cups_odds_budget": 15,
    "use_api_football": True,
    "api_odds_budget": 30,
    "utc_offset": 1,
    "retrain_days": 7,
    "refresh_hours": 6,
}

SETTINGS_FILE = FILES / "settings.json"


def load_settings() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"settings.json could not be read ({e}), using defaults")
    if s["leagues"] == "all":
        s["leagues"] = list(LEAGUES)
    s["leagues"] = [c for c in s["leagues"] if c in LEAGUES]
    return s


def api_key() -> str | None:
    """API-Football key: environment variable, then .env file in the project folder."""
    k = os.environ.get("API_FOOTBALL_KEY")
    if k:
        return k.strip()
    env = FILES / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("API_FOOTBALL_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    return None


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1
