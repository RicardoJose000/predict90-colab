"""Insights: Sofascore-style facts for every match, worked out from our own data (no scraping).

For every club match on the page:
- Odds check: the price now and its implied chance (1 / odds), and how often this team really won at
  about the same price (implied chance within 5 points) in the last 5 seasons and the last 12 months,
  at this price or shorter, and at this price or longer. Too few games: a wider range, then all teams
  of the league at that price (marked), then "not enough games". The number of games is always shown.
- Last 10 matches of each team (all opponents): won, over 1.5 / 2.5, BTTS, scored, clean sheet,
  goal in the 1st half, won the 1st half, over 8.5 corners, over 3.5 / 4.5 cards (yellow + red).
- Goals by half over the same games: scored and conceded in the 1st and the 2nd half.
- Head to head: the last 10 meetings, the same stats, and who won.
- Injuries from API-Football.
- Hot facts: every stat that happened, or did not happen, in 8+ of the last 10 (5+ meetings for
  head to head), strongest first.
League matches only (the free data has no cup games). Corners, cards and half-time goals exist for
the 22 main leagues; the others show "-". National teams are left out: no match stats for them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INTL_LEAGUES, LEAGUES

TAB = "Insights"
N = 10                  # last 10 matches / meetings
HOT = 0.8               # 8 of 10
MIN_GAMES = 8           # a hot fact needs 8+ games with that stat
MIN_H2H = 5             # and 5+ meetings for head to head
BAND, WIDE = 0.05, 0.08  # "same odds" = implied chance within 5 (then 8) points
MIN_ODDS_GAMES = 5
SECTIONS = ("Hot facts", "Odds check", "Last 10 matches", "Goals by half", "Head to head", "Injuries")

# key, row label, hot label when it happened 8/10+, hot label when it did not happen 8/10+
STATS = [
    ("win", "Won", "Won", "No win"),
    ("o15", "Over 1.5 goals", "Over 1.5 goals", "Under 1.5 goals"),
    ("o25", "Over 2.5 goals", "Over 2.5 goals", "Under 2.5 goals"),
    ("btts", "Both teams scored", "Both teams scored", "Not both teams scored"),
    ("scored", "Scored", "Scored", "Did not score"),
    ("cs", "Clean sheet", "Clean sheet", "No clean sheet"),
    ("htg", "Goal in 1st half", "Goal in 1st half", "No goal in 1st half"),
    ("htw", "Won 1st half", "Won 1st half", None),
    ("c85", "Over 8.5 corners", "Over 8.5 corners", "Under 8.5 corners"),
    ("k35", "Over 3.5 cards", "Over 3.5 cards", "Under 3.5 cards"),
    ("k45", "Over 4.5 cards", "Over 4.5 cards", "Under 4.5 cards"),
]
# a hot fact that says the same as a stronger one is left out: over 2.5 -> no over 1.5, under 1.5 -> no under 2.5
IMPLIED = {("o15", True): ("o25", True), ("o25", False): ("o15", False),
           ("k35", True): ("k45", True), ("k45", False): ("k35", False)}


def _long(results: pd.DataFrame) -> pd.DataFrame:
    """Every finished match twice, once for each team, seen from that team."""
    r = results
    nan = pd.Series(np.nan, index=r.index)
    hth, hta = r.get("hthg", nan), r.get("htag", nan)
    corners = r.get("hc", nan) + r.get("ac", nan)
    cards = r.get("hy", nan) + r.get("ay", nan) + r.get("hr", nan) + r.get("ar", nan)
    odds_h = r["o_h"].fillna(r["b_h"]) if "b_h" in r else r["o_h"]
    odds_a = r["o_a"].fillna(r["b_a"]) if "b_a" in r else r["o_a"]
    common = {"date": r["date"], "league": r["league"], "country": r["country"], "corners": corners, "cards": cards}
    home = pd.DataFrame({**common, "team": r["home"], "opp": r["away"], "gf": r["hg"], "ga": r["ag"],
                         "htf": hth, "hta": hta, "odds": odds_h})
    away = pd.DataFrame({**common, "team": r["away"], "opp": r["home"], "gf": r["ag"], "ga": r["hg"],
                         "htf": hta, "hta": hth, "odds": odds_a})
    d = pd.concat([home, away], ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    gf, ga, hf, ha = (d[c].to_numpy(float) for c in ("gf", "ga", "htf", "hta"))
    ht = ~np.isnan(hf) & ~np.isnan(ha)
    c, k = d["corners"].to_numpy(float), d["cards"].to_numpy(float)

    def known(x, ok):
        return np.where(ok, x, np.nan)

    d["win"] = (gf > ga).astype(float)
    d["draw"] = (gf == ga).astype(float)
    d["o15"] = (gf + ga > 1.5).astype(float)
    d["o25"] = (gf + ga > 2.5).astype(float)
    d["btts"] = ((gf > 0) & (ga > 0)).astype(float)
    d["scored"] = (gf > 0).astype(float)
    d["cs"] = (ga == 0).astype(float)
    d["htg"] = known((hf + ha > 0).astype(float), ht)
    d["htw"] = known((hf > ha).astype(float), ht)
    d["c85"] = known((c > 8.5).astype(float), ~np.isnan(c))
    d["k35"] = known((k > 3.5).astype(float), ~np.isnan(k))
    d["k45"] = known((k > 4.5).astype(float), ~np.isnan(k))
    return d


class _Teams:
    def __init__(self, results: pd.DataFrame):
        self.d = _long(results)
        self.by = {t: g.index.to_numpy() for t, g in self.d.groupby("team", sort=False)}
        self.by_league = {lg: g.index.to_numpy() for lg, g in self.d.groupby("league", sort=False)}

    def rows(self, team: str, country: str | None, before) -> pd.DataFrame:
        idx = self.by.get(team)
        if idx is None:
            return self.d.iloc[0:0]
        g = self.d.loc[idx]
        if g["country"].nunique() > 1:          # the same name in two countries: the one of this match
            keep = country if country in set(g["country"]) else g["country"].iloc[-1]
            g = g[g["country"] == keep]
        return g[g["date"] < before]


def _ratio(x: pd.Series) -> tuple[int, int]:
    x = x.dropna()
    return int(x.sum()), int(len(x))


def _fmt(h: int, n: int) -> str:
    return f"{h}/{n}" if n else "-"


def _odds_lines(teams: _Teams, g: pd.DataFrame, name: str, odds: float, day) -> list[tuple[str, str]]:
    if not (odds and odds > 1):
        return [("Odds now", "no odds yet")]
    p = 1 / odds
    five = g[(g["date"] >= day - pd.DateOffset(years=5)) & g["odds"].notna()]
    year = five[five["date"] >= day - pd.DateOffset(years=1)]

    def band(df, w):
        q = 1 / df["odds"]
        return df[(q >= p - w) & (q <= p + w)]

    def rng(w):
        lo, hi = 1 / (p + w), (1 / (p - w) if p - w > 0.01 else None)
        return f"{lo:.2f}-{hi:.2f}" if hi else f"{lo:.2f}+"

    def pct(df):
        w, n = int(df["win"].sum()), len(df)
        return f"{w / n:.0%} ({w} of {n})"

    w, same, who = BAND, band(five, BAND), ""
    if len(same) < MIN_ODDS_GAMES:
        w, same = WIDE, band(five, WIDE)
    if len(same) < MIN_ODDS_GAMES and len(g):
        lg = g["league"].iloc[-1]                 # all teams of its league at this price
        pool = teams.d.loc[teams.by_league.get(lg, [])]
        pool = pool[(pool["date"] >= day - pd.DateOffset(years=5)) & (pool["date"] < day) & pool["odds"].notna()]
        w, same, who = BAND, band(pool, BAND), " - all teams in its league"
    out = [("Odds now", f"{odds:.2f} = {p:.0%} implied")]
    if len(same) < MIN_ODDS_GAMES:
        out.append(("Summary", f"At {odds:.2f} the implied chance is {p:.0%}: not enough old games at this price"))
        return out
    real = same["win"].mean()
    out.append(("Summary", f"When the odds are {odds:.2f} the implied chance is {p:.0%}, but "
                           f"{'teams like ' + name if who else name} actually won {real:.0%} at these odds "
                           f"({int(same['win'].sum())} of {len(same)} games, odds {rng(w)}, last 5 seasons{who})"))
    out.append((f"Odds {rng(w)}, last 5 seasons", pct(same) + (" all teams" if who else "")))
    y = band(year, w)
    out.append((f"Odds {rng(w)}, last 12 months", pct(y) if len(y) else "no games"))
    low, high = five[five["odds"] <= odds + 1e-9], five[five["odds"] >= odds - 1e-9]
    out.append((f"Odds {odds:.2f} or lower", pct(low) if len(low) else "no games"))
    out.append((f"Odds {odds:.2f} or higher", pct(high) if len(high) else "no games"))
    return out


def _hot(stats: dict, min_n: int) -> list[tuple[str, int, int, float]]:
    """(label, times, games, rate) for every stat that happened or did not happen in 80%+ of the games."""
    found, out = {}, []
    for key, _, yes, no in STATS:
        h, n = stats.get(key, (0, 0))
        if n < min_n:
            continue
        if h / n >= HOT:
            found[(key, True)] = (yes, h, n)
        elif no and (n - h) / n >= HOT:
            found[(key, False)] = (no, n - h, n)
    for k, (label, h, n) in found.items():
        if IMPLIED.get(k) in found:
            continue
        out.append((label, h, n, h / n))
    return out


def table(pred: pd.DataFrame, results: pd.DataFrame, league_label, league_type) -> pd.DataFrame:
    cols = ["Date", "Time", "League", "League type", "Home", "Away", "Home logo", "Away logo",
            "Section", "Team", "Stat", "Value", "Hot"]
    if pred is None or pred.empty or results is None or results.empty:
        return pd.DataFrame(columns=cols)
    teams = _Teams(results)
    keys = [k for k, *_ in STATS]
    rows = []
    for _, m in pred[~pred["league"].isin(INTL_LEAGUES)].iterrows():
        day = pd.Timestamp(m["date_local"]).normalize()
        country = LEAGUES[m["league"]][0] if m["league"] in LEAGUES else None
        H, A = str(m["home"]), str(m["away"])
        gh, ga = teams.rows(H, country, day), teams.rows(A, country, day)
        if gh.empty and ga.empty:
            continue
        meta = {"Date": day.strftime("%a %d %b"), "Time": m.get("time_local", ""), "League": league_label(m["league"]),
                "League type": league_type(m["league"]).iloc[0], "Home": H, "Away": A,
                "Home logo": m.get("logo_h") if isinstance(m.get("logo_h"), str) else "",
                "Away logo": m.get("logo_a") if isinstance(m.get("logo_a"), str) else ""}
        out = []

        def add(section, team, stat, value, hot=""):
            out.append({**meta, "Section": section, "Team": team, "Stat": stat, "Value": value, "Hot": hot})

        last = {H: gh.tail(N), A: ga.tail(N)}
        st = {t: {k: _ratio(g[k]) for k in keys} for t, g in last.items()}
        h2h = gh[gh["opp"] == A].tail(N)
        sh = {k: _ratio(h2h[k]) for k in keys}

        # hot facts first: the strongest across both teams and the meetings
        hot = [(t, *x) for t in (H, A) for x in _hot(st[t], MIN_GAMES)]
        neutral = {k: v for k, v in sh.items() if k not in ("win", "scored", "cs", "htw")}   # the same for both teams
        hot += [("", "H2H: " + lab, h, n, r) for lab, h, n, r in _hot(neutral, MIN_H2H)]
        if len(h2h) >= MIN_H2H:
            for t, col in ((H, "win"), (A, None)):
                w = int(h2h["win"].sum()) if col else int(((h2h["win"] == 0) & (h2h["draw"] == 0)).sum())
                if w / len(h2h) >= HOT:
                    hot.append(("", f"H2H: {t} won", w, len(h2h), w / len(h2h)))
        for t, lab, h, n, r in sorted(hot, key=lambda x: (-x[4], -x[3])):
            add("Hot facts", t, lab, f"{h}/{n}", "HOT")

        for t, g in ((H, gh), (A, ga)):
            odds = m.get("odds_h") if t == H else m.get("odds_a")
            for stat, val in _odds_lines(teams, g, t, float(odds) if pd.notna(odds) else 0.0, day):
                add("Odds check", t, stat, val)

        for key, label, *_ in STATS:
            if not (st[H][key][1] or st[A][key][1]):      # no data for either team (corners, cards, half time)
                continue
            for t in (H, A):
                h, n = st[t][key]
                add("Last 10 matches", t, label, _fmt(h, n), "HOT" if n >= MIN_GAMES and (h / n >= HOT or
                                                                                         (n - h) / n >= HOT) else "")

        for t, g in last.items():
            g = g.dropna(subset=["htf", "hta"])
            if len(g):
                f1, f2 = int(g["htf"].sum()), int((g["gf"] - g["htf"]).sum())
                c1, c2 = int(g["hta"].sum()), int((g["ga"] - g["hta"]).sum())
                tot = f1 + f2 + c1 + c2
                add("Goals by half", t, "Scored 1st / 2nd half", f"{f1} / {f2}")
                add("Goals by half", t, "Conceded 1st / 2nd half", f"{c1} / {c2}")
                add("Goals by half", t, "Goals in 2nd half", f"{(f2 + c2) / tot:.0%}" if tot else "-")
                add("Goals by half", t, "Games counted", str(len(g)))

        if len(h2h):
            hw, dr = int(h2h["win"].sum()), int(h2h["draw"].sum())
            add("Head to head", "", "Meetings", str(len(h2h)))
            add("Head to head", "", "Results", f"{H} {hw}, draw {dr}, {A} {len(h2h) - hw - dr}")
            for key, label, *_ in STATS[1:]:
                h, n = sh[key]
                if not n:
                    continue
                if key in ("scored", "cs", "htw"):        # seen from the home team: say so
                    label = f"{H}: {label.lower()}"
                add("Head to head", "", label, _fmt(h, n), "HOT" if n >= MIN_H2H and (h / n >= HOT or (n - h) / n >= HOT) else "")
        else:
            add("Head to head", "", "Meetings", "no meetings in our data")

        for side, t in (("home", H), ("away", A)):
            o, d = str(m.get(f"out_{side}") or ""), str(m.get(f"doubt_{side}") or "")
            if o and o != "nan":
                add("Injuries", t, "Out", o)
            if d and d != "nan":
                add("Injuries", t, "Doubtful", d)
        rows += out
    return pd.DataFrame(rows, columns=cols)
