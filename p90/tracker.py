"""Bet tracker: every suggestion is logged, then settled automatically from the results.

output/bet_tracker.xlsx -> sheet "Bets": you may change the columns
"Played" (Y / N) and "Stake". Everything else is filled in by the system.
Sheet "Summary" shows the live record: all suggestions, and only what you played.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .betting import leg_won, pick_name
from .config import OUTPUT
from .report import write_table

FILE = OUTPUT / "bet_tracker.xlsx"
COLS = ["ID", "Added", "Match day", "Type", "Bet", "Odds", "Model chance", "Stake", "Played",
        "Status", "Return", "Profit", "Legs"]


def load() -> pd.DataFrame:
    if FILE.exists():
        try:
            df = pd.read_excel(FILE, sheet_name="Bets", dtype={"ID": str})
            for c in COLS:
                if c not in df:
                    df[c] = np.nan
            return df[COLS]
        except PermissionError:
            raise
        except Exception as e:
            print(f"  ! bet tracker could not be read ({e}); a fresh one will be started")
            backup = FILE.with_name(f"bet_tracker_backup_{datetime.now():%Y%m%d_%H%M}.xlsx")
            FILE.rename(backup)
    return pd.DataFrame(columns=COLS)


def _leg(r, pick, odds, prob) -> dict:
    return {"league": r["league"], "home": r["home"], "away": r["away"],
            "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), "pick": pick,
            "odds": round(float(odds), 2), "prob": round(float(prob), 4)}


def add(track: pd.DataFrame, entries: list[dict], today: str | None = None) -> pd.DataFrame:
    """Add new suggestions. Each day's slips of each set are logged once, from the first run of that
    day, and only for that day: a second run must not add more slips and inflate the record."""
    have = set(track["ID"].astype(str))
    is_slip = track["Type"].astype(str).str.contains(" slip | acca ")
    locked = set(zip(track.loc[is_slip, "Match day"].astype(str),
                     track.loc[is_slip, "Type"].astype(str).str.split().str[0]))
    new = []
    for e in entries:
        if e["ID"] in have:
            continue
        if " acca " in e["Type"] and ((e["Match day"], e["Type"].split()[0]) in locked or (today and e["Match day"] != today)):
            continue
        new.append(e)
    if not new:
        return track
    add_df = pd.DataFrame(new, columns=COLS)
    return add_df if track.empty else pd.concat([track, add_df], ignore_index=True)


def entries_from(preds: pd.DataFrame, value: pd.DataFrame, sets, s: dict) -> list[dict]:
    """sets = (("Main acca", slips by day), ("Strong acca", ...), ("Banger acca", ...), ("Other acca", ...))"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = []
    names = {"H": "Home", "D": "Draw", "A": "Away"}
    strong = preds[preds[["p_h", "p_d", "p_a"]].max(axis=1) >= s["strong_threshold"]]
    for i, r in strong.iterrows():
        k = int(np.argmax(r[["p_h", "p_d", "p_a"]].to_numpy(float)))
        pick = "HDA"[k]
        odds = r[["odds_h", "odds_d", "odds_a"]].iloc[k]
        if pd.isna(odds):
            continue                     # no odds yet: nothing to settle a bet on
        leg = _leg(r, pick, odds, r[["p_h", "p_d", "p_a"]].iloc[k])
        out.append({"ID": f"S|{leg['date']}|{r['league']}|{r['home']}|{r['away']}", "Added": now,
                    "Match day": leg["date"], "Type": "Strong pick",
                    "Bet": f"{r['home']} v {r['away']}: {names[pick]}", "Odds": leg["odds"],
                    "Model chance": leg["prob"], "Stake": 1.0, "Played": "", "Status": "pending",
                    "Legs": json.dumps([leg])})
    for i, v in value.iterrows():
        r = preds.loc[i]
        leg = _leg(r, v["bet"], v["odds"], v["prob"])
        out.append({"ID": f"V|{leg['date']}|{r['league']}|{r['home']}|{r['away']}|{v['bet']}", "Added": now,
                    "Match day": leg["date"], "Type": "Value bet",
                    "Bet": f"{r['home']} v {r['away']}: {names[v['bet']]}", "Odds": leg["odds"],
                    "Model chance": leg["prob"], "Stake": float(v["stake"]) or 1.0, "Played": "",
                    "Status": "pending", "Legs": json.dumps([leg])})
    for label, by_day in sets:
        for day, day_slips in by_day.items():
            for n, sl in enumerate(day_slips, 1):
                legs = [_leg(preds.loc[i], pick, odds, prob) for i, pick, prob, odds in sl["legs"]]
                marks = [" (price is an estimate)" if i in sl.get("est", ()) else "" for i, *_ in sl["legs"]]
                text = " | ".join(f"{l['home']} v {l['away']}: {pick_name(l['pick'])}{m}" for l, m in zip(legs, marks))
                key = "+".join(sorted(f"{l['league']}:{l['home']}:{l['pick']}" for l in legs))
                out.append({"ID": f"{label[0]}|{day}|{key}", "Added": now, "Match day": day,
                            "Type": f"{label} {n}", "Bet": text, "Odds": round(sl["odds"], 2),
                            "Model chance": round(sl["win_prob"], 4), "Stake": 1.0, "Played": "",
                            "Status": "pending", "Legs": json.dumps(legs)})
    return out


def settle(track: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if track.empty:
        return track
    recent = results[results["date"] >= results["date"].max() - pd.Timedelta(days=60)]
    def stats(r):
        get = lambda c: getattr(r, c, np.nan)
        return {"hc": get("hc"), "ac": get("ac"), "hk": get("hy") + get("hr"), "ak": get("ay") + get("ar"),
                "hs": get("hs"), "as_": get("as_")}

    lookup = {(r.league, r.home, r.away): (r.date, r.hg, r.ag, stats(r)) for r in recent.itertuples()}
    track = track.copy()
    for i, row in track[track["Status"] == "pending"].iterrows():
        try:
            legs = json.loads(row["Legs"])
        except Exception:
            continue
        outcome = "won"
        for l in legs:
            hit = lookup.get((l["league"], l["home"], l["away"]))
            if hit is None or abs((hit[0] - pd.Timestamp(l["date"])).days) > 4:
                outcome = "pending" if outcome != "lost" else "lost"
                continue
            won = leg_won(l["pick"], hit[1], hit[2], hit[3])
            if won is None:
                outcome = "pending" if outcome != "lost" else "lost"
            elif not won:
                outcome = "lost"
        if outcome == "pending" and pd.Timestamp(row["Match day"]) < pd.Timestamp.now() - pd.Timedelta(days=10):
            outcome = "void"          # postponed / not found
        track.at[i, "Status"] = outcome
        stake = float(row["Stake"]) if pd.notna(row["Stake"]) else 1.0
        if outcome == "won":
            track.at[i, "Return"] = round(stake * float(row["Odds"]), 2)
            track.at[i, "Profit"] = round(stake * (float(row["Odds"]) - 1), 2)
        elif outcome == "lost":
            track.at[i, "Return"], track.at[i, "Profit"] = 0.0, -stake
        elif outcome == "void":
            track.at[i, "Return"], track.at[i, "Profit"] = stake, 0.0
    return track


def summary(track: pd.DataFrame) -> pd.DataFrame:
    done = track[track["Status"].isin(["won", "lost"])].copy()
    rows = []
    if done.empty:
        return pd.DataFrame(columns=["Group", "Type", "Bets", "Won", "Hit rate", "Staked", "Profit", "ROI"])
    done["T"] = done["Type"].str.replace(r" \d+$", "", regex=True)
    done["Stake"] = pd.to_numeric(done["Stake"], errors="coerce").fillna(1.0)
    done["Profit"] = pd.to_numeric(done["Profit"], errors="coerce").fillna(0.0)
    for group, sub in (("All suggestions", done),
                       ("Only what I played", done[done["Played"].astype(str).str.upper().str.startswith("Y")])):
        for t, g in list(sub.groupby("T")) + [("TOTAL", sub)]:
            if g.empty:
                continue
            staked = g["Stake"].sum()
            rows.append({"Group": group, "Type": t, "Bets": len(g), "Won": int((g["Status"] == "won").sum()),
                         "Hit rate": (g["Status"] == "won").mean(), "Staked": round(staked, 2),
                         "Profit": round(g["Profit"].sum(), 2), "ROI": g["Profit"].sum() / staked if staked else 0})
    return pd.DataFrame(rows)


def save(track: pd.DataFrame):
    track = track.sort_values(["Match day", "Type"], kind="mergesort")
    write_table(FILE, {"Bets": track, "Summary": summary(track)})
