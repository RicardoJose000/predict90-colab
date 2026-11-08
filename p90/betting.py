"""Value bets, Kelly stakes and the daily accumulator builder."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MAIN_LEAGUES
from .props import corner_1x2, p_over

OUTCOMES = ("H", "D", "A")
NAMES = {"H": "Home", "D": "Draw", "A": "Away"}
PICK_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win", "1X": "Home or draw", "X2": "Draw or away",
              "O": "Over 2.5", "U": "Under 2.5", "BY": "BTTS yes", "BN": "BTTS no",
              "O15": "Over 1.5", "U35": "Under 3.5", "HSC": "Home team to score", "ASC": "Away team to score",
              "H15": "Home +1.5", "A15": "Away +1.5", "FOH": "Home win or over 2.5", "FOA": "Away win or over 2.5",
              "CH": "More corners: home", "CA": "More corners: away"}
MARKET_OF = {"H": "win", "D": "win", "A": "win", "1X": "double_chance", "X2": "double_chance",
             "O": "over_under", "U": "over_under", "BY": "btts", "BN": "btts",
             "O15": "goal_lines", "U35": "goal_lines", "HSC": "team_to_score", "ASC": "team_to_score",
             "H15": "handicap", "A15": "handicap", "FOH": "fav_or_over", "FOA": "fav_or_over",
             "CH": "corner_1x2", "CA": "corner_1x2"}
STAT_NAMES = {"C": ("corners", "Corners", "hc", "ac"), "K": ("cards", "Cards", "hk", "ak"), "S": ("shots", "Shots", "hs", "as_")}
EST_MARGIN = 1.07     # estimated price where no real odds exist: our chance + a normal bookmaker margin
EST_MIN_PROB = 0.60   # with a made-up price, only bets the model is sure of (the level tested in part 5)
PRICES = {
    "avg": {"H": "o_h", "D": "o_d", "A": "o_a", "O": "o_over", "U": "o_under"},
    "bet365": {"H": "b_h", "D": "b_d", "A": "b_a", "O": "b_over", "U": "b_under"},
    "best": {"H": "m_h", "D": "m_d", "A": "m_a", "O": "m_over", "U": "m_under"},
    "live": {"H": "odds_h", "D": "odds_d", "A": "odds_a", "O": "o_over", "U": "o_under", "1X": "b_1x", "X2": "b_x2"},
}


def kelly_stake(p: float, odds: float, bankroll: float, fraction: float, cap_pct: float) -> float:
    """Fractional Kelly stake, capped at cap_pct of the bankroll."""
    if odds <= 1 or p <= 0:
        return 0.0
    f = (p * odds - 1) / (odds - 1)
    if f <= 0:
        return 0.0
    return round(min(f * fraction, cap_pct) * bankroll, 2)


def value_bets(df: pd.DataFrame, s: dict, odds_cols=("o_h", "o_d", "o_a")) -> pd.DataFrame:
    """Every outcome where model probability x odds beats 1 + min_edge."""
    rows = []
    P = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    O = df[list(odds_cols)].to_numpy(float)
    for j, out in enumerate(OUTCOMES):
        p, o = P[:, j], O[:, j]
        edge = p * o - 1
        ok = (edge >= s["value_min_edge"]) & (p >= s["value_min_prob"]) & (o <= s["value_max_odds"]) & (o > 1)
        for i in np.where(ok)[0]:
            rows.append({"idx": df.index[i], "bet": out, "prob": p[i], "odds": o[i],
                         "implied": 1 / o[i], "edge": edge[i],
                         "stake": kelly_stake(p[i], o[i], s["bankroll"], s["kelly_fraction"], s["max_stake_pct"])})
    vb = pd.DataFrame(rows, columns=["idx", "bet", "prob", "odds", "implied", "edge", "stake"])
    # one bet per match: keep the biggest edge
    vb = vb.sort_values("edge", ascending=False).drop_duplicates("idx").set_index("idx")
    return vb


def acca_pool(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """The matches allowed in slips: 'main' = the 22 main leagues, 'all' = all 38, or a list of codes."""
    which = s.get("acca_leagues", "all")
    if which == "all":
        return df
    codes = list(MAIN_LEAGUES) if which == "main" else list(which)
    return df[df["league"].isin(codes)]


def market_of(pick: str) -> str:
    return STAT_NAMES[pick[0]][0] if ":" in pick else MARKET_OF[pick]


def pick_name(pick: str) -> str:
    """C:T:O:9.5 -> Corners over 9.5, C:H:U:3.5 -> Home corners under 3.5"""
    if ":" not in pick:
        return PICK_NAMES[pick]
    kind, side, way, line = pick.split(":")
    name = STAT_NAMES[kind][1]
    name = {"T": name, "H": f"Home {name.lower()}", "A": f"Away {name.lower()}"}[side]
    return f"{name} {'over' if way == 'O' else 'under'} {line}"


def leg_won(pick: str, hg: float, ag: float, stats: dict | None = None) -> bool | None:
    """Did this bet win? stats holds corners, cards and shots. None = that number isn't known yet."""
    if pick in ("CH", "CA"):
        h, a = (stats or {}).get("hc", np.nan), (stats or {}).get("ac", np.nan)
        return None if h != h or a != a else bool(h > a) if pick == "CH" else bool(a > h)
    if ":" in pick:
        kind, side, way, line = pick.split(":")
        _, _, hx, ax = STAT_NAMES[kind]
        h, a = (stats or {}).get(hx, np.nan), (stats or {}).get(ax, np.nan)
        value = {"T": h + a, "H": h, "A": a}[side]
        if value != value:
            return None
        return bool(value > float(line)) if way == "O" else bool(value < float(line))
    return {"H": hg > ag, "D": hg == ag, "A": hg < ag, "1X": hg >= ag, "X2": hg <= ag,
            "O": hg + ag > 2.5, "U": hg + ag < 2.5, "BY": hg > 0 and ag > 0, "BN": hg == 0 or ag == 0,
            "O15": hg + ag > 1.5, "U35": hg + ag < 3.5, "HSC": hg > 0, "ASC": ag > 0,
            "H15": ag - hg < 2, "A15": hg - ag < 2,
            "FOH": hg > ag or hg + ag > 2.5, "FOA": ag > hg or hg + ag > 2.5}[pick]


def _no_margin(a, b):
    """The market's chance of side a, with the margin removed. None if a price is missing."""
    return (1 / a) / (1 / a + 1 / b) if a and b else None


def more_legs(df: pd.DataFrame, s: dict) -> list[dict]:
    """The bets for the other slips. With a real price: half our chance, half the market's.
    With no price: 60%+ only, our chance, and an estimated price marked est, to check at your bookmaker."""
    allowed = set(s["other_markets"])
    rows = []

    def add(i, pick, model, price=None, other=None):
        if market_of(pick) not in allowed or not model > 0:
            return
        if pd.isna(price):
            if model >= EST_MIN_PROB:
                rows.append({"idx": i, "pick": pick, "market": market_of(pick), "odds": 1 / (model * EST_MARGIN),
                             "prob": model, "est": True})
        else:
            market = None if pd.isna(other) else _no_margin(price, other)
            rows.append({"idx": i, "pick": pick, "market": market_of(pick), "odds": price,
                         "prob": model if market is None else (model + market) / 2, "est": False})

    for i, r in df.iterrows():
        m = r["more_odds"] if isinstance(r.get("more_odds"), dict) else {}
        goals = m.get("goals", {})
        add(i, "O15", r["p_over15"], *goals.get(1.5, (None, None)))
        over, under = goals.get(3.5, (None, None))
        add(i, "U35", 1 - r["p_over35"], under, over)
        add(i, "BY", r["p_btts"], r.get("b_btts_yes"), r.get("b_btts_no"))
        add(i, "BN", 1 - r["p_btts"], r.get("b_btts_no"), r.get("b_btts_yes"))
        if pd.notna(r["mk_h"]):                         # needs odds to know who the favourite is
            home_fav = r["mk_h"] >= r["mk_a"]
            yes, no = m.get("cs_away" if home_fav else "cs_home", (None, None))
            add(i, "HSC" if home_fav else "ASC", r["p_h_scores"] if home_fav else r["p_a_scores"], no, yes)
            price, other = m.get("away_plus15" if home_fav else "home_plus15", (None, None))
            add(i, "A15" if home_fav else "H15", 1 - (r["p_h_by2"] if home_fav else r["p_a_by2"]), price, other)
            add(i, "FOH" if home_fav else "FOA", r["p_fav_or_over"])
        for kind, (stat, _, _, _) in STAT_NAMES.items():
            known_h, known_a = pd.notna(r.get(f"{stat}_f_h")), pd.notna(r.get(f"{stat}_f_a"))
            if not (known_h and known_a):
                continue                                # no history of this number here: could never be settled
            for side, word in (("T", "total"), ("H", "home"), ("A", "away")):
                mu, alpha = r[f"mu_{stat}_{word}"], r[f"alpha_{stat}_{word}"]
                near = {math.floor(mu) + 0.5 + k: (None, None) for k in range(-3, 4) if math.floor(mu) + 0.5 + k > 0}
                lines = m.get(f"{stat}_{word}") or near
                for line, (over, under) in lines.items():
                    chance = float(p_over(mu, alpha, line))
                    add(i, f"{kind}:{side}:O:{line}", chance, over, under)
                    add(i, f"{kind}:{side}:U:{line}", 1 - chance, under, over)
            if stat == "corners":
                ch, _, ca = corner_1x2([r["mu_corners_home"]], r["alpha_corners_home"], [r["mu_corners_away"]], r["alpha_corners_away"])
                add(i, "CH", float(ch[0]))
                add(i, "CA", float(ca[0]))
    return rows


def build_other(day: pd.DataFrame, s: dict) -> list[dict]:
    """The other slips: bets with no old odds, so accuracy is tested but profit is not."""
    rows = more_legs(day, s) if not day.empty else []
    if not rows:
        return []
    t = pd.DataFrame(rows).set_index("idx").rename_axis(None)
    t = t[(t["prob"] >= s["acca_leg_min_prob"]) & (t["odds"] >= s["acca_leg_min_odds"])]
    t = t.assign(cost=-np.log(t["prob"]) / np.log(t["odds"])).sort_values(["cost", "prob"], ascending=[True, False])
    return slips_from_legs(t, {**s, "acca_markets": s["other_markets"], "acca_slips": s["other_slips"]})


def build_my_picks(day: pd.DataFrame, s: dict, price: str = "live") -> list[dict]:
    """What the test supports, if I had to choose. Each day, only when the model has them:
    1 long acca (banger, else strong) and up to 2 safer accas at about 2.3 odds. No match is used twice."""
    one = {**s, "acca_markets": ["win"], "acca_slips": 1}
    head = build_accas(day, {**one, "acca_leg_min_prob": s["banger_slips_min_prob"]}, price)
    name = "Long acca: banger"
    if not head:
        head = build_accas(day, {**one, "acca_leg_min_prob": s["strong_slips_min_prob"]}, price)
        name = "Long acca: strong"
    used = {i for sl in head for i, *_ in sl["legs"]}
    rest = day[~day.index.isin(used)]
    safer = build_accas(rest, {**s, "acca_markets": ["win"], "acca_leg_min_prob": s["banger_slips_min_prob"],
                               "acca_min_odds": 2.0, "acca_max_odds": 3.0, "acca_slips": 2}, price) if len(rest) else []
    return [{**sl, "set": name} for sl in head] + [{**sl, "set": "Safer acca"} for sl in safer]


def build_favourites(day: pd.DataFrame, s: dict, min_prob: float) -> list[dict]:
    """Strong and banger accas: favourites to win only, every leg min_prob or more."""
    return build_accas(day, {**s, "acca_markets": ["win"], "acca_leg_min_prob": min_prob}, "live")


def leg_table(df: pd.DataFrame, s: dict, price: str = "avg") -> pd.DataFrame:
    """Every bet allowed in a slip, best first. One match gives several rows.

    Double chance uses real odds when we have them, otherwise it is worked out from the win odds.
    Over/under uses half our chance, half the market's: there the market is a little better than the
    model, so trusting the model alone would pick the matches where it is most wrong.
    """
    cols = PRICES[price]

    def odds(pick):
        c = cols.get(pick)
        return pd.to_numeric(df[c], errors="coerce").to_numpy(float) if c in df else np.full(len(df), np.nan)

    h, d, a = (df[c].to_numpy(float) for c in ("p_h", "p_d", "p_a"))
    chance = {"H": h, "D": d, "A": a, "1X": h + d, "X2": d + a}
    def half_market(model, market):
        return np.where(np.isnan(market), model, (model + market) / 2)

    if "p_over" in df:
        market = df["mk_over"].to_numpy(float) if "mk_over" in df else np.full(len(df), np.nan)
        over = half_market(df["p_over"].to_numpy(float), market)
        chance.update(O=over, U=1 - over)
    price_of = {p: odds(p) for p in chance}
    for pick, x, y in (("1X", "H", "D"), ("X2", "D", "A")):
        worked_out = 1 / (1 / price_of[x] + 1 / price_of[y])
        price_of[pick] = np.where(np.isnan(price_of[pick]), worked_out, price_of[pick])

    allowed = set(s["acca_markets"])
    rows = [pd.DataFrame({"pick": p, "market": MARKET_OF[p], "prob": chance[p], "odds": price_of[p]}, index=df.index)
            for p in chance if MARKET_OF[p] in allowed]
    if not rows:
        return pd.DataFrame(columns=["pick", "market", "prob", "odds", "cost"])
    t = pd.concat(rows)
    t = t[(t["prob"] >= s["acca_leg_min_prob"]) & (t["odds"] >= s["acca_leg_min_odds"]) & (t["odds"] > 1)]
    t = t.assign(cost=-np.log(t["prob"]) / np.log(t["odds"]))
    return t.sort_values(["cost", "prob"], ascending=[True, False])


def slips_from_legs(t: pd.DataFrame, s: dict) -> list[dict]:
    """Build up to `acca_slips` slips from one day's bets (leg_table, best first).

    - a match is used in one slip only, so one bad result can't kill every slip
    - a slip holds at most acca_max_per_type bets of one type, so it is really mixed. With only one
      type allowed in acca_markets there is no limit.
    - a slip is finished once its total odds reach acca_min_odds (never above acca_max_odds)
    """
    if t.empty:
        return []
    mixed = len(set(s["acca_markets"])) > 1
    cap = s.get("acca_max_per_type", 2) if mixed else 99
    pool = list(t.itertuples())
    used: set = set()
    slips = []
    for _ in range(int(s["acca_slips"])):
        legs, total, count = [], 1.0, {}
        for leg in pool:
            if leg.Index in used or leg.Index in {l.Index for l in legs} or total * leg.odds > s["acca_max_odds"]:
                continue
            if count.get(leg.market, 0) >= cap:
                continue
            legs.append(leg)
            total *= leg.odds
            count[leg.market] = count.get(leg.market, 0) + 1
            if total >= s["acca_min_odds"] or len(legs) >= s["acca_max_legs"]:
                break
        if total < s["acca_min_odds"] or (mixed and len(count) < 2):
            break
        used.update(l.Index for l in legs)
        win = math.prod(l.prob for l in legs)
        slips.append({"legs": [(l.Index, l.pick, l.prob, l.odds) for l in legs],
                      "odds": total, "win_prob": win, "ev": win * total - 1,
                      "est": [l.Index for l in legs if getattr(l, "est", False)]})
    return slips


def build_accas(day: pd.DataFrame, s: dict, price: str = "avg") -> list[dict]:
    """The daily slips for one day's matches."""
    return slips_from_legs(leg_table(day, s, price), s) if not day.empty else []
