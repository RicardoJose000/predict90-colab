"""The results as one HTML page: works offline, on a PC or a phone. Same tables as the Excel file."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

INTRO = {"My picks": [
    "My own recommendation: 1 long acca around 5 odds and up to 2 safer accas around 2.3 odds, only when the model has them. No match is used twice. Same small stake on every slip, at most 1% of your bankroll.",
    "This exact rule in the 9 season test: 28.0% of slips won, +1.7% at average odds, -0.3% at Bet365, range -9.8% to +13.5%. So about break-even, not a sure profit. Expect losing runs: up to 18 long slips or 10 safer slips in a row.",
    "I would skip: weak picks, the main acca, other leagues in accas, and the other acca until your tracker shows it works.",
    "Other markets: below the 1X2 accas you get one acca per market from the bangers (Over 2.5, Over 8.5 corners, Over 3.5 cards), total odds about 2 to 3, the same slips as Bangers > Banger accas. In the test all legs won on 33-39% of slips. Corners odds are real Bet365 prices in most matches; cards are partly estimates: check those prices.",
    "One slip per match: a match goes into one acca only, across all acca sheets, so one bad result can only hurt one slip (one_slip_per_match in settings.json).",
    "National teams: competitive matches (no friendlies) can be in My picks, the strong and the banger acca too. Their accuracy is tested (79% for 65%+ picks), their profit is not: no old odds for national teams.",
    "Cups: cup matches can be in them too (cups_in_picks in settings.json). Accuracy tested (65%+ picks: 78% domestic, 69% European), profit not.",
], "Internationals": [
    "National teams, with their own model: ratings for every country plus recent goals. Only in the international windows, and only with the API-Football key.",
    "Test on 11 years (2016-2026), competitive matches: picks 65%+ were right 79% of the time, picks 75%+ 86%. When it says 70%, it happened about 70% of the time.",
    "Friendlies are always WEAK: squads change a lot there. No value bets here: the free data has no old odds for national teams, so profit is not tested.",
], "International acca": [
    "National teams, favourites to win only, competitive matches only (no friendlies), every leg 70%+. Accuracy is tested (79% for 65%+ picks), profit is not: no old odds for national teams.",
], "1X2": [
    "The likeliest result, only when it is 75%+. Test on 9 seasons (each season predicted only with the seasons before it): right 81% of the time, lowest season 78%. National teams (competitive matches only): 86%. Domestic cups: 82% (European cups are not here: 71% in the test).",
], "Over 2.5 goals": [
    "Half our chance, half the bookmaker's, only 70%+ (at 75% there is hardly ever one). Test on 9 seasons: right 73%, lowest season 70%. 75%+ picks were right 81%: the higher the %, the safer. No national teams here: their goal chances were too high in the test.",
], "Over 1.5 goals": [
    "75%+, only matches that are not already in Over 2.5. Test: right 80%, lowest season 79%. National teams: 81%. Cups: 81%.",
], "_corners": [
    "Over 8.5 corners, matches where it is 70%+. Test on 9 seasons: right 71%, lowest season 68%.",
    "Want it safer? Play the same match as over 7.5 corners: the Over 7.5 % and Odds 7.5 columns. Those same picks as over 7.5 were right 81% in the test.",
    "Odds = Bet365 (also its 3-way corners market, which has every line), otherwise an estimate: check the price at your bookmaker.",
], "_cards": [
    "Over 3.5 cards, matches where it is 75%+. Test on 9 seasons: right 80%, lowest season 77%. Cards = yellow + red: bookmakers count them in different ways, check the rules.",
    "Odds = Bet365 when that line has a price, otherwise worked out from Bet365's own cards line (marked estimate): check the price at your bookmaker.",
], "Corners over": [
    "Every match once, at its highest line that is still 75%+ (never below 6.5). Test: right 79%, lowest season 76%.",
    "Odds = Bet365 when that line has a price, otherwise an estimate: check the price at your bookmaker. Corners also use the Bet365 3-way corners market, which has every line, so most corners picks have a real price.",
], "Cards over": [
    "Every match once, at its highest line that is still 75%+ (never below 2.5). Test: right 82%, lowest season 79%. Cards = yellow + red: bookmakers count them in different ways, check the rules.",
], "Shots over": [
    "Every match once, at its highest line that is still 75%+ (never below 15.5). Test: right 78%, lowest season 76%.",
], "Underdog handicap": [
    "Every match with a big favourite (60%+): the underdog's smallest line that is still 80%+, the best price at that safety. Underdog +3.5 loses only if the favourite wins by 4 or more: the same bet as +4 on the 3-way handicap (4:0 on Sportybet).",
    "The +1.5 % to +5.5 % columns: the chance of every line, so you can pick your own. Good as a hedge against a big favourite. Biggest favourites first.",
    "Test: clubs 9 seasons, about 8 picks a day, right 88% (the model said 86%), lowest season 85%. National teams (competitive matches): right 87%. No friendlies. Odds = Bet365 Asian handicap when it has that line, otherwise an estimate: check your bookmaker. Profit not tested: no old handicap odds.",
], "Banger accas": [
    "One acca per market and day, only from that market's bangers, total odds about 2 to 3. One market per slip, so the same match is never twice in a slip.",
    "Test, share of slips where all legs won: 1X2 42%, Over 2.5 39%, over 8.5 corners 33%, over 3.5 cards 36%, shots 36% (the model said 45, 40, 36, 35, 33%). No Over 1.5 accas: the odds are too short. At real old odds the 1X2 acca lost about 8% per slip and the Over 2.5 acca about 11%: that is the bookmaker margin. The others have no old odds, so their profit is not tested. Keep stakes small.",
    "No BTTS bangers: in 100,000 test matches the model never reached 75% for BTTS (highest 71%).",
], "Mixed acca": [
    "One acca a day from the 4 main kinds: the top 2 of 1X2, Over 2.5, Over 8.5 corners and Over 3.5 cards (mixed_acca_per_market in settings.json, 1 = shorter and safer). Safest first, every match only once: a match that is top in two kinds goes where it is safest.",
    "Test on 9 seasons (clubs): 1,712 days, on average 4.6 legs at about 2.8 total odds. Every leg won on 35% of days (the model said 35%), and on another 39% only one leg lost. Top 1 of each kind: 3 legs, about 1.9 odds, won 47%.",
    "Corners: real Bet365 prices in most matches. Cards: partly estimates, so check the real total odds at your bookmaker. Legs paying under 1.15 are left out of the accas.",
], "Insights": [
    "Facts for every match from our own data (20 years of results and odds, no scraping): the odds check, the last 10 matches of each team, goals by half, head to head and injuries. Tap a part to open it.",
    "Odds check: the price now and its implied chance (1 / odds), and how often this team really won at about the same price. Always with the number of games: 45 games is a strong sign, 6 games a weak one.",
    "Hot facts: what happened, or did not happen, in 8+ of the last 10 (5+ meetings for head to head), strongest first. Press Hot facts only to see just those, the matches with the most first.",
    "League games only. Corners, cards and goals by half only for the 22 main leagues, the others show -. No national teams.",
], "All predictions": [
    "Every match of today and tomorrow: leagues, cups and national teams. Nothing is lost: what is not in a list or an acca is still here. Use the filter box to find a team or a league.",
], "Cups": [
    "Champions League, Europa League, Conference League, FA Cup, Carabao Cup, Copa del Rey, DFB Pokal, Coppa Italia, Coupe de France, Scottish Cup and League Cup, KNVB Beker, Taca de Portugal. Their own model, built on the clubs' league form: strength, form, level (a small club below our leagues), country strength in European matches, neutral finals. Bets settle after 90 minutes.",
    "Test on 2 seasons (2023/24 and 2024/25, each predicted only with the seasons before it, 2,134 matches): domestic cups, picks 65%+ right 78%, 75%+ 82%. European cups, 65%+ 69%, 75%+ 71%: so a European pick is never BANGER. When it says 70%, it happened about 70% of the time.",
    "Skipped: European matches with a club from a country we have no league for, and small clubs whose name looks like a league club's (no blind guesses). No value bets on cups: no old cup odds, so profit is not tested.",
], "Cup acca": [
    "Cup favourites to win only, every leg 70%+. In the test 70%+ cup picks were right 80% (domestic cups) and 75% (European cups). Accuracy is tested, profit is not: the free data has no old cup odds.",
]}
GROUPS = [
    ("My picks", ["My picks"]),
    ("Bangers", ["1X2", "Over 2.5 goals", "*corners", "*cards", "Underdog handicap", "Banger accas"]),     # *: Over 8.5 corners / Over 3.5 cards (bangers_lines)
    ("Insights", ["Insights"]),
    ("Mixed acca", ["Mixed acca"]),
    ("All predictions", ["All predictions"]),
    ("Predictions", ["Bangers", "Strong", "Weak", "Cups", "Internationals", "Value bets"]),
    ("More bangers", ["Over 1.5 goals", "Shots over"]),
    ("Accas", ["Main acca", "Strong acca", "Banger acca", "Other acca", "Cup acca", "International acca"]),
    ("Top by league", ["Top favourites", "Top goals", "Top BTTS", "Top corners"]),
    ("Goals", ["Over-Under 2.5", "BTTS", "Goal lines", "Team to score", "Underdog +1.5"]),
    ("Corners, cards, shots", ["Corners", "Cards", "Shots"]),
    ("Injuries", ["Injuries"]),
]
# the wide sheets: only what fits on a phone. The Excel file has every column.
HIDE = {"Home %", "Draw %", "Away %", "Best odds (pick)", "Edge %", "Stake", "Out (home)", "Out (away)", "Doubtful", "Model"}
SHEETS_WITH_ALL_COLUMNS = {"Value bets", "Injuries"}


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v if isinstance(v, (int, str)) else str(v)


CARD_META = ["Date", "Time", "League", "League type", "Match", "Home", "Away", "Home logo", "Away logo"]


def _cards(name: str, df: pd.DataFrame) -> dict:
    """Insights: one row per match for the filters, and its facts as short lists [section, side, stat, value, hot]
    (side 1 = home team, 2 = away team, 0 = both), so the page stays small."""
    rows, facts, sections = [], [], []
    for _, g in df.groupby(["Date", "Time", "League", "Home", "Away"], sort=False, dropna=False):
        f = g.iloc[0]
        rows.append([_clean(f[c]) if c != "Match" else f"{f['Home']} v {f['Away']}" for c in CARD_META])
        out = []
        for sec, team, stat, val, hot in g[["Section", "Team", "Stat", "Value", "Hot"]].itertuples(index=False):
            if sec not in sections:
                sections.append(sec)
            out.append([sections.index(sec), 1 if team == f["Home"] else 2 if team == f["Away"] else 0,
                        _clean(stat), _clean(val), 1 if hot else 0])
        facts.append(out)
    return {"name": name, "columns": CARD_META, "rows": rows, "facts": facts, "sections": sections}


def _sheet(name: str, df: pd.DataFrame) -> dict:
    if {"Section", "Stat", "Value"} <= set(df.columns):
        return _cards(name, df)
    d = df.copy()
    if "Row" not in d and name not in SHEETS_WITH_ALL_COLUMNS:
        d = d[[c for c in d.columns if c not in HIDE]]
    if "Home" in d and "Away" in d:
        d.insert(list(d.columns).index("Home"), "Match", d["Home"].astype(str) + " v " + d["Away"].astype(str))
        d = d.drop(columns=["Home", "Away"])
    return {"name": name, "columns": list(d.columns),
            "rows": [[_clean(v) for v in row] for row in d.astype(object).where(d.notna(), None).to_numpy().tolist()]}


MAIN_GROUPS = ("My picks", "Bangers", "Insights", "Mixed acca", "All predictions")    # the simple page: the rest is behind "More"


def _brand(title: str) -> str:
    """The name at the top of the page, second part in colour: PrincePredict -> Prince<b>Predict</b>, PREDICT/90 -> PREDICT<b>/90</b>."""
    t = title.strip() or "PrincePredict"
    m = re.match(r"^(.+?)(/.*)$", t) or re.match(r"^(.+?[a-z])([A-Z].*)$", t)
    return f"{html.escape(m[1])}<b>{html.escape(m[2])}</b>" if m else html.escape(t)


def _names(names: list[str], sheets) -> list[str]:
    """*corners = the corners bangers sheet, whatever its line: Over 8.5 corners (or Corners over without a fixed line)."""
    out = []
    for n in names:
        if n.startswith("*"):
            k = n[1:]
            out += [x for x in sheets if re.fullmatch(rf"Over [\d.]+ {k}|{k.capitalize()} over", x)]
        elif n in sheets:
            out.append(n)
    return out


def write(path, sheets: dict[str, pd.DataFrame], notes: list[str], simple: bool = True, title: str = "PrincePredict") -> None:
    intro = {k: v for k, v in INTRO.items() if not k.startswith("_")}
    for x in sheets:                                    # Over 8.5 corners / Over 3.5 cards: the texts of their line
        m = re.fullmatch(r"Over [\d.]+ (corners|cards)", x)
        if m:
            intro[x] = INTRO["_" + m[1]]
    data = {"updated": f"{datetime.now():%a %d %b %Y, %H:%M}", "notes": notes, "intro": intro,
            "groups": [{"name": g, "tabs": [_sheet(n, sheets[n]) for n in _names(names, sheets)]} for g, names in GROUPS]}
    data["groups"] = [g for g in data["groups"] if g["tabs"]]
    data["main"] = sum(g["name"] in MAIN_GROUPS for g in data["groups"]) if simple else len(data["groups"])
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = PAGE.replace("__TITLE__", html.escape(title.strip() or "PrincePredict")).replace("__BRAND__", _brand(title))
    path.write_text(page.replace("__DATA__", blob), encoding="utf-8")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#06080b;--panel:#10151c;--panel2:#161c25;--line:rgba(255,255,255,.1);--ink:#eef2ee;--ink2:#b7c0bb;--ink3:#8a948e;--lime:#c8f53c;--dim:rgba(200,245,60,.14);--gold:#f5c542;--mono:ui-monospace,Consolas,monospace;color-scheme:dark}
*{box-sizing:border-box;margin:0}
html{background:var(--bg);-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--ink);font:15px/1.45 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif;padding-bottom:40px}
button,select,input{font:inherit;color:inherit}
header{padding:18px 16px 6px;max-width:1320px;margin:auto;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline}
h1{font-size:21px;letter-spacing:.02em}h1 b{color:var(--lime)}
.stamp{color:var(--ink3);font-size:13px}
nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar{max-width:1320px;margin:auto;padding:8px 16px 0;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}
.bar::-webkit-scrollbar{display:none}
.bar button{flex:none;background:none;border:1px solid transparent;border-radius:999px;padding:6px 13px;cursor:pointer;color:var(--ink2);min-height:36px}
.bar button:hover{color:var(--ink)}
.bar button[aria-pressed=true]{background:var(--dim);color:var(--lime);border-color:rgba(200,245,60,.35)}
#tabs{padding-bottom:8px}
#tabs button{font-size:14px;padding:4px 11px;min-height:32px}
#tabs button[aria-pressed=true]{background:var(--lime);color:#000;border-color:var(--lime);font-weight:600}
main{max-width:1320px;margin:auto;padding:12px 16px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center}
.filters select,.filters input{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:7px 9px;min-height:38px;min-width:0;max-width:100%}
.filters input{flex:1 1 150px}
.filters select{flex:0 1 auto}
.count{color:var(--ink3);font-size:13px;margin-left:auto}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{position:sticky;top:0;background:var(--panel2);color:var(--ink2);text-align:left;font-weight:600;font-size:12.5px;padding:8px 10px;white-space:nowrap;cursor:pointer;user-select:none}
th.on::after{content:" \2193";color:var(--lime)}th.on.up::after{content:" \2191"}
td{padding:7px 10px;border-top:1px solid var(--line);white-space:nowrap}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td.match,th.match{position:sticky;left:0;background:var(--panel);min-width:150px;max-width:210px;white-space:normal;font-weight:500;border-right:1px solid var(--line);z-index:1}
th.match{background:var(--panel2);z-index:2}
td.hi{color:var(--lime);font-weight:600}
.tag{display:inline-block;font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:5px}
.BANGER{background:var(--lime);color:#000}.STRONG{background:var(--dim);color:var(--lime)}.GOOD{background:rgba(255,255,255,.1);color:var(--ink)}.WEAK{color:var(--ink3);border:1px solid var(--line)}
.slips{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:12px}
.slip{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.slip h3{display:flex;justify-content:space-between;gap:8px;align-items:baseline;padding:10px 12px;background:var(--panel2);font-size:14px;font-weight:600}
.slip h3 .odds{color:var(--lime);font:700 19px var(--mono)}
.slip h3 small{color:var(--ink3);font-weight:400;font-size:12.5px;display:block}
.leg{padding:8px 12px;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr auto;gap:0 10px}
.leg b{font-weight:600}.leg .m{color:var(--ink2);font-size:13.5px}.leg .l{color:var(--ink3);font-size:12px}
.leg .o{font-family:var(--mono);text-align:right}.leg .c{font-family:var(--mono);text-align:right;font-size:12.5px;color:var(--ink3)}
.est{color:var(--gold);font-size:11.5px}
.intro{max-width:820px;margin:0 0 12px;padding:10px 12px;border:1px solid var(--line);border-left:3px solid var(--lime);border-radius:9px;background:var(--panel);color:var(--ink2);font-size:14px}.intro p{margin:0 0 6px}.intro p:last-child{margin:0}
.empty{padding:26px;text-align:center;color:var(--ink3);border:1px dashed var(--line);border-radius:12px}
.notes{max-width:820px;color:var(--ink2)}.notes p{margin:0 0 7px}.notes h2{color:var(--ink);font-size:15px;margin:16px 0 6px}.notes h2:first-child{margin-top:0}
footer{max-width:1320px;margin:6px auto 0;padding:0 16px;color:var(--ink3);font-size:13px}
.hotbtn{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:7px 12px;min-height:38px;cursor:pointer;color:var(--ink2)}
.hotbtn[aria-pressed=true]{background:var(--lime);color:#000;border-color:var(--lime);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr));gap:12px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.ch{padding:10px 12px;background:var(--panel2)}
.teams{display:flex;align-items:center;gap:7px;font-weight:600;font-size:15px;flex-wrap:wrap}
.teams .vs{color:var(--ink3);font-weight:400}
.sub{display:flex;gap:8px;align-items:center;color:var(--ink3);font-size:12.5px;margin-top:3px}
.badge{margin-left:auto;background:var(--dim);color:var(--lime);font-size:11.5px;font-weight:700;padding:1px 8px;border-radius:999px;white-space:nowrap}
.logo{width:22px;height:22px;flex:none;object-fit:contain}
.ini{display:inline-grid;place-items:center;flex:none;width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,.1);font-size:10.5px;font-weight:700;color:var(--ink2)}
.ini.both{width:auto;padding:0 6px;border-radius:6px;font-size:10px}
.fr{display:grid;grid-template-columns:auto 1fr auto;gap:9px;align-items:center;padding:6px 12px;border-top:1px solid var(--line);font-size:14px}
.g3{display:grid;grid-template-columns:1fr minmax(54px,auto) minmax(54px,auto);gap:4px 12px;align-items:center;padding:5px 12px;border-top:1px solid var(--line);font-size:13.5px}
.g3.head{color:var(--ink3);font-size:12px;border-top:0}
.g2{display:grid;grid-template-columns:1fr auto;gap:4px 12px;padding:5px 12px;border-top:1px solid var(--line);font-size:13.5px}
.v{font-family:var(--mono);text-align:right;white-space:nowrap}.v.hot{color:var(--lime);font-weight:700}
.para{padding:8px 12px;color:var(--ink);font-size:13.5px;border-top:1px solid var(--line)}
.team{display:flex;align-items:center;gap:7px;padding:8px 12px 2px;font-weight:600;font-size:13.5px}
.card details{border-top:1px solid var(--line)}
.card summary{padding:9px 12px;cursor:pointer;color:var(--ink2);font-weight:600;font-size:13.5px;list-style-position:inside}
.card summary:hover{color:var(--ink)}
.nohot{padding:8px 12px;color:var(--ink3);font-size:13px;border-top:1px solid var(--line)}
</style></head><body>
<header><h1>__BRAND__</h1><span class="stamp" id="stamp"></span></header>
<nav aria-label="Sections"><div class="bar" id="groups"></div><div class="bar" id="tabs"></div></nav>
<main>
<div class="filters" id="filters">
<select id="fDay" aria-label="Day"></select><select id="fType" aria-label="League type"></select><select id="fLeague" aria-label="League"></select>
<input id="fText" type="search" placeholder="Search team" aria-label="Search team"><button id="fHot" class="hotbtn" type="button" aria-pressed="false">Hot facts only</button><span class="count" id="count"></span>
</div>
<div id="view"></div>
</main>
<footer>These are chances, not guarantees. Bet only what you can afford to lose. All columns are in the Excel file.</footer>
<script id="d" type="application/json">__DATA__</script>
<script>
/* everything inside a function: the page also works when it is opened by another page (the password lock) */
(function(){
var D=JSON.parse(document.getElementById("d").textContent),$=function(i){return document.getElementById(i)};
var S={g:0,t:0,sort:null,up:false,day:"",type:"",league:"",text:"",more:false,hot:false};
function el(t,c,x){var e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}
function fmt(col,v){if(v==null||v==="")return"";if(typeof v!=="number")return String(v);
 if(col.indexOf("%")>=0)return(v*100).toFixed(1)+"%";
 if(/odds|^Odds/i.test(col))return v.toFixed(2);
 if(/expected/i.test(col))return v.toFixed(1);
 if(/line/i.test(col))return v.toFixed(1);
 return String(v)}
function isNum(tab,i){return tab.rows.some(function(r){return typeof r[i]==="number"})}
function tab(){return D.groups[S.g].tabs[S.t]}
function btn(bar,text,on,fn){var b=el("button","",text);b.type="button";b.setAttribute("aria-pressed",on?"true":"false");b.onclick=fn;bar.appendChild(b)}
function drawNav(){var g=$("groups"),t=$("tabs");g.textContent="";t.textContent="";
 D.groups.forEach(function(x,i){if(i<D.main||S.more)btn(g,x.name,i===S.g,function(){S.g=i;S.t=0;reset();draw()})});
 if(D.main<D.groups.length)btn(g,S.more?"Less":"More",false,function(){S.more=!S.more;if(!S.more&&S.g>=D.main){S.g=0;S.t=0;reset()}draw()});
 btn(g,"How to read",S.g===-1,function(){S.g=-1;draw()});
 if(S.g>=0)D.groups[S.g].tabs.forEach(function(x,i){btn(t,x.name,i===S.t,function(){S.t=i;reset();draw()})})}
function reset(){S.sort=null;S.up=false}
function isSlip(t){return t.columns[0]==="Row"}
function fill(sel,label,vals,cur){sel.textContent="";var o=el("option","",label);o.value="";sel.appendChild(o);vals.forEach(function(v){var p=el("option","",v);p.value=v;sel.appendChild(p)});sel.value=vals.indexOf(cur)>=0?cur:""}
function uniq(t,col){var i=t.columns.indexOf(col);if(i<0)return[];var s={},out=[];t.rows.forEach(function(r){var v=r[i];if(v&&!s[v]){s[v]=1;out.push(v)}});return out}
function drawFilters(t){var slip=isSlip(t);
 fill($("fDay"),"All days",uniq(t,slip?"Day":"Date"),S.day);$("fDay").style.display=uniq(t,slip?"Day":"Date").length?"":"none";
 fill($("fType"),"All types",uniq(t,"League type"),S.type);$("fType").style.display=slip||t.columns.indexOf("League type")<0?"none":"";
 fill($("fLeague"),"All leagues",uniq(t,"League").sort(),S.league);$("fLeague").style.display=slip||t.columns.indexOf("League")<0?"none":"";
 $("fText").style.display=slip?"none":"";$("fText").value=S.text;
 $("fHot").style.display=t.facts?"":"none";$("fHot").setAttribute("aria-pressed",S.hot?"true":"false");
 S.day=$("fDay").value;S.type=$("fType").value;S.league=$("fLeague").value}
function drawTable(t,box){
 var dc=t.columns.indexOf("Date"),tc=t.columns.indexOf("League type"),lc=t.columns.indexOf("League"),mc=t.columns.indexOf("Match");
 var q=S.text.toLowerCase(),rows=t.rows.filter(function(r){return(!S.day||dc<0||r[dc]===S.day)&&(!S.type||tc<0||r[tc]===S.type)&&(!S.league||lc<0||r[lc]===S.league)&&(!q||mc<0||String(r[mc]).toLowerCase().indexOf(q)>=0)});
 if(S.sort!=null){var i=S.sort,u=S.up?1:-1;rows=rows.slice().sort(function(a,b){var x=a[i],y=b[i];if(x==null)return 1;if(y==null)return-1;return x<y?-u:x>y?u:0})}
 $("count").textContent=rows.length+(rows.length===1?" row":" rows");
 if(!rows.length){box.appendChild(el("div","empty","Nothing here for this filter today."));return}
 var w=el("div","scroll"),tb=el("table"),h=el("thead"),hr=el("tr"),body=el("tbody");
 var show=t.columns.map(function(c,i){return{c:c,i:i,n:isNum(t,i)}}).filter(function(x){return x.c!=="League type"});
 /* what you look at first comes first, so it is on screen on a phone. Top tabs keep the league next to the match */
 var top=t.name.indexOf("Top ")===0,head=top?["Match","League","Rank"]:["Match"],tail=top?["Date","Time"]:["Date","Time","League"];
 function pos(c){var h=head.indexOf(c),e=tail.indexOf(c);return h>=0?h-10:e>=0?e+10:0}
 show=show.map(function(x,k){x.k=k;return x}).sort(function(a,b){return pos(a.c)-pos(b.c)||a.k-b.k});
 show.forEach(function(x){var th=el("th",(x.n?"num ":"")+(x.c==="Match"?"match ":"")+(S.sort===x.i?"on "+(S.up?"up":""):""),x.c);th.setAttribute("scope","col");
  th.onclick=function(){if(S.sort===x.i)S.up=!S.up;else{S.sort=x.i;S.up=!x.n}draw()};hr.appendChild(th)});
 h.appendChild(hr);tb.appendChild(h);
 rows.forEach(function(r){var tr=el("tr");show.forEach(function(x){var v=r[x.i],td=el("td",(x.n?"num ":"")+(x.c==="Match"?"match":""));
  if(x.c==="Strength"&&v){td.appendChild(el("span","tag "+v,v))}else{td.textContent=fmt(x.c,v);if(x.c==="Pick %"&&v>=.65)td.className+=" hi"}
  tr.appendChild(td)});body.appendChild(tr)});
 tb.appendChild(body);w.appendChild(tb);box.appendChild(w)}
function drawSlips(t,box){var c={};t.columns.forEach(function(n,i){c[n]=i});var slips=[],cur=null;
 t.rows.forEach(function(r){if(r[c.Row]==="SLIP"){cur={day:r[c.Day],n:r[c.Slip],set:c.Set!=null?r[c.Set]:"",legs:[],odds:r[c.Odds],chance:r[c["Pick %"]],est:r[c.Price]};slips.push(cur)}else if(cur)cur.legs.push(r)});
 var list=slips.filter(function(s){return!S.day||s.day===S.day});
 $("count").textContent=list.length+(list.length===1?" slip":" slips");
 if(!list.length){box.appendChild(el("div","empty","No slip today: there are not enough safe matches, and slips are never forced."));return}
 var g=el("div","slips");list.forEach(function(s){var a=el("article","slip"),h=el("h3"),l=el("span","",s.set||"Slip "+s.n);l.appendChild(el("small","",s.day+" | chance all legs win "+fmt("%",s.chance)));
  h.appendChild(l);h.appendChild(el("span","odds",fmt("odds",s.odds)));a.appendChild(h);
  s.legs.forEach(function(r){var d=el("div","leg"),p=el("b","",r[c.Pick]),o=el("span","o",fmt("odds",r[c.Odds]));d.appendChild(p);d.appendChild(o);
   d.appendChild(el("span","m",r[c.Match]));d.appendChild(el("span","c",fmt("%",r[c["Pick %"]])));d.appendChild(el("span","l",r[c.League]));
   d.appendChild(r[c.Price]?el("span","est","estimate, check price"):el("span"));a.appendChild(d)});g.appendChild(a)});box.appendChild(g)}
/* Insights: one card per match, hot facts on top, the other parts open on a tap (built only then: fast on a phone) */
function ini(name){var s=String(name||"").replace(/[^A-Za-z0-9 ]/g,"").split(" ").filter(Boolean).slice(0,2).map(function(w){return w[0]}).join("").toUpperCase();return el("span","ini",s||"?")}
function badge(name,logo){if(!logo)return ini(name);var i=el("img","logo");i.alt="";i.loading="lazy";i.onerror=function(){if(i.parentNode)i.parentNode.replaceChild(ini(name),i)};i.src=logo;return i}
function hotv(v,min){var m=/^(\d+)\/(\d+)$/.exec(v||"");if(!m)return false;var h=+m[1],n=+m[2];return n>=min&&(h/n>=.8||(n-h)/n>=.8)}
function short(n){n=String(n||"");return n.length>12?n.slice(0,11)+".":n}
function row(cls,a,b,c){var d=el("div",cls);[a,b,c].forEach(function(x){if(x!=null)d.appendChild(typeof x==="string"?el("span","",x):x)});return d}
function fillSec(dt,name,fs,mark,H,A){
 if(name==="Last 10 matches"||name==="Goals by half"){dt.appendChild(row("g3 head",el("span"),el("span","v",short(H)),el("span","v",short(A))));
  var rows={},order=[];fs.forEach(function(f){if(!rows[f[2]]){rows[f[2]]={};order.push(f[2])}rows[f[2]][f[1]]=f[3]});
  order.forEach(function(s){var v1=rows[s][1]||"-",v2=rows[s][2]||"-",hot=name==="Last 10 matches";
   dt.appendChild(row("g3",s,el("span","v"+(hot&&hotv(v1,8)?" hot":""),v1),el("span","v"+(hot&&hotv(v2,8)?" hot":""),v2)))});return}
 if(name==="Odds check"){[1,2].forEach(function(k){var fk=fs.filter(function(f){return f[1]===k});if(!fk.length)return;
   dt.appendChild(row("team",mark(k),k===1?H:A));
   fk.forEach(function(f){dt.appendChild(f[2]==="Summary"?el("div","para",f[3]):row("g2",f[2],el("span","v",f[3])))})});return}
 if(name==="Head to head"){fs.forEach(function(f){dt.appendChild(f[2]==="Results"?el("div","para",f[3]):row("g2",f[2],el("span","v"+(hotv(f[3],5)?" hot":""),f[3])))});return}
 fs.forEach(function(f){dt.appendChild(row("team",mark(f[1]),(f[1]===1?H:f[1]===2?A:"")+" - "+f[2]));dt.appendChild(el("div","para",f[3]))})}
function drawCards(t,box){var c={};t.columns.forEach(function(n,i){c[n]=i});var q=S.text.toLowerCase(),list=[],sec=t.sections||[],HOT=sec.indexOf("Hot facts");
 t.rows.forEach(function(r,i){if((!S.day||r[c.Date]===S.day)&&(!S.type||r[c["League type"]]===S.type)&&(!S.league||r[c.League]===S.league)&&(!q||String(r[c.Match]).toLowerCase().indexOf(q)>=0)){
  var n=t.facts[i].filter(function(f){return f[0]===HOT}).length;if(!S.hot||n)list.push({r:r,f:t.facts[i],n:n})}});
 if(S.hot)list.sort(function(a,b){return b.n-a.n});
 $("count").textContent=list.length+(list.length===1?" match":" matches");
 if(!list.length){box.appendChild(el("div","empty",S.hot?"No hot facts for this filter today.":"Nothing here for this filter today."));return}
 var g=el("div","cards");
 list.forEach(function(x){var r=x.r,H=r[c.Home],A=r[c.Away],a=el("article","card"),h=el("div","ch"),by={};
  function mark(s){return s===1?badge(H,r[c["Home logo"]]):s===2?badge(A,r[c["Away logo"]]):el("span","ini both","H2H")}
  var tm=el("div","teams");[mark(1),el("span","",H),el("span","vs","v"),el("span","",A),mark(2)].forEach(function(e){tm.appendChild(e)});h.appendChild(tm);
  var sb=el("div","sub",[r[c.Date],r[c.Time],r[c.League]].filter(Boolean).join(" | "));if(x.n)sb.appendChild(el("span","badge",x.n+" hot"));h.appendChild(sb);a.appendChild(h);
  x.f.forEach(function(f){(by[f[0]]=by[f[0]]||[]).push(f)});
  var hf=by[HOT]||[];if(!hf.length)a.appendChild(el("div","nohot","No hot facts (8 of 10 or more) for this match"));
  function hrow(f){return row("fr",mark(f[1]),String(f[2]).replace(/^H2H: /,""),el("span","v hot",f[3]))}
  var top=S.hot?hf:hf.slice(0,5);top.forEach(function(f){a.appendChild(hrow(f))});
  if(hf.length>top.length){var mo=el("details");mo.appendChild(el("summary","",(hf.length-top.length)+" more hot facts"));
   mo.addEventListener("toggle",function(){if(mo.open&&mo.children.length===1)hf.slice(5).forEach(function(f){mo.appendChild(hrow(f))})});a.appendChild(mo)}
  if(!S.hot)sec.forEach(function(name,k){if(k===HOT||!by[k])return;var dt=el("details");dt.appendChild(el("summary","",name));
   dt.addEventListener("toggle",function(){if(dt.open&&dt.children.length===1)fillSec(dt,name,by[k],mark,H,A)});a.appendChild(dt)});
  g.appendChild(a)});
 box.appendChild(g)}
function body(t,box){if(isSlip(t))drawSlips(t,box);else if(t.facts)drawCards(t,box);else drawTable(t,box)}
function drawNotes(box){var n=el("div","notes");D.notes.forEach(function(x,i){if(!x)return;n.appendChild(/^[A-Z0-9 ()\/,]+$/.test(x.split("(")[0].trim())&&x.length<70?el("h2","",x):el("p","",x))});box.appendChild(n)}
function drawIntro(box,name){var x=D.intro[name];if(!x)return;var d=el("div","intro");x.forEach(function(p){d.appendChild(el("p","",p))});box.appendChild(d)}
function draw(){drawNav();var box=$("view");box.textContent="";
 if(S.g===-1){$("filters").style.display="none";drawNotes(box);return}
 $("filters").style.display="";var t=tab();drawFilters(t);drawIntro(box,t.name);body(t,box)}
["fDay","fType","fLeague"].forEach(function(i,k){$(i).onchange=function(){S[["day","type","league"][k]]=this.value;draw()}});
$("fText").oninput=function(){S.text=this.value;var box=$("view");box.textContent="";body(tab(),box)};
$("fHot").onclick=function(){S.hot=!S.hot;draw()};
$("stamp").textContent="Updated "+D.updated;draw();
})();
</script></body></html>
"""
