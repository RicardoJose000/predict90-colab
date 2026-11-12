"""The results as one HTML page: works offline, on a PC or a phone. Same tables as the Excel file."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

INTRO = {"My picks": [
    "My own recommendation: 1 long acca around 5 odds and up to 2 safer accas around 2.3 odds, only when the model has them. No match is used twice. Same small stake on every slip, at most 1% of your bankroll.",
    "This exact rule in the 9 season test: 28.0% of slips won, +1.7% at average odds, -0.3% at Bet365, range -9.8% to +13.5%. So about break-even, not a sure profit. Expect losing runs: up to 18 long slips or 10 safer slips in a row.",
    "I would skip: weak picks, the main acca, other leagues in accas, and the other acca until your tracker shows it works.",
    "National teams: competitive matches (no friendlies) can be in My picks, the strong and the banger acca too. Their accuracy is tested (79% for 65%+ picks), their profit is not: no old odds for national teams.",
], "Internationals": [
    "National teams, with their own model: ratings for every country plus recent goals. Only in the international windows, and only with the API-Football key.",
    "Test on 11 years (2016-2026), competitive matches: picks 65%+ were right 79% of the time, picks 75%+ 86%. When it says 70%, it happened about 70% of the time.",
    "Friendlies are always WEAK: squads change a lot there. No value bets here: the free data has no old odds for national teams, so profit is not tested.",
], "International acca": [
    "National teams, favourites to win only, competitive matches only (no friendlies), every leg 70%+. Accuracy is tested (79% for 65%+ picks), profit is not: no old odds for national teams.",
]}
GROUPS = [
    ("My picks", ["My picks"]),
    ("Predictions", ["All", "Bangers", "Strong", "Weak", "Internationals", "Value bets"]),
    ("Accas", ["Main acca", "Strong acca", "Banger acca", "Other acca", "International acca"]),
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


def _sheet(name: str, df: pd.DataFrame) -> dict:
    d = df.copy()
    if "Row" not in d and name not in SHEETS_WITH_ALL_COLUMNS:
        d = d[[c for c in d.columns if c not in HIDE]]
    if "Home" in d and "Away" in d:
        d.insert(list(d.columns).index("Home"), "Match", d["Home"].astype(str) + " v " + d["Away"].astype(str))
        d = d.drop(columns=["Home", "Away"])
    return {"name": name, "columns": list(d.columns),
            "rows": [[_clean(v) for v in row] for row in d.astype(object).where(d.notna(), None).to_numpy().tolist()]}


def write(path, sheets: dict[str, pd.DataFrame], notes: list[str]) -> None:
    data = {"updated": f"{datetime.now():%a %d %b %Y, %H:%M}", "notes": notes, "intro": INTRO,
            "groups": [{"name": g, "tabs": [_sheet(n, sheets[n]) for n in names if n in sheets]} for g, names in GROUPS]}
    data["groups"] = [g for g in data["groups"] if g["tabs"]]
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    path.write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PREDICT/90 Results</title>
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
</style></head><body>
<header><h1>PREDICT<b>/90</b></h1><span class="stamp" id="stamp"></span></header>
<nav aria-label="Sections"><div class="bar" id="groups"></div><div class="bar" id="tabs"></div></nav>
<main>
<div class="filters" id="filters">
<select id="fDay" aria-label="Day"></select><select id="fType" aria-label="League type"></select><select id="fLeague" aria-label="League"></select>
<input id="fText" type="search" placeholder="Search team" aria-label="Search team"><span class="count" id="count"></span>
</div>
<div id="view"></div>
</main>
<footer>These are chances, not guarantees. Bet only what you can afford to lose. All columns are in the Excel file.</footer>
<script id="d" type="application/json">__DATA__</script>
<script>
var D=JSON.parse(document.getElementById("d").textContent),$=function(i){return document.getElementById(i)};
var S={g:0,t:0,sort:null,up:false,day:"",type:"",league:"",text:""};
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
 D.groups.forEach(function(x,i){btn(g,x.name,i===S.g,function(){S.g=i;S.t=0;reset();draw()})});
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
function drawNotes(box){var n=el("div","notes");D.notes.forEach(function(x,i){if(!x)return;n.appendChild(/^[A-Z0-9 ()\/,]+$/.test(x.split("(")[0].trim())&&x.length<70?el("h2","",x):el("p","",x))});box.appendChild(n)}
function drawIntro(box,name){var x=D.intro[name];if(!x)return;var d=el("div","intro");x.forEach(function(p){d.appendChild(el("p","",p))});box.appendChild(d)}
function draw(){drawNav();var box=$("view");box.textContent="";
 if(S.g===-1){$("filters").style.display="none";drawNotes(box);return}
 $("filters").style.display="";var t=tab();drawFilters(t);drawIntro(box,t.name);if(isSlip(t))drawSlips(t,box);else drawTable(t,box)}
["fDay","fType","fLeague"].forEach(function(i,k){$(i).onchange=function(){S[["day","type","league"][k]]=this.value;draw()}});
$("fText").oninput=function(){S.text=this.value;var box=$("view");box.textContent="";drawTable(tab(),box)};
$("stamp").textContent="Updated "+D.updated;draw();
</script></body></html>
"""
