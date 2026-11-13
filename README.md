# PREDICT/90 Pro - Football Match Predictor

Home / Draw / Away probabilities for 38 leagues, Over/Under 2.5, BTTS, value bets with Kelly stakes,
corners, cards, shots, 4 daily accas, predictions by strength, injuries, lineups and a bet tracker. Results go to an Excel file every day.

---

## A. On your PC (Windows)

1. Install **Python 3.10 or newer** from https://www.python.org/downloads/
   - on the first install screen tick **"Add python.exe to PATH"**
2. Unzip this folder anywhere (for example Documents).
3. Double-click **`RUN_ME.bat`**
   - first time only: it installs the packages (2-5 min), downloads the data (2-4 min) and trains the models (about 5 min)
   - at the end the Excel file opens: `output/predictions_YYYY-MM-DD.xlsx`
4. Next days: double-click `RUN_ME.bat` again. A normal day takes about 1-2 minutes. The models train again by themselves every 7 days.

Close the Excel files before you run it, so they can be updated.

| File | What it does |
|---|---|
| `RUN_ME.bat` | daily predictions -> Excel |
| `OPEN_NOTEBOOK.bat` | opens the Jupyter notebook (all charts, backtest, tests) |
| `SCHEDULE_DAILY.bat` | makes Windows run the predictions by itself every day at a time you choose (the PC must be on) |
| `settings.json` | your settings: leagues, thresholds, bankroll, acca rules (every line is explained inside) |
| `Backtest_Report.html` | the full backtest report with charts. Opens in any browser, also on your phone |

## B. On your phone or any browser (Google Colab)

Nothing to upload. Sign in to Google in your browser first.

1. Open this link: https://colab.research.google.com/github/RicardoJose000/predict90-colab/blob/main/Football_Predictor_Colab.ipynb
2. Tap **Runtime -> Run all**. If Colab says the notebook is not from Google, tap **Run anyway**.
3. Allow access to Google Drive when asked. Your tracker, models, settings and key are kept in the `Football_Predictor_files` folder in My Drive.
4. Results appear in the notebook, part 2. The Excel and page files are saved in `Football_Predictor_files/output` on your Drive. Open them from the Drive app.

The first run in Colab takes longer (about 15-25 min, the free machine is slower). Next runs within 7 days are faster because the models stay on your Drive. Each new Colab session takes the newest code, so fixes reach you by themselves.

## C. API-Football key (free) - for injuries, lineups, more fixtures and odds

1. Sign up at https://dashboard.api-football.com (free plan, 100 requests a day).
2. Profile -> **My Access** -> copy the API-KEY.
3. Open the first cell of the notebook and paste it at `API_FOOTBALL_KEY = ""`, then run the cell. It is saved for you.
   Or in Notepad: write `API_FOOTBALL_KEY=your_key`, and save it in this folder as `.env` with "Save as type: All files".

Without a key everything still works, but only with the matches published by football-data.co.uk (mostly on Fridays and Tuesdays), and without injuries.
A daily run uses about 10-40 of the 100 free requests. The free plan only gives today and tomorrow, and 10 requests a minute, so the first run of the day takes a few minutes longer.

---

## What is inside

**Data** - football-data.co.uk: results, shots and bookmaker odds for 38 leagues (22 main leagues since 2005/06, 16 more since 2012), upcoming fixtures with odds. API-Football: today/tomorrow fixtures, Bet365 odds (1X2, O/U 2.5, BTTS), injuries, lineups. Team names are cleaned, and new spellings in the source are found automatically.

**Features** (all from before kickoff) - Elo ratings with promotion and relegation handled, form over the last 5 and 10 matches, home form at home and away form away, goals for and against, shots on target, attack/defence strength, expected goals, Poisson probabilities, head to head, rest days, matches played, league trends, bookmaker probabilities with the margin removed.

**Models** - Logistic Regression + XGBoost + LightGBM + Neural Network, combined by a stacking meta model trained on the newest 15% of matches, which also calibrates the probabilities. Tuned with Optuna on seasons before 2018/19 (`models/best_params.json`). A second model handles matches with no odds yet. Separate models for Over/Under 2.5 and BTTS.

**Backtest** - walk-forward, season by season: train only on the past, predict the next season. Accuracy, log-loss, Brier score, calibration, accuracy by confidence, by season and by league, significance test against the bookmaker, value-bet profit, and the daily acca slips settled day by day.

**Betting tools** - value bets (chance x odds >= 1.05), quarter-Kelly stakes capped at 3% of the bankroll, a bet tracker with automatic settling that logs each acca apart, and 4 accas (22 main leagues by default, total odds 4.5-7, within one sheet each match once). **Main acca**: up to 5 a day, win, double chance and over/under 2.5, legs 50%+, at most 2 of one type. Its profit is tested: it lost at average odds. **Strong acca**: favourites to win only, every leg 65%+, 1 or 2 a day and only on some days. A small profit in the test, not proven. **Banger acca**: favourites to win only, every leg 70%+, about 40 a season. A small profit in the test, not proven: acca numbers move a lot with tiny changes. **Other acca**: up to 5 a day, BTTS, over 1.5, under 3.5, favourite to score, underdog +1.5, favourite or over 2.5, corners, corner 1X2, cards and shots. No old odds exist for these, so accuracy is tested but profit is not. Where no real odds are found, only bets 60%+ sure are used and the price is marked `estimate`: check the real price at your bookmaker. 1up and 2up are not included, because the free data has no goal minutes. Part 7 of the notebook has all the tests.

**Results page** - every run also saves `output/predictions_...html`: the same results as the Excel file, as one page with tabs and filters, that works on a phone and with no internet. On a PC it opens by itself after `RUN_ME.bat`. In Google Colab it shows inside the notebook, part 2.

**Bangers by market** - the safest pick of each kind, only 75%+ (`bangers_min_prob`): `1X2`, `Over 2.5 goals` (half our chance, half the bookmaker's), `Over 1.5 goals` (only matches not already in Over 2.5), `Corners over`, `Cards over` and `Shots over` (every match once, at its highest line that is still 75%+, never below 6.5 corners, 2.5 cards, 15.5 shots). Test on 9 seasons with exactly this code (part 5c of the notebook): right 1X2 81%, Over 2.5 81%, Over 1.5 80%, corners 79%, cards 82%, shots 78%, lowest season 73-79%; national teams 1X2 86%, Over 1.5 81% (competitive matches only, no Over 2.5: their goal chances were too high in the test). No BTTS bangers: the model never reached 75% for BTTS in 100,000 test matches. `Banger accas`: one acca per market and day from that market's bangers, total odds about 2 to 3, so the same match is never twice in a slip. In the test all legs won in 30-42% of slips; at real old odds the 1X2 acca lost about 8% per slip (the bookmaker margin), the others have no old odds, so profit is not tested. Domestic cups are in 1X2 (82% in the test), Over 2.5 and Over 1.5; European cups only in Over 1.5. The page shows `My picks`, `Bangers`, `Mixed acca` and `All predictions` first, the rest behind the `More` button (`simple_page: false` shows everything). The Excel file always has every sheet.

**Mixed acca** - one acca a day from the bangers of every kind: the top 2 of 1X2, Over 2.5, Over 1.5, corners, cards and shots (`mixed_acca_per_market`, 1 = shorter and safer), safest first, every match only once (a match that is top in two kinds goes where it is safest). Test on 9 seasons: about 7.6 legs at 3.8 total odds, every leg won on 22% of days (the model said 22%), on another 32% only one leg lost. With top 1: about 4.4 legs at 2.1 odds, won 38%. Most corners, cards and shots prices are estimates, so profit is not tested.

**Tabs (the Excel file and the page)** - `My picks`: my own recommendation, 1 long acca and up to 2 safer accas, no match used twice. About break-even in the test, not a sure profit (part 7 of the notebook). `All predictions` (every match: what is not in a list or an acca is still here), `Bangers` (75%+), `Strong` (65-75%) and `Weak` (below 50%), with `Good` (50-65%) inside `All predictions`. The 4 accas. `Top favourites`, `Top goals`, `Top BTTS` and `Top corners`: the top matches of each league, 3 by default (`top_n_per_league` in `settings.json`). `Over-Under 2.5`, `BTTS`, `Goal lines`, `Team to score`, `Underdog +1.5`, `Corners`, `Cards` and `Shots`: the chance of each bet for every match, with real odds when there are any. Filter by day and league, also `League type` (Main, Other, Cup or International).

**National teams** - World Cup and Euro qualifiers, Nations League, AFCON, Copa America, Gold Cup, Asian Cup and friendlies, today and tomorrow, in the tabs `Internationals` and `International acca` (League type = International). Needs the API-Football key, and they only come in the international windows. Their own model: ratings for every country plus recent goals, tested on 11 years (part 5b of the notebook): competitive picks 65%+ were right 79% of the time, 75%+ 86%. Friendlies are always WEAK. **International acca**: favourites to win only, competitive matches only, every leg 70%+. Accuracy tested, profit not: the free data has no old odds for national teams, so they also get no value bets. Competitive national team matches (no friendlies) can also be in `My picks`, the strong and the banger acca (`intl_in_picks: false` keeps those for clubs only). Their results come from API-Football the day after, so the tracker settles them quickly. Turn them off with `internationals: false` in `settings.json`.

**Cups** - Champions League, Europa League, Conference League, FA Cup, Carabao Cup, Copa del Rey, DFB Pokal, Coppa Italia, Coupe de France, Scottish Cup and League Cup, KNVB Beker and Taca de Portugal, today and tomorrow, in the tabs `Cups` and `Cup acca` (League type = Cup). Needs the API-Football key. Their own model on the clubs' league form: strength, form, level (a small club below our leagues is "small club"), country strength in European matches (learned from European results), neutral finals; trained on the 90 minute score, because bets settle after 90 minutes. Tested on 2 seasons, each predicted only with the seasons before it (part 5d of the notebook, 2,134 matches): all cups 57% right, picks 65%+ 75%, 75%+ 79%; domestic cups 75%+ 82%; European cups 75%+ only 71%, so a European pick is never BANGER. Skipped, no blind guesses: European matches with a club from a country we have no league for, and small clubs whose name looks like a league club's. **Cup acca**: cup favourites to win only, every leg 70%+ (`cup_acca_min_prob`). Cup matches can also be in `My picks`, the strong and the banger acca (`cups_in_picks: false` keeps them out). No value bets on cups: no old cup odds, so profit is not tested. Turn them off with `cups: false`.

**Auto run (no PC, no Colab)** - GitHub can run the predictor every morning by itself and put the page on your own link, locked with your password: see github.com/RicardoJose000/predict90-auto (5 minutes to set up, free).

## Check the numbers yourself

In the notebook, part 4, set `RUN_BACKTEST = True` and press Run All (about 1 hour). With `USE_SNAPSHOT = True` it uses the saved data in `data_snapshot/` and fixed random seeds, so you get the same numbers as in `Backtest_Report.html` on the library versions in `requirements.txt`. Other versions, like Google Colab's, give about the same picks but can move the acca numbers.

## Bet tracker

`output/bet_tracker.xlsx` logs every strong pick, value bet and acca slip, and settles them by itself when the results arrive. Write `Y` in the `Played` column for what you really bet, and change `Stake` if you want. The `Summary` sheet shows your real record. Close the file before the next run.

## Leagues (codes for settings.json)

| Code | League | Code | League |
|---|---|---|---|
| E0 | England Premier League | ARG | Argentina Liga Profesional |
| E1 | England Championship | AUT | Austria Bundesliga |
| E2 | England League One | BRA | Brazil Serie A |
| E3 | England League Two | CHN | China Super League |
| EC | England National League | DNK | Denmark Superliga |
| SC0 | Scotland Premiership | FIN | Finland Veikkausliiga |
| SC1 | Scotland Championship | IRL | Ireland Premier Division |
| SC2 | Scotland League One | JPN | Japan J1 League |
| SC3 | Scotland League Two | MEX | Mexico Liga MX |
| D1 | Germany Bundesliga | NOR | Norway Eliteserien |
| D2 | Germany 2. Bundesliga | POL | Poland Ekstraklasa |
| I1 | Italy Serie A | ROU | Romania Liga 1 |
| I2 | Italy Serie B | RUS | Russia Premier League |
| SP1 | Spain La Liga | SWE | Sweden Allsvenskan |
| SP2 | Spain Segunda Division | SWZ | Switzerland Super League |
| F1 | France Ligue 1 | USA | USA MLS |
| F2 | France Ligue 2 | | |
| N1 | Netherlands Eredivisie | | |
| B1 | Belgium Pro League | | |
| P1 | Portugal Primeira Liga | | |
| T1 | Turkey Super Lig | | |
| G1 | Greece Super League | | |

## Folders

| Folder | |
|---|---|
| `p90/` | the full source code, yours to keep |
| `data_snapshot/` | clean data used for the backtest |
| `data/` | downloaded data (filled by the first run) |
| `models/` | tuned settings + trained models |
| `output/` | Excel predictions, bet tracker, backtest results |
| `team_aliases.csv` | team names API-Football -> football-data. If a team is skipped, add a line here |

## If something goes wrong

| Message | What to do |
|---|---|
| Python is not installed | Install it (part A) and tick "Add python.exe to PATH" |
| DLL load failed ... An Application Control policy has blocked this file | Windows Smart App Control is blocking a library file. This is not a folder problem or a project error. Try `python -m pip install "pandas==2.2.3"` and run again. If it stays, use Google Colab (part B), where nothing runs on your PC. Or turn it off: Windows Security > App & browser control > Smart App Control settings > Off. That is your PC's security, so your choice: on older Windows 11 turning it back on can need a reset. |
| Unable to allocate ... (memory error) | Training needs about 2-3 GB of free memory. Close other programs, mainly browsers, and run again |
| bet_tracker.xlsx is open in Excel | Close it and run again |
| Could not download the football data | Check your internet and run again |
| skipped teams with unknown names | Add the team to `team_aliases.csv` |
| No upcoming matches found | No fixtures are published yet. Add the API key (part C) or try later |

These are probabilities, not guarantees. Bet only what you can afford to lose.
