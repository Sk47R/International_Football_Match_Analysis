# International Football Match Analysis

> 150 years of international football: engineered, rated, analysed, predicted and visualised.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-blue?logo=pandas)](https://pandas.pydata.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple?logo=plotly)](https://plotly.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

This project is a full end-to-end data science pipeline built on 49,215 international football matches played between 1872 and 2026. It answers three questions that football analysts, sports journalists and data scientists genuinely care about:

| Question                                       | Answer                                                        |
| ---------------------------------------------- | ------------------------------------------------------------- |
| Who is the greatest national team of all time? | Custom Elo engine rates all 333 nations across 150 years      |
| What patterns exist in 150 years of football?  | 9 EDA visualisations covering goals, home advantage and eras  |
| Who will win a given match?                    | XGBoost model predicts outcomes with calibrated probabilities |

The project covers every stage of a real-world analytics pipeline: data engineering, exploratory analysis, statistical modelling, machine learning and interactive deployment.

---

## Dataset

The raw data covers 49,215 international football matches from the first official match (England vs Scotland, 30 November 1872) to 31 March 2026. Sourced from [martj42/international_results](https://github.com/martj42/international_results), which aggregates data from Wikipedia, rsssf.com and individual football association websites.

| File             | Rows   | Columns | Description                                                                |
| ---------------- | ------ | ------- | -------------------------------------------------------------------------- |
| results.csv      | 49,215 | 9       | Match results: date, teams, score, tournament, city, country, neutral flag |
| goalscorers.csv  | 47,601 | 8       | Goal records: scorer, minute, own goal and penalty flags                   |
| shootouts.csv    | 598    | 5       | Penalty shootout winner and first shooter                                  |
| former_names.csv | 151    | 4       | Historical name changes e.g. Dahomey to Benin, Zaire to DR Congo           |

**Note on naming:** Current team names are used throughout history for consistency. Country names at time of match are preserved in the `country` column. The `neutral` flag distinguishes home fixtures from matches played at neutral venues (26.4% of all matches).

---

## Notebooks

### Data Pipeline

**File:** `notebooks/Data_Analysis_and_Engineering.ipynb`

Transforms 4 raw CSV files into a single clean, analysis-ready dataset with 26 engineered columns.

**Key steps:**

- Resolves 151 historical team name changes using former_names.csv so that e.g. "Dahomey" and "Benin" are treated as the same team
- Separates 72 future World Cup 2026 fixtures (null scores) from 49,215 completed matches
- Engineers new columns: result, goal_diff, total_goals, decade, era, tournament_tier, is_high_stakes, has_home_advantage, went_to_shootout, own_goals_count, penalties_count, unique_scorers
- Joins shootout outcomes and per-match goalscorer aggregates
- Exports to CSV and a SQLite database with 5 tables

**Tournament tier classification:**

| Tier               | K-factor in Elo | Examples                                               |
| ------------------ | --------------- | ------------------------------------------------------ |
| 1 - Major Finals   | 60              | FIFA World Cup, Copa America, UEFA Euro, AFC Asian Cup |
| 2 - Qualifications | 40              | All major tournament qualification campaigns           |
| 3 - Regional Cups  | 30              | Nations League, Gulf Cup, COSAFA, CECAFA               |
| 4 - Friendlies     | 20              | All friendly matches                                   |

**Output:** `clean_matches.csv` with 49,215 rows and 26 columns

---

### Raw EDA: Exploratory Data Analysis

**File:** `notebooks/Data_Visualization.ipynb`

9 Plotly charts exploring the raw data before any modelling. All saved as PNG to `visualizations/eda_charts/`. Each chart is also linked to a specific design decision in the Elo engine.

| Step | Chart                    | Key Finding                                                                     | Elo Design Decision                                            |
| ---- | ------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 1    | Match volume over time   | Grew from 1 match/year (1872) to 1,200+/year (2024). WW1, WW2 and COVID visible | Early ratings pre-1920 are noisy; start all teams at 1000      |
| 2    | Result split             | Home teams win 49.0%, draws 22.8%, away wins 28.3%                              | Add +100 home advantage constant to Elo                        |
| 3    | Goals distribution       | Right-skewed; 2 goals is most common outcome                                    | Use log compression in margin-of-victory multiplier            |
| 4    | Common scorelines        | 1-0 is the most common result in history (5,079 times)                          | Football is low-scoring; Elo must be sensitive to single goals |
| 5    | Home advantage by decade | Gap persists across all 150 years of football                                   | Fixed +100 home bonus is empirically justified                 |
| 6    | Top teams by win rate    | Brazil leads major nations at 63.4%                                             | Elo rankings should reproduce this ordering                    |
| 7    | Most active nations      | Sweden most active (1,099 matches); activity and quality have weak correlation  | Teams with few matches have unreliable Elo ratings             |
| 8    | Goals per decade         | Peaked 1880s at 5.6/game, bottomed 1980s at 2.5/game, plateaued at 2.7 since    | Margin-of-victory calibrated to modern scoring rates           |
| 9    | Tournament breakdown     | World Cup has lowest home win rate at 46.1%                                     | Higher K-factor for major tournaments is justified             |

---

### Elo Rating Engine

**File:** `notebooks/ELO_Rating.ipynb`

A custom Elo rating system built from scratch. Processes all 49,215 matches in chronological order and assigns every team a strength rating at every point in history.

**Core formula:**

```
Expected score : E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
Rating update  : R_new = R_old + K x MoV x (Actual - Expected)
```

**Three improvements over standard Elo:**

| Improvement                  | Value                                             | Justification from EDA                                                   |
| ---------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------ |
| Home advantage constant      | +100 pts to home team effective rating            | Home teams win 49.0% of all non-neutral matches                          |
| K-factor by tournament tier  | K = 60 / 40 / 30 / 20                             | World Cup final is not the same as a friendly                            |
| Margin of victory multiplier | min(ln(goal_diff + 1) x autocorr_correction, 2.5) | Goals distribution is right-skewed; 5-0 is not 5x more valuable than 1-0 |

**Validation on test data:**

| Metric                               | Result |
| ------------------------------------ | ------ |
| Favourite won (all non-draw matches) | 72.2%  |
| Favourite won (FIFA World Cup only)  | 72.1%  |
| Random baseline                      | 50.0%  |

**Outputs:**

- `elo_matches.csv`: 49,215 rows x 22 columns including pre/post Elo for both teams, expected win probability, and upset flag
- `elo_final_ratings.csv`: 333 nations with final rating, rank, total matches, wins, draws, losses and win rate
- `elo_history.csv`: 98,430 rows recording every team's Elo after every match

**Final top 10 Elo rankings:**

| Rank | Team        | Elo Rating | Matches | Win Rate |
| ---- | ----------- | ---------- | ------- | -------- |
| 1    | Argentina   | 1629.63    | 1,064   | 55.3%    |
| 2    | Spain       | 1616.76    | 781     | 58.9%    |
| 3    | France      | 1541.70    | 933     | 51.0%    |
| 4    | England     | 1511.91    | 1,088   | 57.3%    |
| 5    | Colombia    | 1492.77    | 636     | 40.3%    |
| 6    | Brazil      | 1484.61    | 1,057   | 63.4%    |
| 7    | Portugal    | 1457.09    | 693     | 50.1%    |
| 8    | Netherlands | 1451.07    | 877     | 51.5%    |
| 9    | Ecuador     | 1440.55    | 589     | 30.7%    |
| 10   | Morocco     | 1434.38    | 614     | 49.5%    |

Highest Elo ever recorded: **Argentina 1686.58**

---

### ML Match Prediction

**File:** `notebooks/Match_Prediction.ipynb`

An XGBoost classifier predicting match outcomes (home win, draw or away win) with calibrated probabilities. Trained on 28,750 matches (pre-2015) and tested on 7,702 matches (2015 to 2026).

**Why time-based split?**
A random split would leak future match results into training, artificially inflating accuracy. Training on pre-2015 and testing on 2015 onwards simulates real-world deployment where only past data is available.

**37 columns in the ML dataset, with 20 engineered features used for training:**

| Category     | Features                                                              |
| ------------ | --------------------------------------------------------------------- |
| Elo strength | exp_home_win_prob, elo_diff, home_elo_pre, away_elo_pre               |
| Recent form  | Rolling win/draw/loss rate over last 5 and 10 matches (home and away) |
| Goal scoring | Rolling average goals scored and conceded over last 10 matches        |
| Head-to-head | h2h_home_win_rate, h2h_total_matches                                  |
| Context      | tournament_tier, neutral, rest_advantage, month                       |

**Model comparison (test set: 2015 to 2026, 7,702 matches):**

| Model                           | Accuracy | Log-loss | Brier Score |
| ------------------------------- | -------- | -------- | ----------- |
| Naive (always predict home win) | 47.2%    | N/A      | N/A         |
| Logistic Regression (baseline)  | 57.8%    | 0.9012   | 0.1775      |
| XGBoost                         | 58.2%    | 0.9003   | 0.1772      |

**Top 10 features by XGBoost gain-based importance:**

| Rank | Feature              | Importance % |
| ---- | -------------------- | ------------ |
| 1    | exp_home_win_prob    | 23.76%       |
| 2    | elo_diff             | 12.78%       |
| 3    | month                | 7.64%        |
| 4    | home_avg_scored_10   | 7.51%        |
| 5    | home_avg_conceded_10 | 6.40%        |
| 6    | away_avg_conceded_10 | 4.41%        |
| 7    | neutral              | 3.27%        |
| 8    | home_elo_pre         | 3.12%        |
| 9    | home_win_rate_10     | 3.12%        |
| 10   | h2h_total_matches    | 3.11%        |

**2026 World Cup predictions:**

- 72 group stage fixtures predicted
- 42 home team favoured, 30 away team favoured
- Output includes home win %, draw % and away win % for every fixture

---

### Streamlit Dashboard

**File:** `website/dashboard.ipynb`

An interactive 6-page web application that makes all findings accessible without requiring Python knowledge.

| Page            | Content                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| Overview        | KPIs, match volume chart, home advantage decay, key findings               |
| Team Explorer   | Any nation's Elo history, results by decade, biggest wins, goal trends     |
| Head-to-Head    | Full rivalry breakdown between any two nations with complete match history |
| Elo Rankings    | Current world rankings bar chart and multi-team timeline comparison        |
| Match Predictor | Live XGBoost prediction for any fixture with win/draw/loss probabilities   |
| 2026 World Cup  | All 72 predicted group stage fixtures with filterable probability table    |

---

## Getting Started

### Option 1: Google Colab (recommended)

1. Open the desired notebook in Colab
2. Run the pip install cell at the top of the notebook
3. Upload the relevant CSV from the `dataset/` folder to `/content/`
4. Run all cells in order

### Option 2: Run locally

```bash
git clone https://github.com/Sk47R/International_Football_Match_Analysis.git
cd International_Football_Match_Analysis
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in this order:

1. Data_Analysis_and_Engineering.ipynb
2. Data_Visualization.ipynb
3. ELO_Rating.ipynb
4. Match_Prediction.ipynb
5. dashboard.ipynb

To launch the dashboard separately:

```bash
streamlit run module5_dashboard.py
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
plotly>=5.0
kaleido>=0.2
scikit-learn>=1.3
xgboost>=2.0
streamlit>=1.28
sqlalchemy>=2.0
jupyter>=1.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Numbers

| Metric                               | Value                                       |
| ------------------------------------ | ------------------------------------------- |
| Total completed matches              | 49,215                                      |
| Date range                           | 30 Nov 1872 to 31 Mar 2026                  |
| Unique nations                       | 333                                         |
| Unique tournaments                   | 193                                         |
| Total goals scored                   | 144,618                                     |
| Average goals per match              | 2.938                                       |
| Home win rate                        | 49.0%                                       |
| Away win rate                        | 28.3%                                       |
| Draw rate                            | 22.8%                                       |
| Neutral venue matches                | 26.4%                                       |
| Most common scoreline                | 1-0 (5,079 matches)                         |
| Highest scoring match                | Australia 31-0 American Samoa (11 Apr 2001) |
| Most matches played                  | Sweden (1,099)                              |
| Highest win rate among major nations | Brazil (63.4%)                              |
| Unique goalscorers recorded          | 15,335                                      |
| Own goals in dataset                 | 922                                         |
| Penalty goals in dataset             | 3,249                                       |
| Highest Elo ever recorded            | Argentina 1686.58                           |
| Current top Elo nation               | Argentina (1629.63)                         |
| Elo favourite win rate               | 72.2%                                       |
| Elo World Cup accuracy               | 72.1%                                       |
| ML dataset size                      | 36,452 matches x 37 columns                 |
| ML train set                         | 28,750 matches (pre-2015)                   |
| ML test set                          | 7,702 matches (2015 to 2026)                |
| XGBoost accuracy                     | 58.2% (vs 47.2% naive baseline)             |
| XGBoost log-loss                     | 0.9003                                      |
| 2026 WC fixtures predicted           | 72 group stage matches                      |

---

## Pipeline Architecture

```
Raw CSVs (4 files, 1872 to 2026)
              |
              v
    Module 1: Data Pipeline
    49,215 rows x 26 cols output
              |
       +------+--------+
       |               |
       v               v
  Raw EDA          Module 3: Elo Engine
  9 PNG charts     98,430 history records
  Justifies Elo    333 nations rated
  design choices   Validates at 72.2%
       |               |
       |               v
       |         Module 2: EDA with Elo
       |         Dominance eras
       |         Historical patterns
       |               |
       +------+--------+
              |
              v
     Module 4: ML Prediction
     36,452 rows, 20 features
     XGBoost accuracy 58.2%
     72 WC 2026 predictions
              |
              v
     Module 5: Dashboard
     Streamlit, 6 pages
     Live match predictor
```

---

## Acknowledgements

- Dataset: [Mart Jurisoo](https://github.com/martj42/international_results) for maintaining the international football results dataset since 1872
- Data sources: Wikipedia, rsssf.com and individual football association websites
- Elo methodology: Inspired by [FiveThirtyEight Soccer Power Index](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/) and [Club Elo](http://clubelo.com/System)

---

If you found this project useful, consider giving it a star on GitHub.
