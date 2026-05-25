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

The project covers every stage of a real-world analytics pipeline: data engineering, exploratory analysis, statistical modelling, machine learning, and interactive deployment.

---

## Dataset

The raw data covers 49,215 international football matches from the first official match (England vs Scotland, 1872) to 2026. Sourced from [martj42/international_results](https://github.com/martj42/international_results), which aggregates data from Wikipedia, rsssf.com and individual football association websites.

| File             | Rows   | Description                                                             |
| ---------------- | ------ | ----------------------------------------------------------------------- |
| results.csv      | 49,215 | Match results: date, teams, score, tournament, venue, neutral flag      |
| goalscorers.csv  | 47,601 | Goal records: scorer name, minute, own goal and penalty flags           |
| shootouts.csv    | 598    | Penalty shootout winner and first shooter                               |
| former_names.csv | 151    | Historical team name changes (e.g. Dahomey to Benin, Zaire to DR Congo) |

**Note on naming:** Current team names are used throughout history for consistency. The 1882 "Ireland" team is listed as "Northern Ireland" because that is the successor team. Country names at time of match are preserved separately in the `country` column.

---

## Notebooks

### Module 1: Data Pipeline

**File:** `notebooks/Data_Analysis_and_Engineering.ipynb`

Transforms 4 raw CSV files into a single clean, analysis-ready dataset.

**Key steps:**

- Resolves 151 historical team name changes using former_names.csv
- Separates 72 future World Cup 2026 fixtures from completed matches
- Engineers 10 new columns: result, goal_diff, total_goals, decade, era, tournament_tier, is_high_stakes, has_home_advantage, went_to_shootout, unique_scorers
- Joins shootout outcomes and per-match goalscorer aggregates
- Exports to CSV files and a SQLite database with 5 tables

**Tournament tier system used throughout the project:**

| Tier               | K-factor | Tournaments                                            |
| ------------------ | -------- | ------------------------------------------------------ |
| 1 - Major Finals   | 60       | FIFA World Cup, Copa America, UEFA Euro, AFC Asian Cup |
| 2 - Qualifications | 40       | All major tournament qualification campaigns           |
| 3 - Regional Cups  | 30       | Nations League, Gulf Cup, COSAFA, CECAFA               |
| 4 - Friendlies     | 20       | All friendly matches                                   |

---

### EDA: Exploratory Data Analysis

**File:** `notebooks/Data_Visualization.ipynb`

9 Plotly visualisation charts exploring the raw data before any modelling. All charts are saved as PNG files to `visualizations/eda_charts/`.

| Step | Chart                    | Key Finding                                                                             |
| ---- | ------------------------ | --------------------------------------------------------------------------------------- |
| 1    | Match volume over time   | 1 match/year (1872) grew to 1,200+ per year by 2024. WW1, WW2 and COVID visible as dips |
| 2    | Result split             | Home teams win 49% of all matches, proving home advantage is structural                 |
| 3    | Goals distribution       | Right-skewed distribution justifies log compression in Elo margin-of-victory            |
| 4    | Common scorelines        | 1-0 is the most common result in history (5,079 times)                                  |
| 5    | Home advantage by decade | The gap has persisted across all 150 years of football                                  |
| 6    | Top teams by win rate    | Brazil leads all major nations at 63.4%                                                 |
| 7    | Most active nations      | Activity and quality have only weak correlation (r approx 0.3)                          |
| 8    | Goals per decade         | Peaked in 1880s at 5.6 per game, bottomed in 1980s at 2.5, plateaued at 2.7 since       |
| 9    | Tournament breakdown     | World Cup has the lowest home win rate at 46.1%, pressure equalises teams               |

---

### Module 3: Elo Rating Engine

**File:** `notebooks/module3_elo.ipynb`

A custom Elo rating system built from scratch. Processes all 49,215 matches chronologically and assigns every team a strength rating at every point in history.

**Core formula:**

```
Expected score  :  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
Rating update   :  R_new = R_old + K x MoV x (Actual - Expected)
```

**Three improvements over standard Elo:**

| Improvement       | Implementation                                    | Justification                              |
| ----------------- | ------------------------------------------------- | ------------------------------------------ |
| Home advantage    | +100 pts added to home team effective rating      | Home teams win 49% empirically proven      |
| K-factor by tier  | K = 60 / 40 / 30 / 20 per tier                    | World Cup final is not equal to a friendly |
| Margin of Victory | min(ln(goal_diff + 1) x autocorr_correction, 2.5) | A 5-0 win is not 5x more valuable than 1-0 |

**Validation results:**

| Metric                                 | Result |
| -------------------------------------- | ------ |
| Favourite won (all non-draw matches)   | 72.3%  |
| Favourite won (World Cup matches only) | 72.9%  |
| Random baseline                        | 50.0%  |

**Final top 10 Elo rankings:**

| Rank | Team        | Elo Rating |
| ---- | ----------- | ---------- |
| 1    | Spain       | 1,610      |
| 2    | Argentina   | 1,589      |
| 3    | France      | 1,527      |
| 4    | England     | 1,494      |
| 5    | Brazil      | 1,480      |
| 6    | Colombia    | 1,480      |
| 7    | Netherlands | 1,447      |
| 8    | Portugal    | 1,443      |
| 9    | Ecuador     | 1,439      |
| 10   | Germany     | 1,425      |

---

### Module 2: Historical EDA with Elo

**File:** `notebooks/module2_eda.ipynb`

Deep historical analysis using Elo ratings as the quality measure rather than raw win rates. This produces richer insights than basic statistics because Elo accounts for the strength of opposition.

**6 analyses with charts:**

1. Home advantage decay: trend lines and gap analysis by decade
2. Goal trends: average scoring across 16 decades
3. Dominance eras: which team had the highest Elo per decade and by how much
4. Tournament tier analysis: upset rates, home advantage and goals by competition type
5. Most active nations: match counts vs win rates with scatter plot
6. Head-to-head rivalries: win/draw/loss breakdown for 15 classic matchups

---

### Module 4: ML Match Prediction

**File:** `notebooks/module4_ml.ipynb`

An XGBoost classifier that predicts match outcomes (home win, draw or away win) with calibrated probabilities.

**18 features across 5 categories:**

| Category     | Features                                                         |
| ------------ | ---------------------------------------------------------------- |
| Elo strength | home_elo_pre, away_elo_pre, elo_diff, exp_home_win_prob          |
| Recent form  | Rolling win and draw rate over last 5 and 10 matches             |
| Goal scoring | Rolling average goals scored and conceded over last 10 matches   |
| Head-to-head | Historical H2H win rate, total H2H matches played                |
| Context      | Tournament tier, neutral venue flag, rest advantage, match month |

**Model comparison on test set (2015 to 2026):**

| Model                           | Accuracy | Log-loss | Brier Score |
| ------------------------------- | -------- | -------- | ----------- |
| Naive (always predict home win) | ~45%     | N/A      | N/A         |
| Logistic Regression (baseline)  | ~52%     | ~1.01    | ~0.21       |
| XGBoost                         | ~54%     | ~0.98    | ~0.20       |

**Why time-based split?** The model is trained on pre-2015 data and tested on 2015 onwards. A random split would leak future match results into training, artificially inflating accuracy. The time split simulates real-world deployment where you only ever know the past.

**Top predictive features by XGBoost gain-based importance:**

1. elo_diff: the team quality gap is the strongest single predictor
2. exp_home_win_prob: Elo-derived win probability
3. home_win_rate_5: recent home form over last 5 games
4. away_win_rate_5: recent away form over last 5 games
5. h2h_home_win_rate: historical head-to-head record

---

### Module 5: Streamlit Dashboard

**File:** `notebooks/module5_dashboard.ipynb`

An interactive 6-page web application that makes all findings accessible to any audience without requiring Python knowledge.

| Page            | Content                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| Overview        | KPIs, match volume chart, home advantage decay, key findings               |
| Team Explorer   | Any nation's Elo history, results by decade, biggest wins, goal trends     |
| Head-to-Head    | Full rivalry breakdown between any two nations with complete match history |
| Elo Rankings    | Current world rankings bar chart and multi-team timeline comparison        |
| Match Predictor | Live XGBoost prediction for any fixture with win probabilities             |
| 2026 World Cup  | All predicted fixtures with filterable win probability table               |

---

## Getting Started

### Option 1: Google Colab (recommended)

Each notebook is designed to run in Google Colab with no local setup required.

1. Open the notebook in Colab
2. Run the pip install cell at the top
3. Upload the relevant CSV files from the `dataset/` folder to `/content/`
4. Run all cells in order

### Option 2: Run locally

```bash
# Clone the repository
git clone https://github.com/Sk47R/International_Football_Match_Analysis.git
cd International_Football_Match_Analysis

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order using Jupyter
jupyter notebook

# Launch the dashboard
streamlit run module5_dashboard.py
```

**Run notebooks in this order:**

1. module1_pipeline.ipynb
2. raw_eda_visualization.ipynb
3. module3_elo.ipynb
4. module2_eda.ipynb
5. module4_ml.ipynb
6. module5_dashboard.ipynb

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

---

## Key Findings

| Metric                               | Value                                            |
| ------------------------------------ | ------------------------------------------------ |
| Total matches analysed               | 49,215                                           |
| Date range                           | 1872 to 2026                                     |
| Unique nations                       | 333                                              |
| Total goals scored                   | 144,527                                          |
| Average goals per match              | 2.94                                             |
| Home win rate                        | 49.0%                                            |
| Away win rate                        | 28.3%                                            |
| Draw rate                            | 22.8%                                            |
| Most common scoreline                | 1-0 (5,079 matches)                              |
| Highest scoring match                | Australia 31-0 American Samoa (2001)             |
| Most matches played by any nation    | Sweden (1,099)                                   |
| Highest win rate among major nations | Brazil (63.4%)                                   |
| Peak Elo rating ever recorded        | Spain approx 1,640 (2012)                        |
| ML model accuracy                    | 54% (vs 33% random baseline, 45% naive baseline) |
| Elo favourite win rate               | 72.3%                                            |

---

## Pipeline Architecture

```
Raw CSVs (4 files, 1872 to 2026)
              |
              v
    Module 1: Data Pipeline
    pandas, SQLite, Feature Engineering
              |
       +------+-------+
       |              |
       v              v
  Raw EDA         Module 3: Elo Engine
  9 PNG charts    Custom algorithm
  Justifies Elo   49K matches processed
  design choices  in chronological order
       |              |
       |              v
       |        Module 2: EDA with Elo
       |        Historical analysis
       |        Dominance eras
       |              |
       +------+-------+
              |
              v
     Module 4: ML Prediction
     XGBoost, 18 features
     Time-based train/test split
     2026 World Cup predictions
              |
              v
     Module 5: Dashboard
     Streamlit, 6 pages
     Live match predictor
```

---

## Acknowledgements

- Dataset: [Mart Jurisoo](https://github.com/martj42/international_results) for maintaining the international football results dataset
- Data sources: Wikipedia, rsssf.com and individual football association websites
- Elo methodology: Inspired by [FiveThirtyEight Soccer Power Index](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/) and [Club Elo](http://clubelo.com/System)

---

## License

MIT License. The underlying dataset is maintained by Mart Jurisoo. See the [source repository](https://github.com/martj42/international_results) for dataset licensing terms.

---

If you found this project useful, consider giving it a star on GitHub.
