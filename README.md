<![CDATA[<div align="center">

# 🏏 IPL Match Predictor & Analytics Suite

### Machine Learning-powered IPL match outcome prediction with Monte Carlo simulation and interactive analytics

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-006600)](https://xgboost.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

---

**Predict match winners, target scores, and victory margins** for any IPL matchup using ensemble ML models trained on 17 seasons (2008–2024) of real IPL data. Explore historical analytics through an interactive dashboard with head-to-head heatmaps, toss advantage trends, and more.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Feature Engineering](#feature-engineering)
  - [Models Compared](#models-compared)
  - [Multi-Output Regression](#multi-output-regression)
  - [Winner Classification](#winner-classification)
  - [Model Persistence](#model-persistence)
- [Monte Carlo Simulation Engine](#-monte-carlo-simulation-engine)
- [Application Interfaces](#-application-interfaces)
  - [Full Streamlit App (Primary)](#1-full-streamlit-app-primary---app_streamlit_fullpy)
  - [Alternate Interfaces](#2-alternate-interfaces)
- [Analytics Dashboard](#-analytics-dashboard)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Deep Dive](#-technical-deep-dive)
- [Limitations & Future Work](#-limitations--future-work)
- [License](#-license)

---

## 🌟 Overview

This project is an end-to-end machine learning system that:

1. **Trains** multiple ensemble regression and classification models on historical IPL match data (2008–2024)
2. **Predicts** three key outcomes for any hypothetical IPL match:
   - **Target score** (first innings total in runs)
   - **Victory margin** (runs or wickets, contextually interpreted based on toss decision)
   - **Match winner** (with calibrated win probabilities)
3. **Simulates** match uncertainty via a **Monte Carlo engine** that runs hundreds of perturbed scenarios
4. **Visualises** historical IPL analytics through an interactive dashboard with Plotly charts

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Comparison** | Trains XGBoost, Random Forest, and Gradient Boosting regressors, auto-selects the best |
| 🎯 **Dual Prediction Heads** | Multi-output regression (target runs + margin) and binary classification (winner) |
| 🎲 **Monte Carlo Simulation** | 50–2,000 perturbed simulations to quantify prediction uncertainty |
| 📊 **Interactive Analytics** | Head-to-head heatmaps, toss advantage trends, home/away splits, result distributions |
| 🔧 **Feature Engineering** | Team win ratios, home advantage detection, toss advantage flags, temporal features |
| 📈 **Rich Visualisations** | Histograms with boxplots, donut charts, line trends, stacked bars, heatmaps via Plotly |
| 🚀 **Multiple Interfaces** | Full Streamlit app, basic Streamlit, Flask API, CLI — choose your preferred UX |
| 📤 **CSV Upload Support** | Upload your own IPL CSV to refresh analytics on new data |

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IPL PREDICTOR SYSTEM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  IPL Dataset  │───▶│  ipl.py (Train)   │───▶│  Saved Pipelines  │  │
│  │  2008–2024    │    │  Feature Eng.     │    │  .joblib files    │  │
│  │  (1,095 rows) │    │  Model Compare    │    │                   │  │
│  └──────────────┘    │  Best Selection   │    │  best_pipeline    │  │
│                       └──────────────────┘    │  winner_pipeline  │  │
│                                                └────────┬──────────┘  │
│                                                         │             │
│                              ┌──────────────────────────┘             │
│                              ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                    Serving Layer                               │   │
│  │                                                               │   │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐   │   │
│  │  │  Streamlit Full  │  │  Flask API    │  │  CLI Predict   │   │   │
│  │  │  (Primary App)   │  │  (Alternate)  │  │  (Alternate)   │   │   │
│  │  │                  │  │              │  │                │   │   │
│  │  │ • Predict+Sim    │  │ • REST Form  │  │ • Terminal I/O │   │   │
│  │  │ • Monte Carlo    │  │ • JSON resp  │  │ • Quick test   │   │   │
│  │  │ • Analytics      │  │              │  │                │   │   │
│  │  │ • Rich Charts    │  │              │  │                │   │   │
│  │  └─────────────────┘  └──────────────┘  └────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Dataset

**File:** `IPL_2008-2024.csv`  
**Records:** 1,095 matches (after header)  
**Span:** 17 IPL seasons (2008 → 2024)

### Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique match identifier |
| `season` | int | IPL season year |
| `city` | str | City where the match was played |
| `date` | date | Match date (YYYY-MM-DD) |
| `match_type` | str | `League`, `Playoff`, `Qualifier`, `Eliminator`, `Final` |
| `player_of_match` | str | Player of the match |
| `venue` | str | Stadium name |
| `team1` | str | First team listed |
| `team2` | str | Second team listed |
| `toss_winner` | str | Team that won the toss |
| `toss_decision` | str | `bat` or `field` |
| `winner` | str | Match winner |
| `result` | str | Win type — `runs`, `wickets`, or other |
| `result_margin` | mixed | Numeric margin of victory |
| `target_runs` | int | First innings total + 1 (target for chasing team) |
| `target_overs` | float | Overs available for chasing team |
| `super_over` | str | `Y` / `N` — whether match went to a super over |
| `method` | str | `D/L` if Duckworth-Lewis was applied, else `NA` |
| `umpire1` | str | On-field umpire 1 |
| `umpire2` | str | On-field umpire 2 |

---

## 🧠 Machine Learning Pipeline

The training pipeline is implemented in [`ipl.py`](ipl.py) and follows a structured workflow:

### Feature Engineering

The raw CSV is processed into a rich feature set before model training:

```
Raw CSV Data
    │
    ├── Filter: Remove super-over matches (super_over == 'N')
    ├── Parse: Convert date → datetime, extract year, month
    ├── Clean: Drop rows with missing target_runs or result_margin
    │
    ├── Compute Team Strength:
    │   ├── team1_win_ratio = wins / total_matches (historical)
    │   └── team2_win_ratio = wins / total_matches (historical)
    │
    ├── Home Advantage Detection:
    │   ├── team1_is_home = 1 if team1's city name appears in venue string
    │   └── team2_is_home = 1 if team2's city name appears in venue string
    │
    └── Toss Advantage:
        └── toss_advantage = 1 if toss_winner == match_winner, else 0
```

**Final Feature Set:**

| Type | Features |
|---|---|
| **Categorical** (7) | `city`, `match_type`, `team1`, `team2`, `toss_winner`, `toss_decision`, `venue` |
| **Numerical** (8) | `season`, `year`, `month`, `team1_win_ratio`, `team2_win_ratio`, `team1_is_home`, `team2_is_home`, `toss_advantage` |

**Preprocessing Pipeline:**
- Categorical features → `OneHotEncoder` (with `handle_unknown='ignore'`)
- Numerical features → `StandardScaler`
- Combined via `ColumnTransformer`

### Models Compared

Three ensemble methods are trained and evaluated via an 80/20 train-test split:

| Model | Regressor | Hyperparameters |
|---|---|---|
| **XGBoost** | `XGBRegressor` | `n_estimators=200`, `max_depth=7`, `learning_rate=0.05` |
| **Random Forest** | `RandomForestRegressor` | `n_estimators=200`, `max_depth=15` |
| **Gradient Boosting** | `GradientBoostingRegressor` | `n_estimators=200`, `max_depth=7`, `learning_rate=0.05` |

### Multi-Output Regression

Each model is wrapped in `MultiOutputRegressor` to jointly predict two targets:

1. **`target_runs`** — First innings total (what the chasing team needs)
2. **`result_margin_numeric`** — How many runs/wickets the match was won by

**Evaluation Metrics (per model, per target):**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

The best model is selected by **highest average R²** across both outputs.

### Winner Classification

A separate **Random Forest Classifier** (`n_estimators=200`, `max_depth=15`) predicts the binary outcome:
- `1` → Team 1 wins
- `0` → Team 2 wins

This model also outputs **calibrated win probabilities** via `predict_proba()`.

**Classification Metrics:** Accuracy, Precision, Recall, F1 Score

### Model Persistence

Both trained pipelines (preprocessor + model) are serialized using `joblib`:

| File | Size | Contents |
|---|---|---|
| `best_pipeline.joblib` | ~15 MB | Full sklearn Pipeline: ColumnTransformer → MultiOutputRegressor (best of 3) |
| `winner_pipeline.joblib` | ~4 MB | Full sklearn Pipeline: ColumnTransformer → RandomForestClassifier |

---

## 🎲 Monte Carlo Simulation Engine

The simulation engine adds **uncertainty quantification** to predictions by running many perturbed scenarios. This is critical because a single-point prediction hides the inherent randomness in cricket.

### How It Works

```
Base Match Configuration
         │
         ▼
    ┌─── Loop N times (50–2,000) ───┐
    │                                │
    │  1. Perturb win_ratios:        │
    │     • If zero → seed with      │
    │       small random baseline    │
    │     • Add Gaussian noise       │
    │       (scale = noise_level)    │
    │                                │
    │  2. Random toss flips:         │
    │     • 12% chance: flip bat↔field│
    │     • 5% chance: flip toss     │
    │       winner to other team     │
    │                                │
    │  3. Predict with both models   │
    └────────────────────────────────┘
         │
         ▼
    Aggregate Results:
    ├── Target distribution (mean, median, std, percentiles)
    ├── Margin distribution (mean, median, std, percentiles)
    ├── Win counts per team (→ simulated win %)
    ├── Mean win probability for Team 1
    └── Feature importances (top 20, extracted from pipeline)
```

### Parameters

| Parameter | Range | Default | Effect |
|---|---|---|---|
| `n` (simulations) | 50–2,000 | 500 | More = smoother distributions, slower |
| `noise_level` | 0.0–0.2 | 0.02 | Higher = wider spread in predictions |

### Output Visualisations

- **Histogram + Box Plot** — Target runs distribution (5th–95th percentile trimmed view with mean/median lines)
- **Histogram + Box Plot** — Margin distribution
- **Donut Chart** — Simulated win percentage split
- **Horizontal Bar** — Top 20 feature importances (extracted from pipeline internals)

---

## 🖥 Application Interfaces

### 1. Full Streamlit App (Primary) — `app_streamlit_full.py`

The **main application** with two modes accessible via sidebar radio buttons:

#### Mode A: Predict & Simulate

| Section | Description |
|---|---|
| **Team Selection** | Dropdowns populated from CSV data (or fallback list of 8 teams) |
| **Match Config** | Venue (text), toss decision (bat/field), season, month |
| **Quick Predict** | Single-point prediction — target, margin, winner, win probabilities |
| **Monte Carlo** | Configurable simulations (50–2,000) with noise control |
| **Results** | Single-point metrics + simulation summary stats (JSON) |
| **Win Donut** | Interactive Plotly donut chart of simulated win split |
| **Distributions** | Dual histogram+boxplot figures for target and margin |
| **Feature Importance** | Horizontal bar chart of top contributing features |

#### Mode B: Analytics Dashboard

A comprehensive **historical analytics** view (details in [Analytics Dashboard](#-analytics-dashboard) section below).

**Run it:**
```bash
streamlit run app_streamlit_full.py
```

---

### 2. Alternate Interfaces

Located in the [`Alternates/`](Alternates/) directory, these provide simpler or different ways to interact with the same trained models:

#### a) Basic Streamlit — `Alternates/app_streamlit.py`
Minimal Streamlit UI with text inputs for teams/venue, a single "Predict" button, and clean output of target, margin, and win probability. No Monte Carlo, no analytics.

```bash
streamlit run Alternates/app_streamlit.py
```

#### b) Analytics Streamlit — `Alternates/app_streamlit_analytics.py`
Mid-tier version with sidebar inputs, Monte Carlo simulation, distribution histograms, feature importances, and a donut chart — but **without** the full analytics dashboard tab.

```bash
streamlit run Alternates/app_streamlit_analytics.py
```

#### c) Flask Web App — `Alternates/app_flask.py`
Lightweight Flask server with an HTML form. Submit team names and venue, get predictions rendered as simple HTML. Good for embedding or API-style usage.

```bash
python Alternates/app_flask.py
# → Opens at http://127.0.0.1:5000
```

#### d) CLI Predictor — `Alternates/cli_predict.py`
Interactive terminal-based predictor. Prompts for team names, venue, and toss decision via `input()`, prints formatted prediction to stdout.

```bash
python Alternates/cli_predict.py
```

---

## 📊 Analytics Dashboard

The analytics dashboard (Mode B in the full app) provides deep historical insights from the IPL dataset:

### Visualisations Included

| # | Chart | Type | Description |
|---|---|---|---|
| 1 | **Top-Level Stats** | Metrics | Total matches, average target runs, average result margin |
| 2 | **Model R² Comparison** | Bar Chart | R² scores for saved model vs. baseline (mean predictor) — computed on-demand or loaded from `model_results.json` |
| 3 | **Home vs Away Counts** | Donut Chart | Overall proportion of home vs away matches (heuristic: team1's city name in venue) |
| 4 | **Target Runs Trend** | Line Chart | Average target runs per season — reveals scoring inflation/deflation over time |
| 5 | **Result Margins Distribution** | Histogram + Boxplot | Spread of victory margins with trimming controls (5–95% or full range toggle) |
| 6 | **Head-to-Head Heatmap** | Heatmap | Win rate matrix: every team vs every opponent across all historical matches |
| 7 | **Home vs Away Wins per Team** | Grouped Bar | Bars showing each team's home wins vs away wins side by side |
| 8 | **Result Type per Year** | Stacked Bar | Yearly breakdown: wins by runs vs wins by wickets vs other outcomes |
| 9 | **Bat-First Advantage** | Line Chart + Stat | Historical bat-first win percentage trend across seasons |

### Data Handling

- **Default:** Loads `IPL_2008-2024.csv` from the working directory
- **Upload:** Users can upload a custom CSV via the sidebar file uploader
- **Auto-parsing:** Dates are parsed, `year` and `month` are extracted automatically
- **Missing columns:** Charts gracefully degrade with informational messages if expected columns are absent

---

## ⚙ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/shaawtymaker/IPL-PREDICTOR-.git
cd IPL-PREDICTOR-
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit plotly flask
```

<details>
<summary><strong>Full dependency list with roles</strong></summary>

| Package | Version | Role |
|---|---|---|
| `pandas` | ≥1.5 | Data loading, manipulation, feature engineering |
| `numpy` | ≥1.23 | Numerical operations, Monte Carlo noise generation |
| `scikit-learn` | ≥1.2 | ML pipelines, preprocessing, models, metrics |
| `xgboost` | ≥1.7 | XGBRegressor for multi-output regression |
| `joblib` | ≥1.2 | Model serialization/deserialization |
| `streamlit` | ≥1.28 | Interactive web UI framework |
| `plotly` | ≥5.15 | Interactive charts (histograms, heatmaps, pies, bars) |
| `flask` | ≥2.3 | Lightweight web server (alternate interface) |

</details>

### 3. Train the Models (Optional — pre-trained models included)

If you want to retrain from scratch:

```bash
python ipl.py
```

This will:
1. Load and preprocess `IPL_2008-2024.csv`
2. Engineer features (win ratios, home flags, toss advantage)
3. Train 3 regression models + 1 classification model
4. Print comparative metrics for all models
5. Save `best_pipeline.joblib` and `winner_pipeline.joblib`

### 4. Launch the Application

```bash
streamlit run app_streamlit_full.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 🚀 Usage

### Quick Prediction

1. Launch the app → select **"Predict & Simulate"** mode
2. Choose **Team 1** and **Team 2** from the dropdowns
3. Enter the **venue** name
4. Set **toss decision** (bat/field), **season**, and **month**
5. Click **"Quick predict (single)"** for instant results

### Monte Carlo Simulation

1. Adjust the **Simulations** slider (50–2,000)
2. Set the **Noise level** (0.0–0.2) to control how widely scenarios are perturbed
3. Click **"Predict & Run Simulations"**
4. Explore the donut chart, distribution histograms, and feature importance chart

### Analytics Dashboard

1. Switch to **"Analytics Dashboard"** mode via the sidebar
2. Ensure `IPL_2008-2024.csv` is in the working directory (or upload one)
3. Scroll through the suite of interactive charts
4. Toggle **"Show full range"** on the margins distribution for unclipped view
5. Click **"Compute R² now"** to evaluate model performance against the dataset

### CLI Quick Test

```bash
python Alternates/cli_predict.py
```
```
IPL MATCH PREDICTOR – CLI MODE
Enter Team 1: Mumbai Indians
Enter Team 2: Chennai Super Kings
Enter Venue: Wankhede Stadium
Toss Decision (bat/field): field

----- MATCH PREDICTION -----
Match: Mumbai Indians vs Chennai Super Kings
Venue: Wankhede Stadium
Predicted Target Score: 168
Mumbai Indians is predicted to win by 4 wickets (~24 balls left).

Win Probability:
Mumbai Indians: 62.3%
Chennai Super Kings: 37.7%
------------------------------
```

---

## 📁 Project Structure

```
IPL-PREDICTOR-/
│
├── ipl.py                          # 🧠 Model training script
│   ├── Data loading & cleaning
│   ├── Feature engineering (win ratios, home flags, toss advantage)
│   ├── 3-model comparison (XGBoost, RF, GBR)
│   ├── Winner classification model (RF Classifier)
│   ├── Example predictions
│   └── Model serialization (joblib.dump)
│
├── app_streamlit_full.py           # 🖥️ Primary application (635 lines)
│   ├── Model & CSV loading (cached)
│   ├── Match row builder
│   ├── Monte Carlo simulation engine
│   ├── Distribution plotting helper (histogram + boxplot)
│   ├── Predict & Simulate tab
│   └── Analytics Dashboard tab (9 chart types)
│
├── IPL_2008-2024.csv               # 📊 Dataset (1,095 matches, 20 columns)
├── best_pipeline.joblib            # 🤖 Trained regression pipeline (~15 MB)
├── winner_pipeline.joblib          # 🤖 Trained classification pipeline (~4 MB)
│
└── Alternates/                     # 🔄 Alternative interfaces
    ├── app_streamlit.py            #    Basic Streamlit predictor (59 lines)
    ├── app_streamlit_analytics.py  #    Mid-tier with Monte Carlo (282 lines)
    ├── app_flask.py                #    Flask web form (74 lines)
    └── cli_predict.py              #    Terminal CLI predictor (71 lines)
```

---

## 🔬 Technical Deep Dive

### Preprocessing Pipeline (sklearn)

```python
ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
           ['city', 'match_type', 'team1', 'team2', 'toss_winner',
            'toss_decision', 'venue']),
    ('num', StandardScaler(),
           ['season', 'year', 'month', 'team1_win_ratio', 'team2_win_ratio',
            'team1_is_home', 'team2_is_home', 'toss_advantage'])
])
```

The `OneHotEncoder` with `handle_unknown='ignore'` ensures that new teams or venues not seen during training don't crash inference — they simply get a zero vector for that categorical slot.

### Home Advantage Heuristic

```python
team1_is_home = 1 if team1.split()[0].lower() in venue.lower() else 0
```

This checks if the **first word** of the team name (e.g., "Mumbai" from "Mumbai Indians") appears in the venue string (e.g., "Wankhede Stadium" — does NOT match, but "DY Patil Stadium, Mumbai" would). This is a simple heuristic and may not always be accurate, but provides a useful signal.

### Margin Interpretation Logic

The predicted margin is contextually interpreted based on the toss decision:

- **Toss = bat** → Team batting first set a target → margin is in **runs**
- **Toss = field** → Team bowling first defended → margin is in **wickets** (clamped 1–10, with approximate balls-left estimate at `wickets × 6`)

### Streamlit Caching Strategy

| Decorator | Used For | TTL |
|---|---|---|
| `@st.cache_resource` | Model loading (heavy `.joblib` files) | Permanent (app lifetime) |
| `@st.cache_data` | CSV parsing, team list extraction | Permanent |
| `@st.cache_data(ttl=3600)` | R² computation (expensive) | 1 hour |

### Distribution Plotting Helper

The `plot_distribution_with_box()` function creates publication-quality Plotly figures with:
- **Freedman-Diaconis binning** — automatic optimal bin count based on IQR
- **Percentile trimming** — removes extreme outliers (configurable, default 5th–95th)
- **Dual-panel layout** — histogram on top (78% height) + boxplot on bottom (22%)
- **Mean/median lines** — orange dashed (mean) and green dotted (median) vertical reference lines
- **Outlier annotation** — transparent overlay showing how many points were clipped

---

## ⚠ Limitations & Future Work

### Current Limitations

- **No ball-by-ball data** — Predictions are match-level only; no over-by-over or ball-by-ball modeling
- **Home advantage heuristic** — Simple string matching; doesn't account for neutral venues or franchise relocations
- **Static team strength** — Win ratios computed across all historical data, not season-specific or form-weighted
- **No player-level features** — Team composition, player form, injuries, and playing XI are not modeled
- **Toss advantage is post-hoc** — `toss_advantage` flag uses actual match winner during training (potential leakage for that specific feature)

### Potential Enhancements

- [ ] **Rolling team form** — Compute win ratios over sliding windows (last N matches)
- [ ] **Player impact scores** — Incorporate key player availability and form
- [ ] **Venue-specific models** — Train separate models per venue cluster for ground-specific behavior
- [ ] **Live score integration** — Extend to live match prediction with ball-by-ball streaming data
- [ ] **Hyperparameter tuning** — Add GridSearchCV or Optuna for systematic hyperparameter optimization
- [ ] **Model explainability** — Integrate SHAP values for individual prediction explanations
- [ ] **Docker deployment** — Containerize for one-command cloud deployment
- [ ] **API endpoint** — RESTful API with FastAPI for programmatic access

---

## 📜 License

This project is open source and available for educational and research purposes.

---

<div align="center">

**Built with ❤️ for cricket analytics**

*If you find this project useful, consider giving it a ⭐ on GitHub!*

</div>
]]>
