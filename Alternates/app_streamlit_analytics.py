# app_streamlit_analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import random
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="IPL Predictor — Analytics", layout="wide")

# --- Config / paths ---
MODEL_BP = "best_pipeline.joblib"
MODEL_WP = "winner_pipeline.joblib"
CSV_FILE = "IPL_2008-2024.csv"

# --- Helper: load models (cached) ---
@st.cache_resource
def load_models(bp_path=MODEL_BP, wp_path=MODEL_WP):
    if not Path(bp_path).exists() or not Path(wp_path).exists():
        raise FileNotFoundError(f"Missing model files: {bp_path} or {wp_path}. Run your training script to save them.")
    bp = joblib.load(bp_path)
    wp = joblib.load(wp_path)
    return bp, wp

# --- Helper: load teams (cached) ---
@st.cache_data
def load_teams(csv_path=CSV_FILE):
    default_teams = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
        "Rajasthan Royals", "Punjab Kings"
    ]
    p = Path(csv_path)
    if not p.exists():
        return default_teams
    try:
        df = pd.read_csv(p)
        t1 = set(df['team1'].dropna().unique()) if 'team1' in df.columns else set()
        t2 = set(df['team2'].dropna().unique()) if 'team2' in df.columns else set()
        teams = sorted(list(t1.union(t2)))
        return teams if teams else default_teams
    except Exception:
        return default_teams

# --- Build match row consistent with training pipeline ---
def build_match_row(team1, team2, venue, city='Unknown', season=2024, year=2024, month=4, toss_winner=None, toss_decision='field'):
    if toss_winner is None:
        toss_winner = team1
    return {
        'city': city,
        'match_type': 'League',
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'venue': venue,
        'season': int(season),
        'year': int(year),
        'month': int(month),
        'team1_win_ratio': 0.0,
        'team2_win_ratio': 0.0,
        'team1_is_home': 1 if team1.lower().split()[0] in venue.lower() else 0,
        'team2_is_home': 1 if team2.lower().split()[0] in venue.lower() else 0,
        'toss_advantage': 0
    }

# --- Monte-Carlo simulator (vectorized) ---
def monte_carlo_simulate(best_pipeline, winner_pipeline, base_match, n=500, noise_level=0.02):
    """
    Returns dict:
      - targets: list[int]
      - margins: list[float]
      - winner_counts: {team1: int, team2: int}
      - stats: target/margin stats
      - feature_importances: list (may be empty)
    """
    base = base_match.copy()
    rows = []
    for i in range(n):
        r = base.copy()
        # add small randomness to ratios (if zero, create small baseline so predictions vary)
        for k in ['team1_win_ratio', 'team2_win_ratio']:
            val = float(r.get(k, 0.0))
            if val == 0.0:
                val = 0.01 + random.random()*0.02
            noisy = max(0.0, val + np.random.normal(scale=noise_level*max(0.01,val)))
            r[k] = noisy
        # sometimes flip toss_decision to explore scenarios
        if random.random() < 0.12:
            r['toss_decision'] = 'bat' if r.get('toss_decision','field')=='field' else 'field'
        if random.random() < 0.05:
            r['toss_winner'] = r['team2'] if r.get('toss_winner')==r['team1'] else r['team1']
        rows.append(r)
    df_sim = pd.DataFrame(rows)

    # vectorized prediction
    preds = best_pipeline.predict(df_sim)   # shape (n, 2)
    # winner probabilities
    try:
        wp = winner_pipeline.predict_proba(df_sim)
        wp_choice = winner_pipeline.predict(df_sim)
    except Exception:
        # fallback per-row (slower)
        wp = np.array([winner_pipeline.predict_proba(pd.DataFrame([r]))[0] for _, r in df_sim.iterrows()])
        wp_choice = np.array([winner_pipeline.predict(pd.DataFrame([r]))[0] for _, r in df_sim.iterrows()])

    targets = [int(round(x)) for x in preds[:,0]]
    margins = [float(x) for x in preds[:,1]]
    winners_choices = [int(v) for v in wp_choice]
    team1 = base['team1']; team2 = base['team2']
    cnt_team1 = int(sum(1 for v in winners_choices if v==1))
    cnt_team2 = len(winners_choices) - cnt_team1

    def stats_arr(arr):
        a = np.array(arr, dtype=float)
        return {
            'count': int(a.size),
            'mean': float(np.mean(a)) if a.size else None,
            'median': float(np.median(a)) if a.size else None,
            'std': float(np.std(a, ddof=0)) if a.size else None,
            'p5': float(np.percentile(a,5)) if a.size else None,
            'p25': float(np.percentile(a,25)) if a.size else None,
            'p75': float(np.percentile(a,75)) if a.size else None,
            'p95': float(np.percentile(a,95)) if a.size else None,
        }

    target_stats = stats_arr(targets)
    margin_stats = stats_arr(margins)

    # try to extract feature importances (works if pipeline and model structure same as training)
    feat_imps = []
    try:
        pre = best_pipeline.named_steps['preprocessor']
        model = best_pipeline.named_steps['model']  # MultiOutputRegressor
        # categorical names
        cat_cols = pre.transformers_[0][2]
        ohe = pre.transformers_[0][1]
        try:
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            cat_names = list(ohe.get_feature_names(cat_cols))
        num_cols = pre.transformers_[1][2]
        feature_names = list(cat_names) + list(num_cols)
        imps = np.array([est.feature_importances_ for est in model.estimators_])
        mean_imp = np.mean(imps, axis=0)
        pairs = sorted(zip(feature_names, mean_imp), key=lambda x: x[1], reverse=True)
        feat_imps = [{'feature': f, 'importance': float(v)} for f,v in pairs[:20]]
    except Exception:
        feat_imps = []

    return {
        'targets': targets,
        'margins': margins,
        'winner_counts': {team1: cnt_team1, team2: cnt_team2},
        'winner_mean_prob_team1': float(np.mean([p[1] for p in wp])),
        'stats': {'target': target_stats, 'margin': margin_stats},
        'feature_importances': feat_imps
    }

# ---- UI ----
st.title("🏏 IPL Predictor — Visual Analytics")

# load models and teams
try:
    best_pipeline, winner_pipeline = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

teams = load_teams()

# Sidebar inputs
with st.sidebar:
    st.header("Match settings")
    team1 = st.selectbox("Team 1", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
    team2 = st.selectbox("Team 2", teams, index=teams.index("Chennai Super Kings") if "Chennai Super Kings" in teams else (1 if len(teams)>1 else 0))
    venue = st.text_input("Venue", "Wankhede Stadium")
    toss_decision = st.selectbox("Toss Decision", ["field","bat"])
    season = st.number_input("Season", min_value=2008, max_value=2050, value=2024)
    month = st.slider("Month", 1, 12, 4)
    n_sim = st.slider("Monte-Carlo simulations", min_value=50, max_value=2000, value=500, step=50)
    noise_level = st.slider("Simulation noise (how widely to perturb features)", 0.0, 0.2, 0.02, 0.01)
    run_button = st.button("Predict & Simulate")

# Build base match
base_match = build_match_row(team1, team2, venue, season=season, year=season, month=month, toss_decision=toss_decision)

# Single prediction (instant)
if st.button("Quick predict (single)"):
    df = pd.DataFrame([base_match])
    try:
        pred = best_pipeline.predict(df)[0]
        proba = winner_pipeline.predict_proba(df)[0]
        winner_flag = int(winner_pipeline.predict(df)[0])
        predicted_winner = team1 if winner_flag==1 else team2
        st.metric("Predicted target (runs)", int(round(pred[0])))
        if base_match['toss_decision']=='bat':
            st.write(f"Predicted margin: **{predicted_winner}** by **{abs(int(round(pred[1])))} runs**")
        else:
            wk = min(max(int(abs(round(pred[1]))),1),10)
            st.write(f"Predicted margin: **{predicted_winner}** by **{wk} wickets** (~{wk*6} balls left)")
        st.write(f"Win probability — {team1}: {proba[1]*100:.1f}%, {team2}: {proba[0]*100:.1f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# Main pane: run MC simulation and show charts
if run_button:
    with st.spinner(f"Running {n_sim} simulations..."):
        sim = monte_carlo_simulate(best_pipeline, winner_pipeline, base_match, n=n_sim, noise_level=noise_level)

    # show single-point (recompute so it's available)
    df = pd.DataFrame([base_match])
    pred = best_pipeline.predict(df)[0]
    proba = winner_pipeline.predict_proba(df)[0]
    winner_flag = int(winner_pipeline.predict(df)[0])
    predicted_winner = team1 if winner_flag==1 else team2

    # top row: single prediction + doughnut
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Single-point prediction")
        st.write(f"**Predicted target:** {int(round(pred[0]))} runs")
        if base_match['toss_decision']=='bat':
            st.write(f"**Predicted margin:** {predicted_winner} by {abs(int(round(pred[1])))} runs")
        else:
            wk = min(max(int(abs(round(pred[1]))),1),10)
            st.write(f"**Predicted margin:** {predicted_winner} by {wk} wickets (~{wk*6} balls left)")
        st.write(f"**Win probability:** {team1}: {proba[1]*100:.1f}%, {team2}: {proba[0]*100:.1f}%")
        st.markdown("**Summary stats (from simulations)**")
        tstats = sim['stats']['target']
        mstats = sim['stats']['margin']
        st.write({
            "target_mean": round(tstats['mean'],1),
            "target_median": round(tstats['median'],1),
            "target_std": round(tstats['std'],1),
            "margin_mean": round(mstats['mean'],1),
            "margin_median": round(mstats['median'],1)
        })

    with col2:
        st.subheader("Win % (sim)")
        counts = sim['winner_counts']
        labels = [team1, team2]
        values = [counts[team1], counts[team2]]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, sort=False,
                                     marker=dict(colors=['#0d6efd','#198754']))])
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distributions (simulations)")

    c1, c2 = st.columns(2)
    with c1:
        df_targets = pd.DataFrame({'target': sim['targets']})
        fig_t = px.histogram(df_targets, x='target', nbins=30, title='Predicted Target (runs)')
        fig_t.update_layout(margin=dict(t=40,b=20))
        st.plotly_chart(fig_t, use_container_width=True)
    with c2:
        df_m = pd.DataFrame({'margin': sim['margins']})
        fig_m = px.histogram(df_m, x='margin', nbins=30, title='Predicted Margin')
        fig_m.update_layout(margin=dict(t=40,b=20))
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Feature importances (if available)")
    if sim['feature_importances']:
        fi = sim['feature_importances']
        df_fi = pd.DataFrame(fi)
        df_fi['importance_pct'] = df_fi['importance'] / df_fi['importance'].sum() * 100 if df_fi['importance'].sum()>0 else 0
        fig_f = px.bar(df_fi, x='importance_pct', y='feature', orientation='h', height=400, title='Top features')
        fig_f.update_layout(margin=dict(t=40,b=20))
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Feature importances not available for this model/pipeline.")

    st.success("Simulation complete.")
else:
    st.info("Fill the settings in the sidebar and click **Predict & Simulate** to run Monte-Carlo and see visual analytics.")
