# app_streamlit_full.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
import json
import os

st.set_page_config(page_title="IPL Predictor + Analytics", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# Config / paths
# ----------------------------
MODEL_BP = "best_pipeline.joblib"
MODEL_WP = "winner_pipeline.joblib"
DEFAULT_CSV = "IPL_2008-2024.csv"   # default filename to look for in working dir
MODEL_RESULTS_JSON = "model_results.json"

# ----------------------------
# Utilities: load models and CSV
# ----------------------------
@st.cache_resource
def load_models(bp_path=MODEL_BP, wp_path=MODEL_WP):
    bp = None; wp = None
    if Path(bp_path).exists() and Path(wp_path).exists():
        bp = joblib.load(bp_path)
        wp = joblib.load(wp_path)
    return bp, wp

@st.cache_data
def load_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    # parse dates if possible
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
        except Exception:
            pass
    return df

# ----------------------------
# Basic domain helpers
# ----------------------------
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

# Monte-Carlo simulator (vectorized)
def monte_carlo_simulate(best_pipeline, winner_pipeline, base_match, n=500, noise_level=0.02):
    base = base_match.copy()
    rows = []
    for i in range(n):
        r = base.copy()
        for k in ['team1_win_ratio', 'team2_win_ratio']:
            val = float(r.get(k, 0.0))
            if val == 0.0:
                val = 0.01 + random.random()*0.02
            noisy = max(0.0, val + np.random.normal(scale=noise_level*max(0.01,val)))
            r[k] = noisy
        if random.random() < 0.12:
            r['toss_decision'] = 'bat' if r.get('toss_decision','field')=='field' else 'field'
        if random.random() < 0.05:
            r['toss_winner'] = r['team2'] if r.get('toss_winner')==r['team1'] else r['team1']
        rows.append(r)
    df_sim = pd.DataFrame(rows)
    preds = best_pipeline.predict(df_sim)
    try:
        wp = winner_pipeline.predict_proba(df_sim)
        wp_choice = winner_pipeline.predict(df_sim)
    except Exception:
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
    feat_imps = []
    try:
        pre = best_pipeline.named_steps['preprocessor']
        model = best_pipeline.named_steps['model']
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

# ----------------------------
# Distribution plotting helper
# ----------------------------
def plot_distribution_with_box(title, arr, label="value", trim_pct=(5,95), nbins=None):
    """
    Returns a Plotly figure with a histogram (top) and a boxplot (bottom).
    - arr: 1D list/np.array of numeric values
    - trim_pct: tuple (low_pct, high_pct) percentiles used if trimming
    - nbins: number of bins; if None, uses Freedman-Diaconis rule
    """
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)", margin=dict(t=30,b=20))
        return fig

    # compute trimming percentiles
    p_low, p_high = np.percentile(a, trim_pct)
    trimmed = a[(a >= p_low) & (a <= p_high)]
    trimmed_count = trimmed.size
    outliers_low = int(np.sum(a < p_low))
    outliers_high = int(np.sum(a > p_high))
    outliers_total = outliers_low + outliers_high

    # determine nbins via Freedman-Diaconis if not provided
    if nbins is None:
        q75, q25 = np.percentile(a, [75,25])
        iqr = q75 - q25
        n = max(1, len(a))
        if iqr == 0:
            nbins = int(np.sqrt(n))
        else:
            bw = 2 * iqr * (n ** (-1/3))
            if bw <= 0:
                nbins = int(np.sqrt(n))
            else:
                nbins = max(6, int(np.ceil((a.max() - a.min()) / bw)))
    nbins = max(6, int(nbins))

    mean_all = float(np.mean(a))
    median_all = float(np.median(a))

    # make subplot: histogram (row=1) + boxplot (row=2)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02,
                        specs=[[{"type": "xy"}], [{"type": "xy"}]])

    # Histogram (trimmed for clarity)
    hist = go.Histogram(x=trimmed, nbinsx=nbins, name='Trimmed data', marker_color='#6fb3ff', opacity=0.9)
    fig.add_trace(hist, row=1, col=1)

    # invisible full-data histogram (for axis scaling if needed)
    fig.add_trace(go.Histogram(x=a, nbinsx=nbins, name='Full data (outline)', marker_color='rgba(0,0,0,0)',
                                opacity=0, showlegend=False), row=1, col=1)

    # mean & median vertical lines
    fig.add_vline(x=mean_all, line=dict(color='orange', width=2, dash='dash'), row=1, col=1)
    fig.add_vline(x=median_all, line=dict(color='limegreen', width=2, dash='dot'), row=1, col=1)

    # annotate mean/median values
    fig.add_annotation(x=mean_all, y=0.95, xref="x domain", yref="paper",
                       text=f"mean: {mean_all:.1f}", showarrow=False, font=dict(color='orange'),
                       row=1, col=1)
    fig.add_annotation(x=median_all, y=0.90, xref="x domain", yref="paper",
                       text=f"median: {median_all:.1f}", showarrow=False, font=dict(color='limegreen'),
                       row=1, col=1)

    # boxplot for full data
    fig.add_trace(go.Box(x=a, name='Box (full)', boxpoints='outliers', marker_color='#6fb3ff'), row=2, col=1)

    # layout tweaks
    fig.update_xaxes(title_text=label, row=2, col=1)
    fig.update_yaxes(title_text='count', row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(title=title,
                      bargap=0.02,
                      margin=dict(t=40, b=30, l=50, r=30),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      hovermode='x',
                      height=420)

    # Add an annotation about outliers clipped
    if outliers_total > 0:
        txt = f"Showing {trim_pct[0]}–{trim_pct[1]} percentile ({trimmed_count} points). {outliers_total} clipped ({outliers_low} below, {outliers_high} above)."
    else:
        txt = f"Showing full data ({a.size} points)."
    fig.add_annotation(x=0.01, y=0.98, xref='paper', yref='paper', text=txt, showarrow=False, align='left',
                       bgcolor='rgba(0,0,0,0.4)', font=dict(size=11, color='white'))

    return fig

# ----------------------------
# UI: sidebar controls
# ----------------------------
st.sidebar.title("Controls")
tab = st.sidebar.radio("Mode", ["Predict & Simulate", "Analytics Dashboard"])
# file uploader (CSV)
uploaded = st.sidebar.file_uploader("Upload IPL CSV (optional)", type=["csv"])
if uploaded:
    st.sidebar.success("Using uploaded CSV")
data_path = DEFAULT_CSV if Path(DEFAULT_CSV).exists() else None

# Load data (prefer uploaded file, else default path)
if uploaded:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded CSV: {e}")
        df = None
else:
    if data_path:
        try:
            df = load_csv(data_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load {data_path}: {e}")
            df = None
    else:
        df = None
        st.sidebar.info("No IPL CSV found in working dir. Upload one for analytics.")

# Models
best_pipeline, winner_pipeline = load_models()
if best_pipeline is None or winner_pipeline is None:
    st.sidebar.warning("Model files not found (best_pipeline.joblib / winner_pipeline.joblib). Prediction features disabled.")
# Side: provide team list from CSV if available else simple list
team_list = []
if df is not None:
    if 'team1' in df.columns and 'team2' in df.columns:
        team_list = sorted(set(df['team1'].dropna().unique()).union(set(df['team2'].dropna().unique())))
if not team_list:
    team_list = ["Mumbai Indians","Chennai Super Kings","Royal Challengers Bangalore","Kolkata Knight Riders","Sunrisers Hyderabad","Delhi Capitals","Rajasthan Royals","Punjab Kings"]

# ----------------------------
# Predict & Simulate Tab
# ----------------------------
if tab == "Predict & Simulate":
    st.header("Predict & Simulate")
    st.markdown("Enter match details, get single prediction and Monte-Carlo visualizations.")

    col1, col2 = st.columns([2,1])
    with col1:
        team1 = st.selectbox("Team 1", team_list, index=team_list.index("Mumbai Indians") if "Mumbai Indians" in team_list else 0)
        team2 = st.selectbox("Team 2", team_list, index=team_list.index("Chennai Super Kings") if "Chennai Super Kings" in team_list else (1 if len(team_list)>1 else 0))
        venue = st.text_input("Venue", "Wankhede Stadium")
        toss_decision = st.selectbox("Toss Decision", ["field","bat"])
        season = st.number_input("Season", min_value=2008, max_value=2050, value=2024)
        month = st.slider("Month", 1, 12, 4)
    with col2:
        st.write("Monte-Carlo settings")
        n_sim = st.slider("Simulations", min_value=50, max_value=2000, value=500, step=50)
        noise_level = st.slider("Noise level", 0.0, 0.2, 0.02, 0.01)
        run = st.button("Predict & Run Simulations")

    base_match = build_match_row(team1, team2, venue, season=season, year=season, month=month, toss_decision=toss_decision)
    if st.button("Quick predict (single)"):
        if best_pipeline is None or winner_pipeline is None:
            st.error("Models not loaded. Place best_pipeline.joblib and winner_pipeline.joblib in app folder.")
        else:
            df_single = pd.DataFrame([base_match])
            pred = best_pipeline.predict(df_single)[0]
            proba = winner_pipeline.predict_proba(df_single)[0]
            winner_flag = int(winner_pipeline.predict(df_single)[0])
            predicted_winner = team1 if winner_flag==1 else team2
            st.metric("Predicted target (runs)", int(round(pred[0])))
            if base_match['toss_decision']=='bat':
                st.write(f"Predicted margin: **{predicted_winner}** by **{abs(int(round(pred[1])))} runs**")
            else:
                wk = min(max(int(abs(round(pred[1]))),1),10)
                st.write(f"Predicted margin: **{predicted_winner}** by **{wk} wickets** (~{wk*6} balls left)")
            st.write(f"Win probability — {team1}: {proba[1]*100:.1f}%, {team2}: {proba[0]*100:.1f}%")

    if run:
        if best_pipeline is None or winner_pipeline is None:
            st.error("Models not loaded. Place best_pipeline.joblib and winner_pipeline.joblib in app folder.")
        else:
            with st.spinner(f"Running {n_sim} simulations..."):
                sim = monte_carlo_simulate(best_pipeline, winner_pipeline, base_match, n=n_sim, noise_level=noise_level)
            # single point
            df_single = pd.DataFrame([base_match])
            pred = best_pipeline.predict(df_single)[0]
            proba = winner_pipeline.predict_proba(df_single)[0]
            winner_flag = int(winner_pipeline.predict(df_single)[0])
            predicted_winner = team1 if winner_flag==1 else team2

            # display
            left, right = st.columns([2,1])
            with left:
                st.subheader("Single-point Prediction")
                st.write(f"**Target:** {int(round(pred[0]))} runs")
                if base_match['toss_decision']=='bat':
                    st.write(f"**Margin:** {predicted_winner} by {abs(int(round(pred[1])))} runs")
                else:
                    wk = min(max(int(abs(round(pred[1]))),1),10)
                    st.write(f"**Margin:** {predicted_winner} by {wk} wickets (~{wk*6} balls left)")
                st.write(f"**Win % (single):** {team1}: {proba[1]*100:.1f}%, {team2}: {proba[0]*100:.1f}%")
                st.write("**Simulation summary stats:**")
                st.json(sim['stats'])
            with right:
                st.subheader("Win % (sim)")
                counts = sim['winner_counts']
                fig = go.Figure(data=[go.Pie(labels=[team1,team2], values=[counts[team1],counts[team2]], hole=0.5)])
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})

            # histograms (improved)
            st.subheader("Distributions")
            t1, t2 = st.columns(2)
            with t1:
                fig_t = plot_distribution_with_box("Predicted Target (runs)", sim['targets'], label='target', trim_pct=(5,95))
                st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
            with t2:
                fig_m = plot_distribution_with_box("Predicted Margin", sim['margins'], label='margin', trim_pct=(5,95))
                st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar': False, 'responsive': True})

            st.subheader("Feature importances (if available)")
            if sim['feature_importances']:
                df_fi = pd.DataFrame(sim['feature_importances'])
                df_fi['importance_pct'] = df_fi['importance'] / df_fi['importance'].sum() * 100 if df_fi['importance'].sum()>0 else 0
                fig_f = px.bar(df_fi, x='importance_pct', y='feature', orientation='h', title='Top features')
                fig_f.update_layout(margin=dict(t=30,b=20), height=400)
                st.plotly_chart(fig_f, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
            else:
                st.info("Feature importances not available for this pipeline.")

# ----------------------------
# Analytics Dashboard Tab
# ----------------------------
else:
    st.header("Analytics Dashboard")
    if df is None:
        st.warning("No dataset loaded for analytics. Upload IPL CSV via the sidebar or place 'IPL_2008-2024.csv' in the app folder.")
        st.stop()

    # Clean some expected columns
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].dt.year
    if 'month' not in df.columns and 'date' in df.columns:
        df['month'] = df['date'].dt.month

    st.subheader("Top-level stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches", int(len(df)))
    with col2:
        if 'target_runs' in df.columns:
            st.metric("Avg target runs", round(df['target_runs'].mean(),1))
    with col3:
        if 'result_margin' in df.columns:
            st.metric("Avg result margin", round(pd.to_numeric(df['result_margin'], errors='coerce').mean(),1))

    # 1) Model performance comparison (compute if missing)
    st.markdown("### Models R² comparison")

    # cached helper to compute R2 (so button can reuse cached results)
    @st.cache_data(ttl=3600)
    def compute_r2_from_data(df_local, _bp):
        """
        Compute R² for the saved pipeline on rows with target_runs & result_margin.
        Note: _bp is intentionally prefixed with underscore so Streamlit doesn't attempt to hash it.
        """
        if df_local is None or _bp is None:
            return None
        if 'target_runs' not in df_local.columns or 'result_margin' not in df_local.columns:
            return None
        tmp = df_local.copy()
        tmp['result_margin_numeric'] = pd.to_numeric(tmp['result_margin'], errors='coerce')
        valid = tmp.dropna(subset=['target_runs', 'result_margin_numeric']).copy()
        if len(valid) < 10:
            return None
        cat_cols = ['city', 'match_type', 'team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
        num_cols = ['season', 'year', 'month', 'team1_win_ratio', 'team2_win_ratio',
                    'team1_is_home', 'team2_is_home', 'toss_advantage']
        for c in cat_cols:
            if c not in valid.columns:
                valid[c] = ""
        for c in num_cols:
            if c not in valid.columns:
                valid[c] = 0
        # heuristic home flags if needed
        if valid['team1_is_home'].isnull().any() or (valid['team1_is_home'] == 0).all():
            valid['team1_is_home'] = valid.apply(lambda r: 1 if str(r['venue']).lower().find(str(r['team1']).split()[0].lower()) != -1 else 0, axis=1)
        if valid['team2_is_home'].isnull().any() or (valid['team2_is_home'] == 0).all():
            valid['team2_is_home'] = valid.apply(lambda r: 1 if str(r['venue']).lower().find(str(r['team2']).split()[0].lower()) != -1 else 0, axis=1)

        X = valid[cat_cols + num_cols]
        y_target = valid['target_runs'].astype(float)
        y_margin = valid['result_margin_numeric'].astype(float)
        try:
            y_pred = _bp.predict(X)
            pred_target = y_pred[:, 0]
            pred_margin = y_pred[:, 1]
            r2_target = float(r2_score(y_target, pred_target))
            r2_margin = float(r2_score(y_margin, pred_margin))
        except Exception:
            return None
        baseline_target_pred = np.full_like(y_target, y_target.mean(), dtype=float)
        baseline_margin_pred = np.full_like(y_margin, y_margin.mean(), dtype=float)
        baseline_r2_target = float(r2_score(y_target, baseline_target_pred))
        baseline_r2_margin = float(r2_score(y_margin, baseline_margin_pred))
        return {
            'Saved Model Target R2': r2_target,
            'Saved Model Margin R2': r2_margin,
            'Baseline Target R2': baseline_r2_target,
            'Baseline Margin R2': baseline_r2_margin
        }

    # try to load existing model_results.json
    if Path(MODEL_RESULTS_JSON).exists():
        try:
            res = pd.read_json(MODEL_RESULTS_JSON, orient='index', typ='series').to_frame(name='R2').reset_index()
            res.columns = ['Metric','R2']
            fig = px.bar(res, x='Metric', y='R2', color='Metric', title='Model Performance (R²)')
            fig.update_layout(margin=dict(t=30, b=20, l=30, r=10), height=360)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
        except Exception:
            st.info("model_results.json exists but couldn't parse it.")
            st.write(Path(MODEL_RESULTS_JSON).read_text()[:1000])
    else:
        # show compute button and attempt to compute if pressed
        st.info("No precomputed model results found. You can compute R² now using the loaded dataset and saved model.")
        if st.button("Compute R² now (uses CSV & saved model)"):
            if df is None:
                st.error("No CSV available to compute R². Upload or place IPL CSV in working folder.")
            elif best_pipeline is None:
                st.error("Saved model (best_pipeline.joblib) not found.")
            else:
                with st.spinner("Computing R²..."):
                    computed = compute_r2_from_data(df, best_pipeline)
                if computed is None:
                    st.error("Could not compute R² — not enough valid rows or pipeline failed.")
                else:
                    # save to json for future runs (and display)
                    try:
                        json.dump(computed, open(MODEL_RESULTS_JSON, 'w'), indent=2)
                    except Exception as e:
                        st.warning(f"Could not save {MODEL_RESULTS_JSON}: {e}")
                    res = pd.DataFrame(list(computed.items()), columns=['Metric','R2'])
                    fig = px.bar(res, x='Metric', y='R2', color='Metric', title='Model Performance (R²)')
                    fig.update_layout(margin=dict(t=30,b=20,l=30,r=10), height=360)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
                    st.success("Computed and saved model_results.json")

    # 2) Home vs Away pie chart (home detection heuristic)
    st.markdown("### Home vs Away counts")
    def is_home(row):
        try:
            v = str(row.get('venue','')).lower()
            t1 = str(row.get('team1','')).split()[0].lower()
            return t1 in v
        except Exception:
            return False
    df['_team1_is_home_guess'] = df.apply(is_home, axis=1)
    home_count = int(df['_team1_is_home_guess'].sum())
    away_count = int(len(df) - home_count)
    fig = go.Figure(data=[go.Pie(labels=['Home','Away'], values=[home_count, away_count], hole=0.4)])
    fig.update_layout(margin=dict(t=20,b=20,l=10,r=10), height=320)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})

    # 3) Target runs trend across seasons
    st.markdown("### Target runs trend per season")
    if 'season' in df.columns and 'target_runs' in df.columns:
        trend = df.groupby('season')['target_runs'].mean().reset_index()
        fig = px.line(trend, x='season', y='target_runs', markers=True, title='Target Runs Trend Over Seasons')
        fig.update_layout(margin=dict(t=30,b=20), height=360)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    elif 'year' in df.columns and 'target_runs' in df.columns:
        trend = df.groupby('year')['target_runs'].mean().reset_index()
        fig = px.line(trend, x='year', y='target_runs', markers=True, title='Target Runs Trend Over Years')
        fig.update_layout(margin=dict(t=30,b=20), height=360)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    else:
        st.info("No target_runs/season data available in CSV.")

    # 4) Distribution of result margins (improved)
    st.markdown("### Distribution of Result Margins")
    if 'result_margin' in df.columns:
        arr = pd.to_numeric(df['result_margin'], errors='coerce').dropna().values
        cols = st.columns([3,1])
        with cols[1]:
            full_view = st.checkbox("Show full range", value=False)
        if full_view:
            fig_full = plot_distribution_with_box("Distribution of Result Margins (full)", arr, trim_pct=(0,100))
            st.plotly_chart(fig_full, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
        else:
            fig_trim = plot_distribution_with_box("Distribution of Result Margins (5–95% trim)", arr, trim_pct=(5,95))
            st.plotly_chart(fig_trim, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    else:
        st.info("No result_margin column found.")

    # ---- Extra analytics requested ----
    st.markdown("---")
    st.subheader("Extra analytics")

    # A) Team win-rate heatmap (team vs opponent)
    st.markdown("#### Team win-rate heatmap (team vs opponent)")
    if 'winner' in df.columns and 'team1' in df.columns and 'team2' in df.columns:
        teams = sorted(set(df['team1'].dropna().unique()).union(set(df['team2'].dropna().unique())))
        # build matrix
        win_matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=float)
        counts_matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=int)
        for _, r in df.iterrows():
            t1 = r.get('team1'); t2 = r.get('team2'); w = r.get('winner')
            if pd.isna(t1) or pd.isna(t2): continue
            counts_matrix.loc[t1, t2] += 1
            if w == t1:
                win_matrix.loc[t1, t2] += 1
            elif w == t2:
                win_matrix.loc[t2, t1] += 1
        rate_matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)
        for a in teams:
            for b in teams:
                cnt = counts_matrix.loc[a, b]
                if cnt > 0:
                    rate_matrix.loc[a, b] = win_matrix.loc[a, b] / cnt
                else:
                    rate_matrix.loc[a, b] = np.nan
        fig = go.Figure(data=go.Heatmap(z=rate_matrix.values, x=teams, y=teams,
                                       colorbar=dict(title='win rate'), zmin=0, zmax=1))
        fig.update_layout(title='Head-to-head win rate (row team vs column team)', xaxis={'tickangle': -45},
                          margin=dict(t=40, b=120, l=60, r=40), height=600)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    else:
        st.info("Need winner, team1 and team2 columns to compute head-to-head heatmap.")

    # B) Home vs away comparison per team
    st.markdown("#### Home vs Away: wins per team")
    if 'winner' in df.columns and 'team1' in df.columns and 'venue' in df.columns:
        df['home_team_guess'] = df.apply(lambda r: r['team1'] if str(r['team1']).split()[0].lower() in str(r.get('venue','')).lower() else None, axis=1)
        teams_all = sorted(set(df['team1'].dropna().unique()).union(set(df['team2'].dropna().unique())))
        home_wins = {t:0 for t in teams_all}
        away_wins = {t:0 for t in teams_all}
        for _, r in df.iterrows():
            w = r.get('winner'); home = r.get('home_team_guess')
            if pd.isna(w): continue
            if home and home == w:
                home_wins[w] = home_wins.get(w,0) + 1
            else:
                away_wins[w] = away_wins.get(w,0) + 1
        df_hw = pd.DataFrame({'team': teams_all, 'home_wins':[home_wins.get(t,0) for t in teams_all], 'away_wins':[away_wins.get(t,0) for t in teams_all]})
        df_hw = df_hw.sort_values('home_wins', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_hw['team'], y=df_hw['home_wins'], name='Home wins'))
        fig.add_trace(go.Bar(x=df_hw['team'], y=df_hw['away_wins'], name='Away wins'))
        fig.update_layout(barmode='group', xaxis={'tickangle':-45}, title='Home vs Away wins per team', margin=dict(t=30,b=120), height=520)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    else:
        st.info("Need winner/team1/venue columns to compute Home/Away per team.")

    # C) Match result distribution per year
    st.markdown("#### Match result distribution per year (W by runs vs wickets/other)")
    if 'year' in df.columns and 'result' in df.columns and 'result_margin' in df.columns:
        df['result_margin_num'] = pd.to_numeric(df['result_margin'], errors='coerce')
        df['outcome_type'] = df['result'].fillna('unknown')
        g = df.groupby(['year','outcome_type']).size().reset_index(name='count')
        fig = px.bar(g, x='year', y='count', color='outcome_type', title='Match result types per year', barmode='stack')
        fig.update_layout(margin=dict(t=30,b=20), height=400)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
    else:
        st.info("Need year/result/result_margin columns for per-year distribution.")

    # D) Bat-first vs chase-first advantage
    st.markdown("#### Bat-first vs Chase-first advantage (win %)")
    if 'toss_decision' in df.columns and 'toss_winner' in df.columns and 'winner' in df.columns:
        def batting_first_team(row):
            if pd.isna(row.get('toss_winner')) or pd.isna(row.get('team1')) or pd.isna(row.get('team2')): return None
            tw = row.get('toss_winner')
            if row.get('toss_decision') == 'bat':
                return tw
            else:
                return row.get('team2') if tw == row.get('team1') else row.get('team1')
        df['bat_first'] = df.apply(batting_first_team, axis=1)
        df['bat_first_winner'] = df.apply(lambda r: 1 if r['bat_first']==r['winner'] else 0, axis=1)
        bat_first_matches = df[df['bat_first'].notna()]
        if len(bat_first_matches) > 0:
            bat_first_winpct = bat_first_matches['bat_first_winner'].mean() * 100
            st.write(f"Across matches where toss info is available, team batting first won **{bat_first_winpct:.1f}%** of the time (approx).")
            gp = bat_first_matches.groupby('year')['bat_first_winner'].mean().reset_index()
            if not gp.empty:
                fig = px.line(gp, x='year', y='bat_first_winner', title='Bat-first win % over years', markers=True)
                fig.update_yaxes(tickformat='.0%')
                fig.update_layout(margin=dict(t=30,b=20), height=360)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
        else:
            st.info("Insufficient toss / team data to compute bat-first statistics.")
    else:
        st.info("Need toss_decision/toss_winner/winner columns to compute bat-first advantage.")

    