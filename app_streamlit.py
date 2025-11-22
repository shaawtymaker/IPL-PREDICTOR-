import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="IPL Predictor", layout="centered")

# Load models
best_pipeline = joblib.load("best_pipeline.joblib")
winner_pipeline = joblib.load("winner_pipeline.joblib")

st.title("🏏 IPL Match Predictor")

team1 = st.text_input("Team 1", "Mumbai Indians")
team2 = st.text_input("Team 2", "Chennai Super Kings")
venue = st.text_input("Venue", "Wankhede Stadium")
city = st.text_input("City", "Mumbai")
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

if st.button("Predict"):
    match = {
        "city": city,
        "match_type": "League",
        "team1": team1,
        "team2": team2,
        "toss_winner": team1,
        "toss_decision": toss_decision,
        "venue": venue,
        "season": 2024,
        "year": 2024,
        "month": 4,
        "team1_win_ratio": 0,
        "team2_win_ratio": 0,
        "team1_is_home": 1 if team1.lower().split()[0] in venue.lower() else 0,
        "team2_is_home": 1 if team2.lower().split()[0] in venue.lower() else 0,
        "toss_advantage": 0
    }

    df = pd.DataFrame([match])
    pred = best_pipeline.predict(df)[0]
    proba = winner_pipeline.predict_proba(df)[0]
    winner_flag = winner_pipeline.predict(df)[0]
    
    predicted_winner = team1 if winner_flag == 1 else team2

    st.subheader("Predicted Target Score")
    st.write(f"**{int(pred[0])} runs**")

    st.subheader("Predicted Result")
    margin = abs(int(pred[1]))
    if toss_decision == "bat":
        st.write(f"**{predicted_winner}** predicted to win by **{margin} runs**")
    else:
        wk = min(max(margin, 1), 10)
        st.write(f"**{predicted_winner}** predicted to win by **{wk} wickets** (~{wk*6} balls left)")

    st.subheader("Win Probability")
    st.write(f"{team1}: **{proba[1]*100:.1f}%**")
    st.write(f"{team2}: **{proba[0]*100:.1f}%**")
