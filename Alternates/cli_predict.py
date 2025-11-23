import joblib
import pandas as pd

# Load trained models
best_pipeline = joblib.load("best_pipeline.joblib")
winner_pipeline = joblib.load("winner_pipeline.joblib")

# Basic fallback if you don't save team_stats.json
team_stats = {}

def build_match(team1, team2, venue, city="Unknown", season=2024, year=2024, toss_decision="field"):
    match = {
        "city": city,
        "match_type": "League",
        "team1": team1,
        "team2": team2,
        "toss_winner": team1,   # default
        "toss_decision": toss_decision,
        "venue": venue,
        "season": season,
        "year": year,
        "month": 4,
        "team1_win_ratio": team_stats.get(team1, {}).get("win_ratio", 0),
        "team2_win_ratio": team_stats.get(team2, {}).get("win_ratio", 0),
        "team1_is_home": 1 if team1.lower().split()[0] in venue.lower() else 0,
        "team2_is_home": 1 if team2.lower().split()[0] in venue.lower() else 0,
        "toss_advantage": 0
    }
    return match


def predict(match):
    df = pd.DataFrame([match])
    y_pred = best_pipeline.predict(df)[0]
    proba = winner_pipeline.predict_proba(df)[0]
    winner_flag = winner_pipeline.predict(df)[0]

    team1 = match["team1"]
    team2 = match["team2"]

    predicted_winner = team1 if winner_flag == 1 else team2

    print("\n----- MATCH PREDICTION -----")
    print(f"Match: {team1} vs {team2}")
    print(f"Venue: {match['venue']}")
    print(f"Predicted Target Score: {int(y_pred[0])}")

    # Margin
    margin = abs(int(y_pred[1]))
    if match["toss_decision"] == "bat":
        print(f"{predicted_winner} is predicted to win by {margin} runs.")
    else:
        wk = min(max(margin, 1), 10)
        print(f"{predicted_winner} is predicted to win by {wk} wickets (~{wk*6} balls left).")

    print(f"\nWin Probability:")
    print(f"{team1}: {proba[1]*100:.1f}%")
    print(f"{team2}: {proba[0]*100:.1f}%")
    print("------------------------------\n")


if __name__ == "__main__":
    print("IPL MATCH PREDICTOR – CLI MODE")
    t1 = input("Enter Team 1: ")
    t2 = input("Enter Team 2: ")
    venue = input("Enter Venue: ")
    toss_decision = input("Toss Decision (bat/field): ") or "field"

    match_data = build_match(t1, t2, venue, toss_decision=toss_decision)
    predict(match_data)
