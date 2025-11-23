from flask import Flask, request, render_template_string, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

best_pipeline = joblib.load("best_pipeline.joblib")
winner_pipeline = joblib.load("winner_pipeline.joblib")

FORM = """
<h1>IPL Predictor</h1>
<form method='post'>
Team 1: <input name='team1' value='Mumbai Indians'><br>
Team 2: <input name='team2' value='Chennai Super Kings'><br>
Venue: <input name='venue' value='Wankhede Stadium'><br>
Toss decision: 
<select name="toss_decision">
    <option value="field">Field</option>
    <option value="bat">Bat</option>
</select><br><br>
<input type='submit' value='Predict'>
</form>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        venue = request.form["venue"]
        toss_decision = request.form["toss_decision"]

        match = {
            "city": "Unknown",
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
        
        winner = team1 if winner_flag == 1 else team2
        margin = abs(int(pred[1]))

        return f"""
        <h2>Prediction Result</h2>
        <p>Target: {int(pred[0])} runs</p>
        <p>Winner: {winner}</p>
        <p>Margin: {margin} (runs or wickets depending on toss)</p>
        <p>Win Probability:</p>
        <p>{team1}: {proba[1]*100:.1f}%</p>
        <p>{team2}: {proba[0]*100:.1f}%</p>
        <br><a href='/'>Back</a>
        """

    return FORM

if __name__ == "__main__":
    app.run(debug=True)
