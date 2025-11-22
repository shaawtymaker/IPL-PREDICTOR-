import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# Load the data
data = pd.read_csv("IPL_2008-2024.csv")

# Initial data exploration
print("Dataset shape:", data.shape)
print("\nColumns:", data.columns.tolist())

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Filter relevant data - keep only completed matches with target_runs
data = data[data["super_over"] == 'N']

# Preprocess result_margin and create formatted result
data['result_margin_numeric'] = pd.to_numeric(data['result_margin'], errors='coerce')

# Create a new column that formats the result in a more intuitive way
def format_result(row):
    if pd.isna(row['result_margin_numeric']):
        return "Unknown"
    
    if row['result'] == 'runs':
        return f"Won by {int(row['result_margin_numeric'])} runs"
    elif row['result'] == 'wickets':
        # We don't have balls left in the data, but we can calculate an estimate based on wickets
        # 1 wicket typically means around 6 balls left in T20 cricket (rough estimate)
        approx_balls_left = int(row['result_margin_numeric'] * 6)
        return f"Won by {int(row['result_margin_numeric'])} wickets (~{approx_balls_left} balls left)"
    else:
        return str(row['result'])

data['formatted_result'] = data.apply(format_result, axis=1)

# Drop rows with missing target_runs or result_margin
data = data.dropna(subset=['target_runs', 'result_margin_numeric'])

print("\nFiltered dataset shape:", data.shape)
print("\nTarget runs statistics:")
print(data['target_runs'].describe())
print("\nResult margin statistics:")
print(data['result_margin_numeric'].describe())

# Feature engineering
# Compute team strength based on win ratio
team_stats = {}
for team in set(data['team1'].unique()) | set(data['team2'].unique()):
    matches_won = len(data[data['winner'] == team])
    matches_played_team1 = len(data[data['team1'] == team])
    matches_played_team2 = len(data[data['team2'] == team])
    total_matches = matches_played_team1 + matches_played_team2
    if total_matches > 0:
        win_ratio = matches_won / total_matches
    else:
        win_ratio = 0
    team_stats[team] = {'win_ratio': win_ratio}

# Add team stats to the dataframe
data['team1_win_ratio'] = data['team1'].map(lambda x: team_stats.get(x, {}).get('win_ratio', 0))
data['team2_win_ratio'] = data['team2'].map(lambda x: team_stats.get(x, {}).get('win_ratio', 0))

# Home advantage feature
data['team1_is_home'] = data.apply(lambda row: 1 if row['venue'].lower().find(row['team1'].lower().split()[0]) != -1 else 0, axis=1)
data['team2_is_home'] = data.apply(lambda row: 1 if row['venue'].lower().find(row['team2'].lower().split()[0]) != -1 else 0, axis=1)

# Toss advantage
data['toss_advantage'] = data.apply(lambda row: 1 if row['toss_winner'] == row['winner'] else 0, axis=1)

# Extract more features from the dataset
categorical_features = ['city', 'match_type', 'team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
numerical_features = ['season', 'year', 'month', 'team1_win_ratio', 'team2_win_ratio', 
                      'team1_is_home', 'team2_is_home', 'toss_advantage']

# Select features and target
X = data[categorical_features + numerical_features]
y = data[['target_runs', 'result_margin_numeric']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Create and compare different models
models = {
    'XGBoost': MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)
    ),
    'RandomForest': MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    ),
    'GradientBoosting': MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)
    )
}

# Define pipelines
pipelines = {name: Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
]) for name, model in models.items()}

# Train and evaluate models
results = {}
for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics for each target
    metrics = {
        'target_runs': {
            'MAE': mean_absolute_error(y_test['target_runs'], y_pred[:, 0]),
            'MSE': mean_squared_error(y_test['target_runs'], y_pred[:, 0]),
            'R²': r2_score(y_test['target_runs'], y_pred[:, 0])
        },
        'result_margin': {
            'MAE': mean_absolute_error(y_test['result_margin_numeric'], y_pred[:, 1]),
            'MSE': mean_squared_error(y_test['result_margin_numeric'], y_pred[:, 1]),
            'R²': r2_score(y_test['result_margin_numeric'], y_pred[:, 1])
        },
        'overall': {
            'MAE': (mean_absolute_error(y_test['target_runs'], y_pred[:, 0]) + 
                    mean_absolute_error(y_test['result_margin_numeric'], y_pred[:, 1])) / 2,
            'MSE': (mean_squared_error(y_test['target_runs'], y_pred[:, 0]) + 
                   mean_squared_error(y_test['result_margin_numeric'], y_pred[:, 1])) / 2,
            'R²': (r2_score(y_test['target_runs'], y_pred[:, 0]) + 
                  r2_score(y_test['result_margin_numeric'], y_pred[:, 1])) / 2
        }
    }
    
    results[name] = metrics
    
    print(f"{name} - Target Runs - MAE: {metrics['target_runs']['MAE']:.2f}, MSE: {metrics['target_runs']['MSE']:.2f}, R²: {metrics['target_runs']['R²']:.4f}")
    print(f"{name} - Result Margin - MAE: {metrics['result_margin']['MAE']:.2f}, MSE: {metrics['result_margin']['MSE']:.2f}, R²: {metrics['result_margin']['R²']:.4f}")
    print(f"{name} - Overall Average - MAE: {metrics['overall']['MAE']:.2f}, MSE: {metrics['overall']['MSE']:.2f}, R²: {metrics['overall']['R²']:.4f}")

# Find the best model based on overall R²
best_model_name = max(results, key=lambda k: results[k]['overall']['R²'])
best_pipeline = pipelines[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Target Runs - R²: {results[best_model_name]['target_runs']['R²']:.4f}")
print(f"Result Margin - R²: {results[best_model_name]['result_margin']['R²']:.4f}")
print(f"Overall Average - R²: {results[best_model_name]['overall']['R²']:.4f}")

# Add a winner prediction model
print("\nTraining winner prediction model...")
# Create a new target variable for the winner
data['winner_is_team1'] = (data['winner'] == data['team1']).astype(int)

# For winner prediction, use a binary classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Features remain the same, target is winner_is_team1
X_winner = X
y_winner = data['winner_is_team1']

# Split the data
X_train_winner, X_test_winner, y_train_winner, y_test_winner = train_test_split(X_winner, y_winner, test_size=0.2, random_state=42)

# Create winner prediction pipeline
winner_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42))
])

# Train winner model
winner_pipeline.fit(X_train_winner, y_train_winner)

# Evaluate winner model
y_pred_winner = winner_pipeline.predict(X_test_winner)
winner_accuracy = accuracy_score(y_test_winner, y_pred_winner)
winner_precision = precision_score(y_test_winner, y_pred_winner)
winner_recall = recall_score(y_test_winner, y_pred_winner)
winner_f1 = f1_score(y_test_winner, y_pred_winner)

print(f"Winner Prediction - Accuracy: {winner_accuracy:.4f}, Precision: {winner_precision:.4f}, Recall: {winner_recall:.4f}, F1: {winner_f1:.4f}")

# Create a prediction function for new matches
def predict_match_outcome(match_data):
    """
    Predict target runs, result margin, and match winner for a new match
    
    Parameters:
    match_data (dict): Dictionary containing match information
    
    Returns:
    dict: Predicted target runs, result margin, and winner with formatted description
    """
    # Create a dataframe with the match data
    match_df = pd.DataFrame([match_data])
    
    # Make prediction for target and margin
    predictions = best_pipeline.predict(match_df)[0]
    predicted_runs = predictions[0]
    predicted_margin = predictions[1]
    
    # Predict winner (1 for team1, 0 for team2)
    win_prob = winner_pipeline.predict_proba(match_df)[0]
    predicted_winner_is_team1 = winner_pipeline.predict(match_df)[0]
    team1_win_probability = win_prob[1]
    team2_win_probability = win_prob[0]
    
    # Determine predicted winner
    team1 = match_data['team1']
    team2 = match_data['team2']
    predicted_winner = team1 if predicted_winner_is_team1 == 1 else team2
    
    # Format the result margin prediction
    if match_data.get('toss_decision') == 'bat':
        formatted_margin = f"{predicted_winner} predicted to win by {abs(int(predicted_margin))} runs"
    else:
        # If team chose to field, then the margin would be in wickets
        wickets = min(max(int(abs(predicted_margin)), 1), 10)  # Ensure it's between 1 and 10
        approx_balls_left = wickets * 6  # Rough estimation
        formatted_margin = f"{predicted_winner} predicted to win by {wickets} wickets (~{approx_balls_left} balls left)"
    
    return {
        'predicted_target': int(predicted_runs),
        'predicted_margin': predicted_margin,
        'predicted_winner': predicted_winner,
        'team1_win_probability': f"{team1_win_probability:.2%}",
        'team2_win_probability': f"{team2_win_probability:.2%}",
        'formatted_prediction': f"Predicted target score: {int(predicted_runs)} runs. {formatted_margin} (Win probability: {team1_win_probability:.1%} for {team1}, {team2_win_probability:.1%} for {team2})"
    }

# Example usage with multiple team matchups
example_matches = [
    {
        'city': 'Mumbai',
        'match_type': 'League',
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat',
        'venue': 'Wankhede Stadium',
        'season': 2024,
        'year': 2024,
        'month': 4,
        'team1_win_ratio': team_stats.get('Mumbai Indians', {}).get('win_ratio', 0),
        'team2_win_ratio': team_stats.get('Chennai Super Kings', {}).get('win_ratio', 0),
        'team1_is_home': 1,
        'team2_is_home': 0,
        'toss_advantage': 0
    },
    {
        'city': 'Bangalore',
        'match_type': 'League',
        'team1': 'Royal Challengers Bangalore',
        'team2': 'Kolkata Knight Riders',
        'toss_winner': 'Kolkata Knight Riders',
        'toss_decision': 'field',
        'venue': 'M Chinnaswamy Stadium',
        'season': 2024,
        'year': 2024,
        'month': 4,
        'team1_win_ratio': team_stats.get('Royal Challengers Bangalore', {}).get('win_ratio', 0),
        'team2_win_ratio': team_stats.get('Kolkata Knight Riders', {}).get('win_ratio', 0),
        'team1_is_home': 1,
        'team2_is_home': 0,
        'toss_advantage': 0
    },
    {
        'city': 'Delhi',
        'match_type': 'Playoff',
        'team1': 'Delhi Capitals',
        'team2': 'Rajasthan Royals',
        'toss_winner': 'Delhi Capitals',
        'toss_decision': 'bat',
        'venue': 'Arun Jaitley Stadium',
        'season': 2024,
        'year': 2024,
        'month': 5,
        'team1_win_ratio': team_stats.get('Delhi Capitals', {}).get('win_ratio', 0),
        'team2_win_ratio': team_stats.get('Rajasthan Royals', {}).get('win_ratio', 0),
        'team1_is_home': 1,
        'team2_is_home': 0,
        'toss_advantage': 0
    }
]

print("\nExample predictions:")
for i, match in enumerate(example_matches, 1):
    prediction = predict_match_outcome(match)
    print(f"\nMatch {i}: {match['team1']} vs {match['team2']} at {match['venue']}")
    print(prediction['formatted_prediction'])
    print(f"Winner prediction: {prediction['predicted_winner']} (Probability: {prediction['team1_win_probability']} for {match['team1']}, {prediction['team2_win_probability']} for {match['team2']})")

import joblib

joblib.dump(best_pipeline, "best_pipeline.joblib")
joblib.dump(winner_pipeline, "winner_pipeline.joblib")

print("\nSaved trained models as best_pipeline.joblib and winner_pipeline.joblib")
