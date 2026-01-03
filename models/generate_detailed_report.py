
"""
Generate Detailed Model Performance Report
==========================================
Generates:
1. Classification Report (Precision, Recall, F1)
2. Confusion Matrix Plot
3. Feature Importance Plot
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Ensure clean output
if not os.path.exists('reports'):
    os.makedirs('reports')

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def process_form_string(form_str):
    if pd.isna(form_str): return [1] * 5 
    mapping = {'W': 2, 'D': 1, 'L': 0}
    parts = form_str.split(',')
    numeric_form = [mapping.get(x.strip(), 1) for x in parts]
    if len(numeric_form) > 5: numeric_form = numeric_form[:5]
    elif len(numeric_form) < 5: numeric_form += [1] * (5 - len(numeric_form))
    return numeric_form

def feature_engineering(df):
    # Process Form
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])

    # Basic Features
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_total_market_value', 'away_total_market_value',
        'home_avg_height', 'away_avg_height',
        'home_missing_count', 'away_missing_count', # New Lineup Features
    ]
    
    # Add BTTS Odds if present
    if 'odd_btts_yes' in df.columns: features.append('odd_btts_yes')
    if 'odd_btts_no' in df.columns: features.append('odd_btts_no')
    
    # Add Rolling columns
    rolling_cols = [c for c in df.columns if 'avg_' in c and c not in features]
    features.extend(rolling_cols)
    
    # Add Streak columns
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    # Handle missing values
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    home_form_df = home_form_df.loc[df_clean.index].reset_index(drop=True)
    away_form_df = away_form_df.loc[df_clean.index].reset_index(drop=True)
    
    df_clean[features] = df_clean[features].fillna(-1)
    df_clean = df_clean.reset_index(drop=True)
    
    le = LabelEncoder()
    df_clean['league_encoded'] = le.fit_transform(df_clean['league'])
    features.append('league_encoded')

    # --- DROP USELESS FEATURES ---
    DROP_FEATURES = [
        'streak_h2h_away_first_half_winner_pct', 'streak_both_less_than_2_5_goals_count', 
        'streak_both_both_teams_scoring_count', 'streak_both_both_teams_scoring_len', 
        'streak_both_less_than_2_5_goals_len', 'streak_both_less_than_2_5_goals_pct', 
        'streak_h2h_away_no_goals_conceded_len', 'streak_h2h_away_no_goals_conceded_pct', 
        'streak_both_more_than_2_5_goals_count', 'streak_both_more_than_2_5_goals_len', 
        'streak_both_more_than_2_5_goals_pct', 'streak_home_more_than_4_5_cards_count', 
        'streak_both_both_teams_scoring_pct', 'streak_h2h_away_no_goals_conceded_count', 
        'streak_home_more_than_4_5_cards_pct', 'streak_home_more_than_4_5_cards_len', 
        'streak_both_without_clean_sheet_count', 'streak_both_without_clean_sheet_len', 
        'streak_both_more_than_4_5_cards_len', 'streak_both_more_than_4_5_cards_pct', 
        'streak_both_without_clean_sheet_pct', 'streak_both_more_than_4_5_cards_count', 
        'streak_both_no_losses_len', 'streak_both_no_losses_count', 
        'streak_both_no_losses_pct', 'streak_both_less_than_10_5_corners_count', 
        'streak_both_less_than_10_5_corners_pct', 'streak_both_less_than_10_5_corners_len', 
        'streak_away_losses_count', 'a_form_4', 'a_form_5', 'a_form_1',
        'streak_h2h_home_first_half_winner_count', 'streak_h2h_home_first_half_winner_len',
        'streak_h2h_away_wins_count', 'streak_h2h_away_wins_len',
        'streak_h2h_home_no_losses_len'
    ]
    features = [f for f in features if f not in DROP_FEATURES]
    
    # --- ADD ADVANCED METRICS IF PRESENT ---
    advanced_metrics = ['xg_diff', 'possession_diff', 'market_value_log_diff']
    for m in advanced_metrics:
        if m in df.columns: features.append(m)

    X = pd.concat([df_clean[features], home_form_df, away_form_df], axis=1)
    X = X.drop(columns=[c for c in DROP_FEATURES if c in X.columns], errors='ignore')
    
    # --- CRITICAL: DROP ODDS FROM TRAINING ---
    odds_cols = ['odd_1', 'odd_X', 'odd_2', 'odd_btts_yes', 'odd_btts_no']
    X = X.drop(columns=[c for c in odds_cols if c in X.columns], errors='ignore')

    y = df_clean['target']
    
    return X, y, features + list(home_form_df.columns) + list(away_form_df.columns)

def generate_report(input_file="dataset_rolling_features.csv"):
    print(f">>> GENERATING REPORT FROM: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = load_and_preprocess_data(input_file)
    X, y, feature_names = feature_engineering(df)
    
    # 80/20 Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    
    # Train Model
    print("Training with GPU (cuda)...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42,
        verbosity=0,
        reg_alpha=0.5,
        reg_lambda=1.5,
        device="cuda"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # 1. Classification Report
    target_names = ['Home Win', 'Draw', 'Away Win']
    report = classification_report(y_test, preds, target_names=target_names)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60)
    
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    print("\n[GRAPH] Saved Confusion Matrix to reports/confusion_matrix.png")
    
    # 3. Feature Importance
    importance = model.feature_importances_
    # Map feature names to importance
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    # Save FULL list
    feat_imp.to_csv('reports/feature_importance_full.csv', index=False)
    print("\n[DATA] Saved Full Feature Importance to reports/feature_importance_full.csv")

    # Plot Top 20
    feat_imp_top = feat_imp.head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_top, palette='viridis')
    plt.title('Top 20 Features Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    print("[GRAPH] Saved Feature Importance to reports/feature_importance.png")
    
    print("\nTop 5 Influential Features:")
    print(feat_imp.head(5).to_string(index=False))

if __name__ == "__main__":
    generate_report()
