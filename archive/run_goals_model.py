import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_goals_model():
    print(">>> LOADING DATA FOR GOALS MODEL...")
    try:
        df = pd.read_csv('sofascore_dataset_v2.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- TARGET ENGINEERING (Over 2.5 Goals) ---
    df['total_goals'] = df['score_home'] + df['score_away']
    # Target: 1 if Over 2.5, 0 if Under 2.5
    df['target'] = (df['total_goals'] > 2.5).astype(int)
    
    print(f"Total Matches: {len(df)}")
    print(f"Over 2.5 Rate: {df['target'].mean()*100:.1f}%")
    
    # --- FEATURE ENGINEERING ---
    features = [
        'odd_1', 'odd_X', 'odd_2', # Proxy for team strength disparity
        'home_avg_rating', 'away_avg_rating',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_avg_height', 'away_avg_height',
        'odd_btts_yes', 'odd_btts_no'
    ]
    
    # Add Streak Features
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    # Clean Data
    df_clean = df.copy()
    df_clean[streak_cols] = df_clean[streak_cols].fillna(0)
    df_clean[features] = df_clean[features].fillna(-1)
    
    X = df_clean[features]
    y = df_clean['target']
    
    # --- TRAIN/TEST SPLIT ---
    # Using random split for this specific feature test to ensure balanced classes if possible
    # But chronological is better for realism. Let's stick to last 20% like before.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # --- MODEL TRAINING ---
    print("\n>>> TRAINING XGBOOST (Binary Classification)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # --- EVALUATION ---
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] # Prob of Over 2.5
    
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # --- HIGH CONFIDENCE CHECK ---
    # Let's see if high confidence predictions are more accurate
    high_conf_indices = [i for i, p in enumerate(probs) if p > 0.60 or p < 0.40]
    if high_conf_indices:
        y_high = y_test.iloc[high_conf_indices]
        preds_high = preds[high_conf_indices]
        acc_high = accuracy_score(y_high, preds_high)
        print(f"High Confidence (>60% or <40%) Accuracy: {acc_high:.4f} (on {len(high_conf_indices)} matches)")
    
    # --- FEATURE IMPORTANCE ---
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 15 Features for Total Goals:")
    print(fi_df.head(15).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), hue='Feature', palette='magma', legend=False)
    plt.title('Feature Importance for Total Goals (Over/Under 2.5)')
    plt.tight_layout()
    plt.savefig('reports/goals_model_importance.png')
    print("\n[GRAPH] Saved plot to 'reports/goals_model_importance.png'")

if __name__ == "__main__":
    train_goals_model()
