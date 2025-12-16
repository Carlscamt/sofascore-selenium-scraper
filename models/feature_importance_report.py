import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report():
    print(">>> LOADING DATA...")
    try:
        df = pd.read_csv('sofascore_dataset_v2.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Preprocessing (Same as Run Strategies)
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
    
    # Feature Engineering
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_total_market_value', 'away_total_market_value',
        'home_avg_height', 'away_avg_height'
    ]
    
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    df_clean = df.copy()
    df_clean[streak_cols] = df_clean[streak_cols].fillna(0)
    df_clean[features] = df_clean[features].fillna(-1)
    
    X = df_clean[features]
    y = df_clean['target']
    
    print(f"Training model on {len(X)} matches with {len(features)} features...")
    
    # Train Model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # Extract Feature Importance
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    
    # Print Text Report
    print("\n" + "="*40)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*40)
    print(fi_df.head(20).to_string(index=False))
    
    # Analyze Streak vs Non-Streak
    streak_imp = fi_df[fi_df['Feature'].str.startswith('streak_')]['Importance'].sum()
    other_imp = fi_df[~fi_df['Feature'].str.startswith('streak_')]['Importance'].sum()
    print(f"\nTotal Streak Features Importance: {streak_imp:.4f} ({(streak_imp/(streak_imp+other_imp))*100:.1f}%)")
    print(f"Total Other Features Importance:  {other_imp:.4f}")
    
    # Save Plot
    plt.figure(figsize=(10, 10))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(25), hue='Feature', palette='viridis', legend=False)
    plt.title('Top 25 Feature Importance (Full 624 Match Dataset)')
    plt.tight_layout()
    plt.savefig('reports/feature_importance_full.png')
    print("\n[GRAPH] Saved plot to 'reports/feature_importance_full.png'")

if __name__ == "__main__":
    generate_report()
