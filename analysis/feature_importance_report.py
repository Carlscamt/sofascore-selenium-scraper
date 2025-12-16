import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
from run_full_model import load_and_preprocess_data, feature_engineering_full

def generate_importance_report():
    print("="*60)
    print("GENERATING FEATURE IMPORTANCE REPORT")
    print("="*60)
    
    # 1. Load Data
    try:
        df = load_and_preprocess_data("sofascore_dataset_v2.csv")
        print(f"Loaded {len(df)} matches")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare Features
    try:
        X, y, _ = feature_engineering_full(df)
        print(f"Features Prepared: {X.shape[1]} features")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return

    # 3. Train Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss',
        use_label_encoder=False, 
        random_state=42
    )
    model.fit(X, y)
    
    # 4. Extract Importance
    importance = model.feature_importances_
    features = X.columns
    
    # Create DataFrame
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # 5. Print Text Report
    print("\n" + "="*40)
    print("TOP 15 FEATURES")
    print("="*40)
    print(f"{'Rank':<5} {'Feature':<30} {'Importance':<10}")
    print("-" * 50)
    for i in range(min(15, len(fi_df))):
        row = fi_df.iloc[i]
        print(f"{i+1:<5} {row['Feature']:<30} {row['Importance']:.4f}")
    
    # 6. Plot
    plt.figure(figsize=(12, 8))
    # Plot top 20
    top_n = 20
    plot_df = fi_df.head(top_n).sort_values(by='Importance', ascending=True) # Ascending for horiz bar
    
    plt.barh(plot_df['Feature'], plot_df['Importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title(f'XGBoost Feature Importance (Top {top_n})')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    outfile = "feature_importance_report.png"
    plt.savefig(outfile)
    print(f"\n[GRAPH] Saved feature importance plot to: {outfile}")
    
    # 7. Lineup Features Analysis
    print("\n" + "="*40)
    print("LINEUP FEATURES ANALYSIS")
    print("="*40)
    lineup_cols = [
        'home_total_market_value', 'away_total_market_value', 
        'home_avg_height', 'away_avg_height',
        'home_defenders', 'home_midfielders', 'home_forwards',
        'away_defenders', 'away_midfielders', 'away_forwards'
    ]
    
    lineup_fi = fi_df[fi_df['Feature'].isin(lineup_cols)]
    if not lineup_fi.empty:
        print(lineup_fi.to_string(index=False))
        total_imp = lineup_fi['Importance'].sum()
        print(f"\nTotal Lineup Features Importance: {total_imp:.4f} ({total_imp/fi_df['Importance'].sum()*100:.1f}%)")
    else:
        print("No lineup features found in model.")

if __name__ == "__main__":
    # Fix for windows encoding if needed
    sys.stdout.reconfigure(encoding='utf-8')
    generate_importance_report()
