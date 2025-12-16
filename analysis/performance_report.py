"""
Comprehensive Visual Performance Report
========================================
Generates multiple charts showing model performance, betting results, and strategy analysis.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from run_full_model import load_and_preprocess_data, feature_engineering_full

def generate_performance_report():
    print("="*60)
    print("GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("="*60)
    
    # 1. Load Data
    df = load_and_preprocess_data("sofascore_dataset_v2.csv")
    print(f"Loaded {len(df)} matches")
    
    # 2. Prepare Features
    X, y, df_clean = feature_engineering_full(df)
    print(f"Clean data points: {len(X)}")
    
    # 3. Train Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss',
        use_label_encoder=False, 
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    
    # 4. Get Predictions and Probabilities
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    # 5. Calculate Bets
    bets = []
    for i in range(len(X)):
        odd_1 = X.iloc[i]['odd_1']
        odd_X = X.iloc[i]['odd_X']
        odd_2 = X.iloc[i]['odd_2']
        
        p_1, p_X, p_2 = probs[i]
        actual = y.iloc[i]
        
        ev_1 = (p_1 * odd_1) - 1
        ev_X = (p_X * odd_X) - 1
        ev_2 = (p_2 * odd_2) - 1
        
        best_ev = max(ev_1, ev_X, ev_2)
        if best_ev > 0:
            if ev_1 == best_ev:
                choice, odd, prob = 0, odd_1, p_1
            elif ev_X == best_ev:
                choice, odd, prob = 1, odd_X, p_X
            else:
                choice, odd, prob = 2, odd_2, p_2
            
            profit = (odd - 1) if choice == actual else -1
            bets.append({
                'choice': choice,
                'odd': odd,
                'prob': prob,
                'ev': best_ev,
                'actual': actual,
                'pnl': profit,
                'result': 'WIN' if choice == actual else 'LOSS'
            })
    
    print(f"Total Bets: {len(bets)}")
    
    # === CREATE MULTI-PANEL FIGURE ===
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # --- Panel 1: Cumulative Profit ---
    ax1 = fig.add_subplot(gs[0, :])
    cumsum = np.cumsum([b['pnl'] for b in bets])
    ax1.plot(cumsum, color='blue', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.fill_between(range(len(cumsum)), cumsum, 0, 
                     where=[c > 0 for c in cumsum], color='green', alpha=0.3)
    ax1.fill_between(range(len(cumsum)), cumsum, 0,
                     where=[c <= 0 for c in cumsum], color='red', alpha=0.3)
    roi = (sum(b['pnl'] for b in bets) / len(bets)) * 100
    ax1.set_title(f'Cumulative Profit Over Time (ROI: {roi:.1f}%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Profit (Units)')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Win/Loss Distribution ---
    ax2 = fig.add_subplot(gs[1, 0])
    wins = sum(1 for b in bets if b['result'] == 'WIN')
    losses = len(bets) - wins
    ax2.bar(['Wins', 'Losses'], [wins, losses], color=['green', 'red'])
    ax2.set_title(f'Win/Loss Distribution (Win Rate: {wins/len(bets)*100:.1f}%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count')
    for i, v in enumerate([wins, losses]):
        ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    # --- Panel 3: Bet Type Distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    home_bets = sum(1 for b in bets if b['choice'] == 0)
    draw_bets = sum(1 for b in bets if b['choice'] == 1)
    away_bets = sum(1 for b in bets if b['choice'] == 2)
    ax3.pie([home_bets, draw_bets, away_bets], labels=['Home', 'Draw', 'Away'], 
            autopct='%1.1f%%', colors=['skyblue', 'gray', 'coral'], startangle=90)
    ax3.set_title('Bet Type Distribution', fontsize=12, fontweight='bold')
    
    # --- Panel 4: ROI by Odds Range ---
    ax4 = fig.add_subplot(gs[2, 0])
    odds_ranges = [(1.0, 1.5, 'Short'), (1.5, 2.5, 'Mid'), (2.5, 4.0, 'Long'), (4.0, 100, 'Longshot')]
    range_data = []
    for low, high, label in odds_ranges:
        range_bets = [b for b in bets if low <= b['odd'] < high]
        if range_bets:
            range_roi = (sum(b['pnl'] for b in range_bets) / len(range_bets)) * 100
            range_data.append((label, range_roi, len(range_bets)))
    
    labels, rois, counts = zip(*range_data) if range_data else ([], [], [])
    colors = ['green' if r > 0 else 'red' for r in rois]
    bars = ax4.bar(labels, rois, color=colors)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_title('ROI by Odds Range', fontsize=12, fontweight='bold')
    ax4.set_ylabel('ROI (%)')
    # Add count labels
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'n={count}', ha='center', fontsize=9)
    
    # --- Panel 5: Feature Importance (Top 10) ---
    ax5 = fig.add_subplot(gs[2, 1])
    importance = model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
    
    ax5.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
    ax5.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Importance')
    
    # Add metrics summary
    fig.suptitle('Football Betting Model - Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    outfile = "performance_dashboard.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n[SUCCESS] Dashboard saved to: {outfile}")
    
    # Print summary stats
    print("\n" + "="*40)
    print("SUMMARY STATISTICS")
    print("="*40)
    print(f"Total Matches: {len(df)}")
    print(f"Clean Data Points: {len(X)}")
    print(f"Total Bets Placed: {len(bets)}")
    print(f"Wins: {wins} ({wins/len(bets)*100:.1f}%)")
    print(f"Losses: {losses} ({losses/len(bets)*100:.1f}%)")
    print(f"Total Profit: {sum(b['pnl'] for b in bets):.2f} units")
    print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    generate_performance_report()
