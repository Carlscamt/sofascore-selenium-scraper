
"""
Monte Carlo Simulation for Bankroll Management
==============================================
Simulates 10,000 runs of a betting season to estimate bankruptcy risk.
"""
import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo(win_rate, avg_odds, roi, n_bets=331, n_sims=10000, start_bank=1000, kelly_fraction=0.25):
    print(f"--- CONFIGURATION ---")
    print(f"Win Rate: {win_rate*100:.1f}%")
    print(f"Avg Odds: {avg_odds:.2f}")
    print(f"ROI (Edge): {roi*100:.1f}%")
    print(f"Kelly Fraction: {kelly_fraction}")
    
    # Kelly Criterion
    # f = (bp - q) / b
    b = avg_odds - 1
    p = win_rate
    q = 1 - p
    full_kelly = (b * p - q) / b
    
    bet_pct = full_kelly * kelly_fraction
    print(f"Full Kelly: {full_kelly*100:.2f}%")
    print(f"Bet Size (Fractional): {bet_pct*100:.2f}% of Bankroll")
    
    results = []
    bankruptcy_count = 0
    
    np.random.seed(42)
    
    for _ in range(n_sims):
        bank = start_bank
        
        # Vectorized simulation for speed
        outcomes = np.random.random(n_bets) < win_rate
        
        # Iterate to update bankroll (compound)
        # Note: Vectorizing compounding is compounding, so loop is okay for 300 bets
        path = [bank]
        curr_bank = bank
        
        for win in outcomes:
            stake = curr_bank * bet_pct
            if stake < 1: stake = 1 # Minimum bet
            
            if win:
                curr_bank += stake * (avg_odds - 1)
            else:
                curr_bank -= stake
            
            if curr_bank <= 0:
                curr_bank = 0
                break
        
        if curr_bank <= 0:
            bankruptcy_count += 1
            
        results.append(curr_bank)
        
    results = np.array(results)
    
    print("\n--- RESULTS (10,000 Sims) ---")
    print(f"Mean Final Bank: ${results.mean():.2f}")
    print(f"Median Final Bank: ${np.median(results):.2f}")
    print(f"P10 (Worst 10%): ${np.percentile(results, 10):.2f}")
    print(f"P90 (Best 10%): ${np.percentile(results, 90):.2f}")
    print(f"Bankruptcy Rate: {bankruptcy_count / n_sims * 100:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(start_bank, color='red', linestyle='--', label='Starting Bank')
    plt.title(f"Monte Carlo: Final Bankroll Distribution (Kelly {kelly_fraction})")
    plt.xlabel("Final Bankroll ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/monte_carlo_distribution.png')
    print("[GRAPH] Saved to reports/monte_carlo_distribution.png")

if __name__ == "__main__":
    # Parameters from the "Longshot" Strategy (The best performer)
    # Win Rate: 27.8%
    # ROI: 16.07%
    # Implied Odds from Win Rate + ROI? 
    # Profit = (Odds-1)*WR - (1-WR) = Odds*WR - WR - 1 + WR = Odds*WR - 1
    # ROI = Profit / 1 = Odds*WR - 1
    # 1.1607 = Odds * 0.278 => Odds = 4.17
    
    run_monte_carlo(win_rate=0.278, avg_odds=4.17, roi=0.1607)
