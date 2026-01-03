import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.feature_engineering import RollingStatsCalculator

class TestRollingStats(unittest.TestCase):
    def test_rolling_average_logic(self):
        # Create dummy data
        data = {
            'id': [1, 2, 3, 4, 5, 6],
            'date': ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-02', '2023-01-09', '2023-01-16'],
            'league': ['L1', 'L1', 'L1', 'L1', 'L1', 'L1'],
            'home': ['TeamA', 'TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamB'],
            'away': ['TeamB', 'TeamB', 'TeamB', 'TeamA', 'TeamA', 'TeamA'],
            # Team A stats when home
            'stats_home_ballPossession': [50, 60, 70, None, None, None], 
            # Team B stats when away
            'stats_away_ballPossession': [50, 40, 30, None, None, None],
            # Team B stats when home
            'stats_home_ballPossession': [None, None, None, 40, 50, 60], # Overwritten if dict, but this is DF construction
        }
        
        # Better DF construction
        rows = [
            {'id': 1, 'date': '2023-01-01', 'home': 'TeamA', 'away': 'TeamB', 'stats_home_ballPossession': 50, 'stats_away_ballPossession': 50},
            {'id': 2, 'date': '2023-01-08', 'home': 'TeamA', 'away': 'TeamC', 'stats_home_ballPossession': 60, 'stats_away_ballPossession': 40},
            {'id': 3, 'date': '2023-01-15', 'home': 'TeamA', 'away': 'TeamD', 'stats_home_ballPossession': 70, 'stats_away_ballPossession': 30},
            # Team A playing away
            {'id': 4, 'date': '2023-01-22', 'home': 'TeamE', 'away': 'TeamA', 'stats_home_ballPossession': 45, 'stats_away_ballPossession': 55},
        ]
        df = pd.DataFrame(rows)
        df['league'] = 'L1'
        
        calculator = RollingStatsCalculator(window_size=2)
        result_df = calculator.calculate_rolling_averages(df)
        
        # Check Team A (Home in matches 1,2,3; Away in 4)
        # Match 1: No history -> NaN
        # Match 2: Hist: [50] -> Avg 50
        # Match 3: Hist: [50, 60] -> Avg 55
        # Match 4: Hist: [60, 70] (last 2) -> Avg 65. (Team A is Away here)
        
        row1 = result_df[result_df['id'] == 1].iloc[0]
        row2 = result_df[result_df['id'] == 2].iloc[0]
        row3 = result_df[result_df['id'] == 3].iloc[0]
        row4 = result_df[result_df['id'] == 4].iloc[0]
        
        print("\nRow 1 (Home Avg):", row1.get('home_avg_ballPossession'))
        print("Row 2 (Home Avg):", row2.get('home_avg_ballPossession'))
        print("Row 3 (Home Avg):", row3.get('home_avg_ballPossession'))
        print("Row 4 (Away Avg):", row4.get('away_avg_ballPossession'))
        
        # Assertions
        self.assertTrue(pd.isna(row1['home_avg_ballPossession']), "First match should be NaN")
        self.assertEqual(row2['home_avg_ballPossession'], 50.0)
        self.assertEqual(row3['home_avg_ballPossession'], 55.0) # (50+60)/2
        self.assertEqual(row4['away_avg_ballPossession'], 65.0) # (60+70)/2 (Window moves)

if __name__ == '__main__':
    unittest.main()
