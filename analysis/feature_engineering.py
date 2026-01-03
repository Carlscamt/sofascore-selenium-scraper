import pandas as pd
import numpy as np

class RollingStatsCalculator:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.metrics = [] # Will be populated dynamically

    def discover_metrics(self, df):
        """
        Dynamically finds all statistic columns present in the dataframe.
        """
        stats = set()
        for col in df.columns:
            if col.startswith('stats_home_'):
                metric = col.replace('stats_home_', '')
                stats.add(metric)
        self.metrics = list(stats)
        print(f"   [Feature Engineering] Discovered {len(self.metrics)} statistics to roll.")

    def prepare_team_data(self, df):
        """
        Transforms match-level data (Home vs Away) into team-level data.
        Returns a DataFrame with one row per team per match.
        """
        # Select relevant columns for Home team
        # Explicitly add Lineup columns if they exist
        extra_cols_home = []
        if 'home_lineup_rating' in df.columns: extra_cols_home.extend(['home_lineup_rating', 'home_missing_count'])
        
        home_cols = ['id', 'date', 'league', 'home'] + [f'stats_home_{m}' for m in self.metrics if f'stats_home_{m}' in df.columns] + extra_cols_home
        home_df = df[home_cols].copy()
        
        # Rename to generic columns
        rename_dict_home = {'home': 'team', 'home_lineup_rating': 'lineup_rating', 'home_missing_count': 'missing_count'}
        for m in self.metrics:
            if f'stats_home_{m}' in df.columns:
                rename_dict_home[f'stats_home_{m}'] = m
        home_df = home_df.rename(columns=rename_dict_home)
        home_df['is_home'] = 1

        # Select relevant columns for Away team
        extra_cols_away = []
        if 'away_lineup_rating' in df.columns: extra_cols_away.extend(['away_lineup_rating', 'away_missing_count'])

        away_cols = ['id', 'date', 'league', 'away'] + [f'stats_away_{m}' for m in self.metrics if f'stats_away_{m}' in df.columns] + extra_cols_away
        away_df = df[away_cols].copy()
        
        # Rename to generic columns
        rename_dict_away = {'away': 'team', 'away_lineup_rating': 'lineup_rating', 'away_missing_count': 'missing_count'}
        for m in self.metrics:
            if f'stats_away_{m}' in df.columns:
                rename_dict_away[f'stats_away_{m}'] = m
        away_df = away_df.rename(columns=rename_dict_away)
        away_df['is_home'] = 0

        # Combine
        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df['date'] = pd.to_datetime(team_df['date'])
        team_df = team_df.sort_values(by=['team', 'date'])
        
        return team_df

    def calculate_rolling_averages(self, df):
        """
        Calculates rolling averages for each team and merges detailed stats back to the original dataframe.
        """
        # 0. Auto-discover metrics
        self.discover_metrics(df)

        # 1. Convert stats columns to numeric, handling '%' and errors
        for col in self.metrics:
            home_col = f'stats_home_{col}'
            away_col = f'stats_away_{col}'
            
            if home_col in df.columns:
                df[home_col] = df[home_col].astype(str).str.replace('%', '').replace('None', np.nan)
                df[home_col] = pd.to_numeric(df[home_col], errors='coerce')
                
            if away_col in df.columns:
                df[away_col] = df[away_col].astype(str).str.replace('%', '').replace('None', np.nan)
                df[away_col] = pd.to_numeric(df[away_col], errors='coerce')

        # 2. Get team-level data
        team_df = self.prepare_team_data(df)

        # 3. Calculate Rolling Stats
        # Group by Team (and League if we want league-specific form, but general form is usually better)
        # We will roll purely by Team
        
        
        cols_to_roll = [m for m in self.metrics if m in team_df.columns]
        # Explicitly add lineup stats to rolling list
        if 'lineup_rating' in team_df.columns: cols_to_roll.append('lineup_rating')
        if 'missing_count' in team_df.columns: cols_to_roll.append('missing_count')
        
        for col in cols_to_roll:
            # Shift 1 to ensure we use specific PREVIOUS games, not including current one
            team_df[f'rolling_{col}'] = team_df.groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window=self.window_size, min_periods=1).mean()
            )

        # 4. Merge back to original Match DataFrame
        
        # Split back into Home vs Away stats to merge
        # We need to merge rolling stats for the Home Team
        home_rolling = team_df[team_df['is_home'] == 1][['id'] + [f'rolling_{c}' for c in cols_to_roll]]
        home_rolling.columns = ['id'] + [f'home_avg_{c}' for c in cols_to_roll]
        
        # Merge rolling stats for the Away Team
        away_rolling = team_df[team_df['is_home'] == 0][['id'] + [f'rolling_{c}' for c in cols_to_roll]]
        away_rolling.columns = ['id'] + [f'away_avg_{c}' for c in cols_to_roll]
        
        # Merge to main DF
        # Merge to main DF
        df = df.merge(home_rolling, on='id', how='left')
        df = df.merge(away_rolling, on='id', how='left')
        
        # 5. Advanced Metric Derivation (Diffs & Ratios)
        df = self.calculate_advanced_metrics(df)

        return df

    def calculate_advanced_metrics(self, df):
        """
        Computes comparative metrics (Home vs Away Diffs) to give the model context.
        """
        # xG Difference
        if 'home_avg_expectedGoals' in df.columns and 'away_avg_expectedGoals' in df.columns:
            df['xg_diff'] = df['home_avg_expectedGoals'] - df['away_avg_expectedGoals']
            
        # Possession Difference
        if 'home_avg_ballPossession' in df.columns and 'away_avg_ballPossession' in df.columns:
            df['possession_diff'] = df['home_avg_ballPossession'] - df['away_avg_ballPossession']
            
        # Market Value Ratio (Log Scale for stability)
        # Using 1 as minimum to avoid log(0)
        if 'home_total_market_value' in df.columns and 'away_total_market_value' in df.columns:
            hmv = df['home_total_market_value'].fillna(0) + 1
            amv = df['away_total_market_value'].fillna(0) + 1
            df['market_value_log_diff'] = np.log(hmv) - np.log(amv)

        return df

    def calculate_h2h_history(self, df):
        """
        Calculates cumulative Head-to-Head stats chronologically to avoid look-ahead bias.
        It iterates through the sorted DataFrame and maintains a history of matchups.
        """
        # Ensure chronological order
        df = df.sort_values(by='date')
        
        # History store: keys are frozenset({teamA, teamB}) -> {'wins': {team: count}, 'draws': count}
        history = {}
        
        h2h_home_wins = []
        h2h_away_wins = []
        h2h_draws = []
        
        for idx, row in df.iterrows():
            home = row['home']
            away = row['away']
            pair = frozenset({home, away})
            
            # 1. Retrieve current history (BEFORE this match)
            if pair not in history:
                history[pair] = {'wins': {}, 'draws': 0}
                
            stats = history[pair]
            
            # Append features for this match based on PAST history
            h2h_home_wins.append(stats['wins'].get(home, 0))
            h2h_away_wins.append(stats['wins'].get(away, 0))
            h2h_draws.append(stats['draws'])
            
            # 2. Update history with THIS match result (for FUTURE matches)
            # Determine winner
            score_home = row['score_home']
            score_away = row['score_away']
            
            if score_home > score_away:
                stats['wins'][home] = stats['wins'].get(home, 0) + 1
            elif score_away > score_home:
                stats['wins'][away] = stats['wins'].get(away, 0) + 1
            else:
                stats['draws'] += 1
                
        # Add new columns to DF
        df['h2h_home_wins'] = h2h_home_wins
        df['h2h_away_wins'] = h2h_away_wins
        df['h2h_draws'] = h2h_draws
        
        return df
