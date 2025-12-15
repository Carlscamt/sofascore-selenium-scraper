import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set Page Config
st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

# --- FUNCTIONS ---

@st.cache_data
def load_data(history_path, future_path):
    """Loads historical and future data."""
    try:
        df_hist = pd.read_csv(history_path)
        df_future = pd.read_csv(future_path)
        return df_hist, df_future
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None

def process_form_string(form_str):
    """Parses form string to numeric list."""
    if pd.isna(form_str):
        return [1] * 5 
    mapping = {'W': 2, 'D': 1, 'L': 0}
    parts = form_str.split(',')
    numeric_form = [mapping.get(x.strip(), 1) for x in parts]
    if len(numeric_form) > 5: numeric_form = numeric_form[:5]
    elif len(numeric_form) < 5: numeric_form += [1] * (5 - len(numeric_form))
    return numeric_form

def feature_engineering(df, is_training=True):
    """
    Transforms dataframe into X features.
    If is_training=True, also returns y target.
    """
    # 1. Process Form
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])
    
    # 2. Key Features
    # Ensure all needed columns exist (fill missing cols with NaN if future data lacks them)
    required_cols = ['odd_1', 'odd_X', 'odd_2', 'home_avg_rating', 'away_avg_rating', 
                     'home_position', 'away_position', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'league']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows without odds for TRAINING only (we need odds to train/bet)
    # For future, we also need odds to predict Betting Strategy, so drop if missing.
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    # Re-align separate form dfs
    home_form_df = home_form_df.loc[df_clean.index]
    away_form_df = away_form_df.loc[df_clean.index]
    
    # 3. Label Encoding for League
    # Simple hash encoding or frequency encoding would be better for unknown leagues in future,
    # but for now we stick to simple LabelEncoding. Using a fixed encoder is tricky if new leagues appear.
    # Hack: Use hash of league name to handle new unseen leagues roughly.
    df_clean['league_encoded'] = df_clean['league'].apply(lambda x: hash(x) % 1000)
    
    feature_cols = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'league_encoded'
    ]
    
    df_clean[feature_cols] = df_clean[feature_cols].fillna(-1) 
    
    X = pd.concat([
        df_clean[feature_cols].reset_index(drop=True),
        home_form_df.reset_index(drop=True),
        away_form_df.reset_index(drop=True)
    ], axis=1)
    
    if is_training:
        # Create Target: 0: Home, 1: Draw, 2: Away
        y = np.select(
            [df_clean['score_home'] > df_clean['score_away'], df_clean['score_home'] == df_clean['score_away']],
            [0, 1],
            default=2
        )
        return df_clean, X, y
    else:
        return df_clean, X

@st.cache_resource
def train_model(X, y):
    """Trains the XGBoost model."""
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X, y)
    return model

# --- MAIN APP ---

st.title("⚽ Football Match Predictor")

# 1. Load Data
with st.spinner("Loading Data..."):
    df_hist, df_future = load_data("sofascore_large_dataset.csv", "sofascore_future_matches.csv")

if df_hist is not None and df_future is not None:
    
    # 2. Train Model
    st.sidebar.header("Model Status")
    df_train_clean, X_train, y_train = feature_engineering(df_hist, is_training=True)
    st.sidebar.success(f"Training Data: {len(X_train)} matches")
    
    with st.spinner("Training XGBoost Model..."):
        model = train_model(X_train, y_train)
    st.sidebar.success("Model Trained ✅")
    
    # 3. Predict Future
    df_future_clean, X_future = feature_engineering(df_future, is_training=False)
    
    if len(X_future) > 0:
        probs = model.predict_proba(X_future)
        
        # 4. Generate Predictions & Advice
        predictions = []
        for i, row_idx in enumerate(df_future_clean.index):
            original_row = df_future_clean.loc[row_idx]
            
            p_1, p_X, p_2 = probs[i]
            
            o_1 = original_row['odd_1']
            o_X = original_row['odd_X']
            o_2 = original_row['odd_2']
            
            # EV Calc
            ev_1 = (p_1 * o_1) - 1
            ev_X = (p_X * o_X) - 1
            ev_2 = (p_2 * o_2) - 1
            
            # Find best bet
            options = [
                {'Type': 'Home', 'Prob': p_1, 'Odd': o_1, 'EV': ev_1},
                {'Type': 'Draw', 'Prob': p_X, 'Odd': o_X, 'EV': ev_X},
                {'Type': 'Away', 'Prob': p_2, 'Odd': o_2, 'EV': ev_2},
            ]
            best_bet = max(options, key=lambda x: x['EV'])
            
            # Determine Advice Color/Strength
            strength = "Low"
            if best_bet['EV'] > 0.20: strength = "🔥 STRONG"
            elif best_bet['EV'] > 0.05: strength = "✅ GOOD"
            elif best_bet['EV'] > 0: strength = "⚠️ MARGINAL"
            else: strength = "❌ NO VALUE"
            
            predictions.append({
                'Date': original_row['date'],
                'League': original_row['league'],
                'Match': f"{original_row['home']} vs {original_row['away']}",
                'Pick': best_bet['Type'],
                'Confidence': f"{best_bet['Prob']*100:.1f}%",
                'Odds': best_bet['Odd'],
                'Expected Value': f"{best_bet['EV']*100:.1f}%",
                'Strength': strength,
                'ev_raw': best_bet['EV'], # Hidden for sorting
                'prob_raw': best_bet['Prob']
            })
            
        df_preds = pd.DataFrame(predictions)
        
        # --- UI LAYOUT ---
        
        # Filters
        st.sidebar.header("Filters")
        min_ev = st.sidebar.slider("Minimum EV (%)", -20, 50, 0)
        min_conf = st.sidebar.slider("Minimum Confidence (%)", 0, 100, 30)
        
        # Filter Data
        filtered_preds = df_preds[
            (df_preds['ev_raw'] * 100 >= min_ev) & 
            (df_preds['prob_raw'] * 100 >= min_conf)
        ].copy()
        
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Matches", len(df_preds))
        col2.metric("Value Bets Found", len(filtered_preds[(filtered_preds['ev_raw'] > 0)]))
        avg_roi = filtered_preds['ev_raw'].mean() * 100 if not filtered_preds.empty else 0
        col3.metric("Avg Expected ROI", f"{avg_roi:.1f}%")
        
        st.subheader(f"Upcoming Predictions ({len(filtered_preds)})")
        
        # Styling
        def style_strength(val):
            color = 'black'
            if 'STRONG' in val: color = 'green'
            elif 'GOOD' in val: color = 'lightgreen'
            elif 'MARGINAL' in val: color = 'orange'
            elif 'NO VALUE' in val: color = 'red'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            filtered_preds.drop(columns=['ev_raw', 'prob_raw']).style.map(style_strength, subset=['Strength']),
            use_container_width=True,
            hide_index=True
        )
        
        # Visuals
        if not filtered_preds.empty:
            st.subheader("Expected Value by Match")
            st.bar_chart(filtered_preds.set_index("Match")['ev_raw'])
            
    else:
        st.warning("No future matches found with sufficient data (Odds) to predict.")
else:
    st.error("Data loading failed.")
