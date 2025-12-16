import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate():
    # 1. Load Data
    try:
        df = pd.read_csv('sofascore_dataset_v2.csv')
        print(f"Loaded dataset with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Preprocessing
    # Target
    def get_result(row):
        if row['score_home'] > row['score_away']: return 0 # Home
        elif row['score_home'] < row['score_away']: return 2 # Away
        else: return 1 # Draw
    
    df['target'] = df.apply(get_result, axis=1)
    
    # Drop non-numeric / leakage columns
    drop_cols = ['id', 'date', 'league', 'home', 'away', 'status', 'score_home', 'score_away', 'target']
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # Handle NaNs
    # For streaks, NaN means 0 (no streak). For odds/ratings, maybe mean?
    # Let's fill 0 for everything for now as streaks are sparse.
    X = X.fillna(0)
    
    # Convert 'form' strings to something numeric if present?
    # Our scraper keeps them as strings "W,L,D". Let's drop them for this quick test or simple count.
    # Dropping complex object columns
    X = X.select_dtypes(include=['number'])
    
    print(f"Features: {X.shape[1]}")
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # 6. Feature Importance
    importance = model.feature_importances_
    feat_names = X.columns
    
    # Create DataFrame
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    
    print("\nTop 10 Features:")
    print(fi_df.head(10))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title('Top 20 Feature Importance (with Streaks)')
    plt.tight_layout()
    plt.savefig('reports/test_model_performance.png')
    print("\nSaved feature importance plot to reports/test_model_performance.png")

if __name__ == "__main__":
    train_and_evaluate()
