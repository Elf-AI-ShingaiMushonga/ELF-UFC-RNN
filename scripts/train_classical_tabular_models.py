import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt

# 1. Load Data
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "data" / "ufc_fights_rnn.csv"
df = pd.read_csv(INPUT_CSV)

#Convert percentage strings into numbers
pct_cols = [c for c in df.columns if 'accuracy' in c or 'defense' in c]
for col in pct_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100.0

# Calculate the difference in percentage success 
df['sig_str_accuracy_diff_f1_minus_f2'] = df['fighter_1_sig_str_accuracy_pre'] - df['fighter_2_sig_str_accuracy_pre']
df['sig_str_defense_diff_f1_minus_f2'] = df['fighter_1_sig_str_defense_pre'] - df['fighter_2_sig_str_defense_pre']
df['td_accuracy_diff_f1_minus_f2'] = df['fighter_1_td_accuracy_pre'] - df['fighter_2_td_accuracy_pre']
df['td_defense_diff_f1_minus_f2'] = df['fighter_1_td_defense_pre'] - df['fighter_2_td_defense_pre']

# Strategy: Drop any row that doesn't have Striking/Takedown data
critical_columns = [
    'sig_str_landed_per_min_diff_f1_minus_f2',
    'td_landed_per_15_diff_f1_minus_f2'
]
df = df.dropna(subset=critical_columns)
# Now, any REMAINING NaNs are likely just rookies with 0 UFC fights. 
# We can safely fill those with 0 (meaning "average/no data advantage").

feature_cols = [col for col in df.columns if 'diff' in col]
df[feature_cols] = df[feature_cols].fillna(0)

# 2. Sort Chronologically (Crucial for preventing time-travel leakage)
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# 3. Select Features (The diffs are perfect for classical ML)
feature_cols = [col for col in df.columns if 'diff' in col]

# 4. Target Variable (1 if Fighter 1 wins, 0 otherwise)
np.random.seed(42) 
swap_indices = np.random.choice(df.index, size=int(len(df)/2), replace=False)
df.loc[swap_indices, feature_cols] = df.loc[swap_indices, feature_cols] * -1

df['target'] = 1
df.loc[swap_indices, 'target'] = 0
X = df[feature_cols]
y = df['target']

# 5. Handle Missing Data (Impute with 0, meaning 'no difference')
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer.fit_transform(X)

# 6. Chronological Train/Test Split (80% past fights, 20% future fights)
split_idx = int(len(X_imputed) * 0.8)
X_train, X_test = X_imputed[:split_idx], X_imputed[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 7. Scale Data (Required for SVM to calculate Margins correctly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Random Trees

'''# --- 8.1. Define the Grid of Hyperparameters to test ---
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10] # How many samples needed to make a split
}

# --- 8.2. Set up the Chronological Cross-Validator ---
# This ensures we never use future fights to predict past fights during tuning
tscv = TimeSeriesSplit(n_splits=5)

# --- 3. Set up the Grid Search ---
print("Initiating Grid Search for Random Forest... (This might take a minute)")
rf_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=tscv,          # Use the Time Series Splitter!
    scoring='accuracy',
    n_jobs=-1         # Use all your computer's CPU cores to do it faster
)

# --- 8.4. Run the Search ---
# We fit it on the training data. It will test all combinations.
grid_search.fit(X_train, y_train)
print("\n=== GRID SEARCH RESULTS ===")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}")

# --- 8.5. Test the Best Model on the Unseen Future Data ---
best_rf = grid_search.best_estimator_
final_preds = best_rf.predict(X_test)
print(f"\nFinal Test Accuracy (on future unseen fights): {accuracy_score(y_test, final_preds):.3f}")


# --- Train Random Forest ---, best params is 200, max_depth = 5, sample split = 2
'''
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, min_samples_split=2)
rf.fit(X_train, y_train) 
rf_preds = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.3f}")


# Assuming 'rf' is your trained Random Forest and 'X' is your DataFrame of features
importances = rf.feature_importances_
feature_names = X.columns

# Sort them from most important to least important
indices = np.argsort(importances)[::-1]

print("\n--- TOP 15 MOST IMPORTANT FEATURES ---")
for i in range(15):
    # Print rank, name, and the percentage of importance
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]] * 100:.2f}%")

# Optional: Plot them on a bar chart
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importance")
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.show()

# --- Train SVM (Margin Maximization) ---
print("Training Support Vector Machine (RBF Kernel)...")
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(X_train_scaled, y_train) # SVM absolutely requires scaling
svm_preds = svm.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds):.3f}")
