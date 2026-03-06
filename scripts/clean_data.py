import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# PART 1: DATA CLEANING & EXPORT
# ==========================================
print("--- Starting Data Pipeline ---")

# 1. Load Raw Data
df = pd.read_csv("/Users/shingaimushonga/Desktop/UFC_Elf/" + "/data" + "/ufc_fights_rnn.csv")

# 2. Fix the Percentage Bug (Strings to Floats)
pct_cols = [c for c in df.columns if 'accuracy' in c or 'defense' in c]
for col in pct_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0

df['sig_str_accuracy_diff_f1_minus_f2'] = df['fighter_1_sig_str_accuracy_pre'] - df['fighter_2_sig_str_accuracy_pre']
df['sig_str_defense_diff_f1_minus_f2'] = df['fighter_1_sig_str_defense_pre'] - df['fighter_2_sig_str_defense_pre']
df['td_accuracy_diff_f1_minus_f2'] = df['fighter_1_td_accuracy_pre'] - df['fighter_2_td_accuracy_pre']
df['td_defense_diff_f1_minus_f2'] = df['fighter_1_td_defense_pre'] - df['fighter_2_td_defense_pre']

# 4. Define the core stats we actually want the AI to look at
# We are using 'pre' stats because we want their history BEFORE the fight
base_features = [
    'age_days', 'win_rate_pre', 'td_landed_per_15_pre', 
    'sig_str_absorbed_per_min_pre', 'days_since_last_fight',
    'losses_pre', 'sig_str_defense_pre', 'wins_pre',
    'control_time_per_min_pre', 'avg_fight_duration_sec_pre',
    'td_absorbed_per_15_pre', 'reach_cm', 'win_streak_pre',
    'sub_attempts_per_15_pre', 'sig_str_accuracy_pre'
]

f1_cols = [f'fighter_1_{col}' for col in base_features]
f2_cols = [f'fighter_2_{col}' for col in base_features]

# 5. Drop rows missing critical stats (No data = no prediction)
df = df.dropna(subset=f1_cols + f2_cols).copy()
print(len(df))
# 6. Balance the Dataset (The 50% Swap)
# We must teach the model what a "Fighter 2 Win" looks like
df = df[df['outcome_label'].isin(['fighter_1_win', 'fighter_2_win'])].copy()
np.random.seed(42)
swap_idx = np.random.choice(df.index, size=int(len(df)/2), replace=False)

# Swap the features
temp_f1 = df.loc[swap_idx, f1_cols].copy()
df.loc[swap_idx, f1_cols] = df.loc[swap_idx, f2_cols].values
df.loc[swap_idx, f2_cols] = temp_f1.values

# Swap the Target Label
df['target'] = (df['outcome_label'] == 'fighter_1_win').astype(int)
df.loc[swap_idx, 'target'] = 0

# 7. Sort Chronologically & Save the Cleaned Dataset
df = df.sort_values('event_date').reset_index(drop=True)
clean_path ="/Users/shingaimushonga/Desktop/UFC_Elf/" + "/data" + '/ufc_fights_cleaned.csv'
df.to_csv(clean_path, index=False)
print(f"Clean data saved to {clean_path}. Total valid fights: {len(df)}")


# ==========================================
# PART 2: PYTORCH SETUP & MODEL
# ==========================================