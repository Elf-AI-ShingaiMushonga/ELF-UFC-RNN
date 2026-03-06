import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
print("Loading and preparing data...")
df = pd.read_csv("/Users/shingaimushonga/Desktop/UFC_Elf/" + "/data" + "/ufc_fights_rnn.csv") # Point this to your newly scraped file

# Fix the % bug if your new scraper still exports them as strings
pct_cols = [c for c in df.columns if 'accuracy' in c or 'defense' in c]
for col in pct_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0

# Define the optimal features
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

# Drop rows missing these critical stats
df = df.dropna(subset=f1_cols + f2_cols).copy()

# ==========================================
# 2. BALANCE THE DATASET (50% SWAP)
# ==========================================
print("Balancing matchups...")
df = df[df['outcome_label'].isin(['fighter_1_win', 'fighter_2_win'])].copy()
np.random.seed(42)
swap_idx = np.random.choice(df.index, size=int(len(df)/2), replace=False)

temp_f1 = df.loc[swap_idx, f1_cols].copy()
df.loc[swap_idx, f1_cols] = df.loc[swap_idx, f2_cols].values
df.loc[swap_idx, f2_cols] = temp_f1.values

df['target'] = (df['outcome_label'] == 'fighter_1_win').astype(int)
df.loc[swap_idx, 'target'] = 0

# Sort chronologically for strict Train/Test split
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# ==========================================
# 3. PYTORCH DATASETS
# ==========================================
class UFCDataset(Dataset):
    def __init__(self, f1_data, f2_data, labels):
        self.f1 = torch.tensor(f1_data, dtype=torch.float32)
        self.f2 = torch.tensor(f2_data, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.f1[idx], self.f2[idx], self.labels[idx]

split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

scaler = StandardScaler()
X_train_f1 = scaler.fit_transform(train_df[f1_cols])
X_train_f2 = scaler.transform(train_df[f2_cols]) # Transform only!
X_test_f1 = scaler.transform(test_df[f1_cols])
X_test_f2 = scaler.transform(test_df[f2_cols])

y_train, y_test = train_df['target'], test_df['target']

train_loader = DataLoader(UFCDataset(X_train_f1, X_train_f2, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(UFCDataset(X_test_f1, X_test_f2, y_test), batch_size=64, shuffle=False)

# ==========================================
# 4. SIAMESE NETWORK ARCHITECTURE
# ==========================================
class AdvancedSiameseUFC(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AdvancedSiameseUFC, self).__init__()
        
        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Matchmaker
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_f1, x_f2):
        enc_f1 = self.encoder(x_f1)
        enc_f2 = self.encoder(x_f2)
        combined = torch.cat((enc_f1, enc_f2), dim=1)
        return self.classifier(combined)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
model = AdvancedSiameseUFC(input_dim=len(base_features))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # L2 Regularization

epochs = 40
print(f"\nTraining on {len(train_df)} fights, Testing on {len(test_df)} fights...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for b_f1, b_f2, b_labels in train_loader:
        optimizer.zero_grad()
        preds = model(b_f1, b_f2)
        loss = criterion(preds, b_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{epochs} | BCE Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 6. FINAL EVALUATION
# ==========================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for b_f1, b_f2, b_labels in test_loader:
        preds = model(b_f1, b_f2)
        predicted_classes = (preds > 0.5).float()
        total += b_labels.size(0)
        correct += (predicted_classes == b_labels).sum().item()

print(f"\n======================================")
print(f"Final PyTorch Siamese Accuracy: {correct / total:.3f}")
print(f"======================================")