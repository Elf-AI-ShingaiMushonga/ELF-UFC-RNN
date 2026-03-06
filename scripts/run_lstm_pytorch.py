import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD AND PREP THE SEQUENTIAL DATA
# ==========================================
from pathlib import Path
print("Loading raw UFC data...")
# Load the rawest data you have (we need individual fight performances, not pre-fight averages!)
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT_DIR / "data" / "ufc_lstm_sequences.csv"
df = pd.read_csv(OUTPUT_CSV)

# Drop draws and no-contests
df = df[df['outcome_label'].isin(['fighter_1_win', 'fighter_2_win'])].copy()

# Automatically detect the sequence and feature counts
# (We know it's 5 sequences, 12 features, but this is future-proof!)
SEQ_LENGTH = 5
NUM_FEATURES = 12

f1_cols = [f'f1_seq_{s}_stat_{f}' for s in range(SEQ_LENGTH) for f in range(NUM_FEATURES)]
f2_cols = [f'f2_seq_{s}_stat_{f}' for s in range(SEQ_LENGTH) for f in range(NUM_FEATURES)]

# Drop rows if they somehow have missing data
df = df.dropna(subset=f1_cols + f2_cols).copy()

# ==========================================
# 2. THE 50% SWAP (MANDATORY BALANCE)
# ==========================================
print("Balancing Matchups (50% Swap)...")
np.random.seed(42)
swap_idx = np.random.choice(df.index, size=int(len(df)/2), replace=False)

# Swap Fighter 1's history with Fighter 2's history
temp_f1 = df.loc[swap_idx, f1_cols].copy()
df.loc[swap_idx, f1_cols] = df.loc[swap_idx, f2_cols].values
df.loc[swap_idx, f2_cols] = temp_f1.values

# Create the Target Label (1 if F1 wins, 0 if F2 wins)
df['target'] = (df['outcome_label'] == 'fighter_1_win').astype(int)
df.loc[swap_idx, 'target'] = 0

df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# ==========================================
# 3. SPLITTING AND 3D SCALING
# ==========================================
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# We scale the flattened 2D arrays, and PyTorch will reshape them into 3D later
scaler = StandardScaler()
X_train_f1 = scaler.fit_transform(train_df[f1_cols].values)
X_train_f2 = scaler.transform(train_df[f2_cols].values)
X_test_f1 = scaler.transform(test_df[f1_cols].values)
X_test_f2 = scaler.transform(test_df[f2_cols].values)

y_train = train_df['target'].values
y_test = test_df['target'].values

# ==========================================
# 4. THE 3D PYTORCH DATASET
# ==========================================
class LSTMDataset(Dataset):
    def __init__(self, f1_flat, f2_flat, labels, seq_len, num_features):
        # RESHAPE MAGIC: Fold the flat 60-stat array into (5 fights, 12 stats)
        self.f1 = torch.tensor(f1_flat, dtype=torch.float32).view(-1, seq_len, num_features)
        self.f2 = torch.tensor(f2_flat, dtype=torch.float32).view(-1, seq_len, num_features)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.f1[idx], self.f2[idx], self.labels[idx]

train_loader = DataLoader(LSTMDataset(X_train_f1, X_train_f2, y_train, SEQ_LENGTH, NUM_FEATURES), batch_size=64, shuffle=True)
test_loader = DataLoader(LSTMDataset(X_test_f1, X_test_f2, y_test, SEQ_LENGTH, NUM_FEATURES), batch_size=64, shuffle=False)

# ==========================================
# 5. THE SIAMESE LSTM ARCHITECTURE
# ==========================================
class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(SiameseLSTM, self).__init__()
        
        # The LSTM Brain (Processes sequences chronologically)
        # batch_first=True tells it our data is shaped (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # The Matchmaker (Compares the FINAL thoughts of both LSTMs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        # x shape: (Batch, 5, 12)
        lstm_out, (hidden, cell) = self.lstm(x)
        # We only care about the LSTM's conclusion AFTER reading the final fight
        # hidden shape: (num_layers, Batch, hidden_size). We grab the very last layer [-1]
        return hidden[-1] 

    def forward(self, x_f1, x_f2):
        # Both fighters pass through the exact same LSTM weights
        f1_thought = self.forward_one(x_f1)
        f2_thought = self.forward_one(x_f2)
        
        # Combine their timelines and predict
        combined = torch.cat((f1_thought, f2_thought), dim=1)
        return self.classifier(combined)

# ==========================================
# 6. TRAINING WITH EARLY STOPPING
# ==========================================
print("Initializing Siamese LSTM...")
model = SiameseLSTM(input_size=NUM_FEATURES, hidden_size=128, num_layers=4, dropout_rate=0.314)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0018, weight_decay=1e-4)

max_epochs = 500
patience = 50
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "champion_lstm_model.pth"

print(f"Training on {len(train_df)} matches...\n")

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    for b_f1, b_f2, b_labels in train_loader:
        optimizer.zero_grad()
        preds = model(b_f1, b_f2)
        loss = criterion(preds, b_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)

    # Validation Pass
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for b_f1, b_f2, b_labels in test_loader:
            preds = model(b_f1, b_f2)
            loss = criterion(preds, b_labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(test_loader)
    
    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} <-- New Best!")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at Epoch {epoch+1}!")
            break

# ==========================================
# 7. FINAL EVALUATION
# ==========================================
model.load_state_dict(torch.load(best_model_path))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for b_f1, b_f2, b_labels in test_loader:
        preds = model(b_f1, b_f2)
        predicted_classes = (preds > 0.5).float()
        total += b_labels.size(0)
        correct += (predicted_classes == b_labels).sum().item()

print(f"\n======================================")
print(f"🏆 FINAL LSTM ACCURACY: {correct / total:.4f} 🏆")
print(f"======================================")