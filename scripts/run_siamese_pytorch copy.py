import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna

# ==========================================
# 1. LOAD PRE-PROCESSED DATA (Same as before)
# ==========================================
print("Loading data for Optuna Optimization...")
df = pd.read_csv("/Users/shingaimushonga/Desktop/UFC_Elf/" + "/data" + "/ufc_fights_cleaned.csv") # Point this to your newly scraped file


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

df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

scaler = StandardScaler()
# Use .values to strip feature names and avoid Scikit-Learn errors!
X_train_f1 = scaler.fit_transform(train_df[f1_cols].values)
X_train_f2 = scaler.transform(train_df[f2_cols].values)
X_test_f1 = scaler.transform(test_df[f1_cols].values)
X_test_f2 = scaler.transform(test_df[f2_cols].values)

y_train = train_df['target'].values
y_test = test_df['target'].values

class UFCDataset(Dataset):
    def __init__(self, f1, f2, labels):
        self.f1 = torch.tensor(f1, dtype=torch.float32)
        self.f2 = torch.tensor(f2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.f1[idx], self.f2[idx], self.labels[idx]

train_loader = DataLoader(UFCDataset(X_train_f1, X_train_f2, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(UFCDataset(X_test_f1, X_test_f2, y_test), batch_size=64, shuffle=False)

# ==========================================
# 2. DYNAMIC CROSS-ATTENTION MODEL
# ==========================================
class DynamicCrossAttentionSiameseUFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_layers):
        super(DynamicCrossAttentionSiameseUFC, self).__init__()
        
        # 1. Build the Encoder dynamically
        encoder_layers = []
        
        # The first layer always goes from input_dim -> hidden_dim
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.BatchNorm1d(hidden_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout_rate))
        
        # Loop to add extra hidden layers if Optuna asks for them
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            
        # Unpack the list into a Sequential block using the * operator
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 2. The Cross-Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 3. The Matchmaker (We'll keep this fixed to avoid too much instability, 
        # but you could dynamically loop this too!)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_f1, x_f2):
        enc_f1 = self.encoder(x_f1)
        enc_f2 = self.encoder(x_f2)
        
        seq_f1 = enc_f1.unsqueeze(1) 
        seq_f2 = enc_f2.unsqueeze(1)
        
        attn_f1, _ = self.attention(query=seq_f1, key=seq_f2, value=seq_f2)
        attn_f2, _ = self.attention(query=seq_f2, key=seq_f1, value=seq_f1)
        
        combined = torch.cat((attn_f1.squeeze(1), attn_f2.squeeze(1)), dim=1)
        return self.classifier(combined)
# ==========================================
# 3. THE OPTUNA OBJECTIVE FUNCTION
# ==========================================
# ==========================================
# 5. THE CHAMPION MODEL RUN
# ==========================================
print(f"Initializing the Champion Optuna Architecture...")

# Plug in your exact winning parameters!
BEST_DIM = 128
BEST_DROPOUT = 0.314
BEST_LAYERS = 4
STATIC_LR = 0.0018
STATIC_WEIGHT_DECAY = 1e-4

model = DynamicCrossAttentionSiameseUFC(
    input_dim=len(base_features), 
    hidden_dim=BEST_DIM, 
    dropout_rate=BEST_DROPOUT, 
    num_layers=BEST_LAYERS
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=STATIC_LR, weight_decay=STATIC_WEIGHT_DECAY)

# --- EARLY STOPPING SETUP ---
max_epochs = 300           # We set this crazy high because Early Stopping will catch it
patience = 25              # How many epochs we tolerate NO improvement before killing it
best_val_loss = float('inf') 
patience_counter = 0
best_model_path = "champion_ufc_model.pth"

print(f"Training Champion Model on {len(train_df)} fights...\n")

for epoch in range(max_epochs):
    # --- 1. TRAINING PASS ---
    model.train()
    total_train_loss = 0
    
    for b_f1, b_f2, b_labels in train_loader:
        optimizer.zero_grad()
        preds = model(b_f1, b_f2)
        loss = criterion(preds, b_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)

    # --- 2. VALIDATION PASS (The Watchdog) ---
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad(): # Don't learn from the test data! Just look at it.
        for b_f1, b_f2, b_labels in test_loader:
            preds = model(b_f1, b_f2)
            val_loss = criterion(preds, b_labels)
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(test_loader)
    
    # --- 3. EARLY STOPPING LOGIC ---
    if avg_val_loss < best_val_loss:
        # We found a new global minimum!
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path) # Save this exact brain state
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} <-- New Best! Saved.")
    else:
        # The AI didn't improve.
        patience_counter += 1
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at Epoch {epoch+1}! The AI started to overfit.")
            print(f"Reverting back to the best brain state from {patience} epochs ago...")
            break # Kill the training loop!

# ==========================================
# 6. FINAL EVALUATION
# ==========================================
# Load the ultimate best version of the brain that we saved during Early Stopping
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
print(f"🏆 FINAL CHAMPION ACCURACY: {correct / total:.4f} 🏆")
print(f"======================================")