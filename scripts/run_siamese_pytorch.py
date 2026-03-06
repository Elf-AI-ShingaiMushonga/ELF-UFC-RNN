import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD PRE-PROCESSED DATA
# ==========================================
print("Loading pre-processed data...")
# Point this to your cleaned and swapped dataset!
df = pd.read_csv("/Users/shingaimushonga/Desktop/UFC_Elf/" + "/data" + "/ufc_fights_cleaned.csv") 

# The exact Top 15 features your Random Forest selected
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

# Ensure chronological sorting to prevent data leakage
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# ==========================================
# 2. CHRONOLOGICAL SPLIT & SCALING
# ==========================================
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Fit the scaler ONLY on Fighter 1's training data. 
# By adding .values, we strip the column names, so Scikit-Learn just sees raw numbers.
scaler = StandardScaler()
X_train_f1 = scaler.fit_transform(train_df[f1_cols].values)

# Transform everything else using that exact same scaling ruler
X_train_f2 = scaler.transform(train_df[f2_cols].values)
X_test_f1 = scaler.transform(test_df[f1_cols].values)
X_test_f2 = scaler.transform(test_df[f2_cols].values)

y_train = train_df['target'].values
y_test = test_df['target'].values
# ==========================================
# 3. PYTORCH DATASET SETUP
# ==========================================
class UFCDataset(Dataset):
    def __init__(self, f1_data, f2_data, labels):
        self.f1 = torch.tensor(f1_data, dtype=torch.float32)
        self.f2 = torch.tensor(f2_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Ensure shape is (Batch, 1)

    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, idx): 
        return self.f1[idx], self.f2[idx], self.labels[idx]

# Create DataLoaders to feed the network in batches of 64
train_loader = DataLoader(UFCDataset(X_train_f1, X_train_f2, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(UFCDataset(X_test_f1, X_test_f2, y_test), batch_size=64, shuffle=False)

# ==========================================
# 4. SIAMESE NETWORK ARCHITECTURE
# ==========================================
class CrossAttentionSiameseUFC(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CrossAttentionSiameseUFC, self).__init__()
        
        # 1. The Encoder (Same as before)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. THE UPGRADE: Multi-Head Attention
        # num_heads=4 means the AI looks at the matchup from 4 different "perspectives" 
        # (e.g., Striking perspective, Grappling perspective, Age/Cardio perspective)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 3. The Matchmaker
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
        # 1. Encode the raw stats
        enc_f1 = self.encoder(x_f1)
        enc_f2 = self.encoder(x_f2)
        
        # PyTorch Attention expects 3D shapes: (Batch_Size, Sequence_Length, Hidden_Dim)
        # Since we just have 1 set of stats per fighter, Sequence Length is 1.
        # We use unsqueeze(1) to add that dimension.
        seq_f1 = enc_f1.unsqueeze(1) # Shape becomes (64, 1, 64)
        seq_f2 = enc_f2.unsqueeze(1)
        
        # 2. CROSS-ATTENTION: Fighter 1 analyzes Fighter 2
        # Query = F1, Key/Value = F2
        attn_f1, _ = self.attention(query=seq_f1, key=seq_f2, value=seq_f2)
        
        # 3. CROSS-ATTENTION: Fighter 2 analyzes Fighter 1
        # Query = F2, Key/Value = F1
        attn_f2, _ = self.attention(query=seq_f2, key=seq_f1, value=seq_f1)
        
        # 4. Remove the sequence dimension with squeeze(1)
        final_f1 = attn_f1.squeeze(1)
        final_f2 = attn_f2.squeeze(1)
        
        # 5. Concatenate the specifically "attended" vectors and classify
        combined = torch.cat((final_f1, final_f2), dim=1)
        return self.classifier(combined)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print(f"Initializing Neural Network...")
model = CrossAttentionSiameseUFC(input_dim=len(base_features))

# Loss Function and Optimizer
criterion = nn.BCELoss()
# Weight Decay (1e-4) applies L2 Regularization to stop massive weights from forming
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

epochs = 40
print(f"Training on {len(train_df)} fights, Testing on {len(test_df)} fights...\n")

for epoch in range(epochs):
    model.train() # Enable Dropout and BatchNorm
    total_loss = 0
    
    for b_f1, b_f2, b_labels in train_loader:
        optimizer.zero_grad() # Clear previous gradients
        
        preds = model(b_f1, b_f2) # Forward pass
        loss = criterion(preds, b_labels) # Calculate Error
        
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        
        total_loss += loss.item()
        
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | BCE Loss: {avg_loss:.4f}")

# ==========================================
# 6. FINAL EVALUATION
# ==========================================
model.eval() # Disable Dropout for accurate testing
correct = 0
total = 0

with torch.no_grad(): # Disable gradient tracking to save memory
    for b_f1, b_f2, b_labels in test_loader:
        preds = model(b_f1, b_f2)
        # Convert probabilities: > 0.5 means F1 wins, <= 0.5 means F2 wins
        predicted_classes = (preds > 0.5).float() 
        
        total += b_labels.size(0)
        correct += (predicted_classes == b_labels).sum().item()

test_accuracy = correct / total
print(f"\n======================================")
print(f"Final PyTorch Siamese Accuracy: {test_accuracy:.3f}")
print(f"======================================")