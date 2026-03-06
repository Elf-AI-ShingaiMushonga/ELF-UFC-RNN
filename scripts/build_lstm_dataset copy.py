from pathlib import Path

import pandas as pd

print("Loading raw UFC data...")
# Load the rawest data you have (we need individual fight performances, not pre-fight averages!)
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "data" / "ufc_fight_details_lstm.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "ufc_lstm_sequences.csv"
df = pd.read_csv(INPUT_CSV)

# Ensure chronological order (CRITICAL for Time-Series)
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# 1. Which stats do we want to track over time? 
# Note: We don't want 'pre' stats anymore. We want what they ACTUALLY did in the cage that night.
# These are raw per-fight columns from ufc_fight_details_lstm.csv.
# For fighter_2 we swap sides so each fighter vector remains "self vs opponent" aligned.
performance_stats = [
    'kd_1',
    'kd_2',
    'sig_str_1_landed',
    'sig_str_1_attempted',
    'sig_str_2_landed',
    'sig_str_2_attempted',
    'td_1_landed',
    'td_1_attempted',
    'td_2_landed',
    'td_2_attempted',
    'sub_1',
    'ctrl_seconds_1',
]


def swap_fighter_side(col_name):
    if "_1_" in col_name:
        return col_name.replace("_1_", "_2_")
    if "_2_" in col_name:
        return col_name.replace("_2_", "_1_")
    if col_name.endswith("_1"):
        return col_name[:-2] + "_2"
    if col_name.endswith("_2"):
        return col_name[:-2] + "_1"
    return col_name


f1_stat_columns = performance_stats
f2_stat_columns = [swap_fighter_side(col) for col in performance_stats]
required_stat_columns = sorted(set(f1_stat_columns + f2_stat_columns))
missing_columns = [col for col in required_stat_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required fight stat columns: {missing_columns}")
for col in required_stat_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# 2. Initialize our Fighter History Dictionary
# This will hold a list of historical performances for every fighter ID
fighter_histories = {}

# 3. Setting the Sequence Length
SEQ_LENGTH = 5

print("Constructing Time-Series Sequences. This might take a minute...")

# We will store our new 3D-ready rows here
lstm_dataset = []

for index, row in df.iterrows():
    f1_id = row['fighter_1_id']
    f2_id = row['fighter_2_id']
    
    # --- A. Retrieve their past sequences BEFORE this fight ---
    # If they aren't in the dictionary yet, give them an empty list
    f1_history = fighter_histories.get(f1_id, [])
    f2_history = fighter_histories.get(f2_id, [])
    
    # We only want the LAST 5 fights (SEQ_LENGTH)
    f1_seq = f1_history[-SEQ_LENGTH:] if len(f1_history) >= SEQ_LENGTH else f1_history
    f2_seq = f2_history[-SEQ_LENGTH:] if len(f2_history) >= SEQ_LENGTH else f2_history
    
    # --- B. Pad with Zeros if they have less than 5 fights ---
    # We pad at the BEGINNING so the most recent fight is always at the end of the array
    num_features = len(performance_stats)
    
    if len(f1_seq) < SEQ_LENGTH:
        padding = [[0] * num_features for _ in range(SEQ_LENGTH - len(f1_seq))]
        f1_seq = padding + f1_seq
        
    if len(f2_seq) < SEQ_LENGTH:
        padding = [[0] * num_features for _ in range(SEQ_LENGTH - len(f2_seq))]
        f2_seq = padding + f2_seq
        
    # --- C. Save this matchup to our new dataset ---
    # We flatten the sequences into a single row for now so we can save it to a CSV.
    # We will reshape it back into a 3D Tensor inside PyTorch later!
    matchup_data = {
        'fight_id': row['fight_id'],
        'event_date': row['event_date'],
        'outcome_label': row['outcome_label'],
        'f1_id': f1_id,
        'f2_id': f2_id
    }
    
    # Flatten F1's 5 fights into columns (e.g., f1_fight_1_sig_str, f1_fight_2_sig_str...)
    for step, fight_stats in enumerate(f1_seq):
        for stat_idx, stat_value in enumerate(fight_stats):
            matchup_data[f'f1_seq_{step}_stat_{stat_idx}'] = stat_value
            
    # Flatten F2's 5 fights into columns
    for step, fight_stats in enumerate(f2_seq):
        for stat_idx, stat_value in enumerate(fight_stats):
            matchup_data[f'f2_seq_{step}_stat_{stat_idx}'] = stat_value
            
    lstm_dataset.append(matchup_data)
    
    # --- D. Update their histories AFTER the fight happens ---
    # Now that the fight is over, extract what they actually did in the cage today
    # and append it to their permanent record for the NEXT time they fight.
    
    f1_performance_today = [float(row[col]) for col in f1_stat_columns]
    f2_performance_today = [float(row[col]) for col in f2_stat_columns]

    fighter_histories.setdefault(f1_id, []).append(f1_performance_today)
    fighter_histories.setdefault(f2_id, []).append(f2_performance_today)

print("Finished sliding window construction!")
# Convert back to a DataFrame and save
lstm_df = pd.DataFrame(lstm_dataset)
lstm_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(lstm_df)} sequential matchups!")
