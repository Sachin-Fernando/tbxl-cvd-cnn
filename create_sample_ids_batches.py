import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split

# ----------------------------
# Load PTB-XL metadata
# ----------------------------
df = pd.read_csv("../ptbxl-data/ptbxl_database.csv")

# Load SCP statements
scp = pd.read_csv("../ptbxl-data/scp_statements.csv", index_col=0)
diagnostic_codes = scp[scp['diagnostic'] == 1].index

# Parse scp_codes from string to dict
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

def aggregate_superclass(codes):
    classes = []
    for code in codes:
        if code in diagnostic_codes:
            classes.append(scp.loc[code, 'diagnostic_class'])
    return list(set(classes))

df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_superclass)
df = df[df['diagnostic_superclass'].map(lambda x: len(x) > 0)]
df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: x[0])

# Keep only the 3 classes for your dissertation
df = df[df['diagnostic_superclass'].isin(['NORM', 'MI', 'STTC'])]

print("Total selected samples:", len(df))
print(df['diagnostic_superclass'].value_counts())

# ----------------------------
# Split into train and test sets
# ----------------------------

# Reserve 20% for final test set
df_train, df_test = train_test_split(
    df,
    test_size=0.20,
    stratify=df['diagnostic_superclass'],
    random_state=42
)

# Save test set
df_test[['ecg_id', 'filename_lr', 'diagnostic_superclass']].to_csv(
    "sample_ids_test.csv", index=False
)
print(f"✅ Saved sample_ids_test.csv with {len(df_test)} samples.")

# ----------------------------
# Shuffle the training data
# ----------------------------

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Split training data into batches
# ----------------------------

batch_size = 1000  # adjust as needed
num_batches = int(len(df_train) / batch_size) + 1

os.makedirs("batches", exist_ok=True)

for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    df_batch = df_train.iloc[start:end]
    if len(df_batch) == 0:
        continue

    csv_name = f"batches/sample_ids_batch{i+1}.csv"
    df_batch[['ecg_id', 'filename_lr', 'diagnostic_superclass']].to_csv(
        csv_name,
        index=False
    )
    print(f"✅ Saved {csv_name} with {len(df_batch)} rows.")
    print(df_batch['diagnostic_superclass'].value_counts())
    print()
