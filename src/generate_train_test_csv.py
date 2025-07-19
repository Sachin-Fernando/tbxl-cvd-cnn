import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import os

# Paths
PTBXL_DB_PATH = "../ptbxl-data/ptbxl_database.csv"
SCP_STATEMENTS_PATH = "../ptbxl-data/scp_statements.csv"
OUTPUT_TEST = "sample_ids_test.csv"
OUTPUT_TRAIN = "all_batches.csv"

# Load metadata
df = pd.read_csv(PTBXL_DB_PATH)
scp = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)

# Extract diagnostic codes only
diagnostic_codes = scp[scp['diagnostic'] == 1].index
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

# Create superclass
def extract_superclass(codes):
    classes = []
    for code in codes:
        if code in diagnostic_codes:
            classes.append(scp.loc[code, 'diagnostic_class'])
    return list(set(classes))

df['diagnostic_superclass'] = df['scp_codes'].apply(extract_superclass)
df = df[df['diagnostic_superclass'].map(lambda x: len(x) > 0)]
df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: x[0])

# Filter to 3 classes
df = df[df['diagnostic_superclass'].isin(['NORM', 'MI', 'STTC'])]

# Stratified split
df_train, df_test = train_test_split(
    df,
    test_size=0.20,
    stratify=df['diagnostic_superclass'],
    random_state=42
)

# Save CSVs
df_test[['ecg_id', 'filename_lr', 'diagnostic_superclass']].to_csv(OUTPUT_TEST, index=False)
df_train[['ecg_id', 'filename_lr', 'diagnostic_superclass']].to_csv(OUTPUT_TRAIN, index=False)

# Summary
print("✅ sample_ids_test.csv saved:", len(df_test))
print("✅ all_batches.csv saved:", len(df_train))
print("\n✅ Train distribution:\n", df_train['diagnostic_superclass'].value_counts())
print("\n✅ Test distribution:\n", df_test['diagnostic_superclass'].value_counts())
