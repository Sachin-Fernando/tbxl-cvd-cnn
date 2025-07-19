import pandas as pd
import ast
from collections import Counter

# Load metadata
df = pd.read_csv("../ptbxl-data/ptbxl_database.csv")

# Extract SCP codes and map to superclass diagnostic category
scp_statements = pd.read_csv("../ptbxl-data/scp_statements.csv", index_col=0)
diagnostic_scps = scp_statements[scp_statements['diagnostic'] == 1]
scp_to_superclass = diagnostic_scps['diagnostic_class'].to_dict()

# Count ECGs per superclass
superclass_counter = Counter()

for scp_code_str in df['scp_codes']:
    scp_dict = ast.literal_eval(scp_code_str)
    for code in scp_dict:
        if code in scp_to_superclass:
            superclass = scp_to_superclass[code]
            superclass_counter[superclass] += 1
            break  # Count each ECG only once, based on first matching SCP

# Display result
print("✅ ECG counts by superclass diagnostic category:\n")
for superclass, count in superclass_counter.items():
    print(f"{superclass:<10} : {count}")
