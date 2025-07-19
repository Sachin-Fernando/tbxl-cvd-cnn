import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from extract import load_dataset_from_csv
from legacy.train_batches import build_model, early_stop, lr_scheduler

# ----------------------------
# STEP 1 — Merge all batches
# ----------------------------

batch_files = sorted(glob.glob("batches/sample_ids_batch*.csv"))
print("✅ Found batch files:", batch_files)

dfs = [pd.read_csv(f) for f in batch_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("all_batches.csv", index=False)
print("✅ Merged all batches into all_batches.csv")

# ----------------------------
# STEP 2 — Load full dataset
# ----------------------------

X, y = load_dataset_from_csv(
    csv_path="all_batches.csv",
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

print("✅ Loaded full dataset:")
print("X shape:", X.shape)

# ----------------------------
# STEP 3 — Shuffle dataset
# ----------------------------

# Generate random permutation
perm = np.random.permutation(len(X))

X = X[perm]
y = y[perm]

print("✅ Data shuffled.")

# ----------------------------
# STEP 4 — Encode labels
# ----------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)
print("✅ Classes:", list(le.classes_))

# ----------------------------
# STEP 5 — Compute class weights
# ----------------------------

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_encoded
)
class_weights_dict = dict(zip(classes, weights))
print("✅ Class weights:", class_weights_dict)

# ----------------------------
# STEP 6 — Build model
# ----------------------------

model = build_model(len(classes))
model.summary()

# ----------------------------
# STEP 7 — Train model
# ----------------------------

history = model.fit(
    X,
    y_encoded,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ----------------------------
# STEP 8 — Save model
# ----------------------------

model.save("models/lead1_model_full.keras")
print("✅ Final model saved as models/lead1_model_full.keras")

# ----------------------------
# STEP 9 — Evaluate on test set
# ----------------------------

X_test, y_test = load_dataset_from_csv(
    csv_path="sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=False,
    leads=[0]
)

y_test_encoded = le.transform(y_test)

results = model.evaluate(X_test, y_test_encoded, verbose=1)
print("✅ Test Loss:", results[0])
print("✅ Test Accuracy:", results[1])
