import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from extract import load_dataset_from_csv

# ----------------------------
# Load your saved model
# ----------------------------

model_path = "models/lead1_model_full_hpo.keras"

print(f"✅ Loading model from: {model_path}")
model = keras.models.load_model(model_path)

# ----------------------------
# Load test set
# ----------------------------

X_test, y_test = load_dataset_from_csv(
    csv_path="sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=False,
    leads=[0]
)

print("✅ Test data loaded:")
print("X_test shape:", X_test.shape)

# ----------------------------
# Encode labels
# ----------------------------

# Fit label encoder on test labels
le = LabelEncoder()
le.fit(y_test)
y_test_encoded = le.transform(y_test)

print("✅ Classes:", list(le.classes_))

# ----------------------------
# Evaluate the model
# ----------------------------

results = model.evaluate(X_test, y_test_encoded, verbose=1)

print("\n✅ Test Loss:", results[0])
print("✅ Test Accuracy:", results[1])
