import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed

# ----------------------------
# Load all models to ensemble
# ----------------------------

model_paths = [
    "outputs/fixed_ecg_model.keras",
    "models/lead1_model_full_hpo_V3.keras",
    "models/lead1_model_full_hpo_V4.keras",
    "models/lead1_model_full_hpo_V5.keras"
]

print("✅ Loading models for ensemble...")
models = []
for path in model_paths:
    print(f"🔹 Loading model: {path}")
    model = tf.keras.models.load_model(path, custom_objects={"focal_loss_fixed": focal_loss_fixed})
    models.append(model)

# ----------------------------
# Load noisy test set
# ----------------------------

X_test, y_test = load_dataset_from_csv(
    csv_path="data/sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=True,       # Enable noise
    leads=[0]           # Lead I only
)

print("✅ Noisy test data loaded. Shape:", X_test.shape)

# ----------------------------
# Encode labels
# ----------------------------

le = LabelEncoder()
le.fit(y_test)
y_test_encoded = le.transform(y_test)

print("✅ Encoded classes:", list(le.classes_))

# ----------------------------
# Make ensemble predictions
# ----------------------------

print("🔁 Generating ensemble predictions...")
all_preds = [model.predict(X_test) for model in models]
avg_pred_probs = np.mean(all_preds, axis=0)
y_pred = np.argmax(avg_pred_probs, axis=1)

# ----------------------------
# Evaluate accuracy
# ----------------------------

accuracy = np.mean(y_pred == y_test_encoded)
print(f"\n✅ Ensemble Accuracy (Noisy Test): {accuracy:.4f}")

# ----------------------------
# Plot confusion matrix
# ----------------------------

cm = confusion_matrix(y_test_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

print("📊 Displaying confusion matrix...")
disp.plot(cmap='Purples')
plt.title("Confusion Matrix - Ensemble (Noisy Test Set)")
plt.tight_layout()
plt.show()
