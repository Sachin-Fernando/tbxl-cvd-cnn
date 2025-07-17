import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed



# ----------------------------
# Load trained model - use comment load when using focal fucntion
# ----------------------------

model_path = "models/lead1_model_full_hpo_V3.keras" 

print(f"✅ Loading model from {model_path}")

model = tf.keras.models.load_model(
    model_path,
    custom_objects={"focal_loss_fixed": focal_loss_fixed}
)

# model = tf.keras.models.load_model(model_path)

# ----------------------------
# Load noisy test set
# ----------------------------

X_test, y_test = load_dataset_from_csv(
    csv_path="sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=True,       # turn on noise here
    leads=[0]
)

print("✅ Noisy test data loaded:")
print("X_test shape:", X_test.shape)

# ----------------------------
# Encode labels
# ----------------------------

le = LabelEncoder()
le.fit(y_test)
y_test_encoded = le.transform(y_test)

print("✅ Classes:", list(le.classes_))

# ----------------------------
# Evaluate the model
# ----------------------------

results = model.evaluate(X_test, y_test_encoded, verbose=1)

print("\n✅ Noisy Test Loss:", results[0])
print("✅ Noisy Test Accuracy:", results[1])

# ----------------------------
# Predict classes
# ----------------------------

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ----------------------------
# Plot confusion matrix
# ----------------------------

cm = confusion_matrix(y_test_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

print("\n✅ Displaying confusion matrix...")
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Noisy Test Set")
plt.show()
