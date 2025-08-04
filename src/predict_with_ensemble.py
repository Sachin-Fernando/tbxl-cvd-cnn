# predict_with_ensemble.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from extract import load_dataset_from_csv
from sklearn.preprocessing import LabelEncoder

# --------------------------------------
# Load models
# --------------------------------------
MODEL_DIR = "models/ensemble"
model_mi_norm = load_model(os.path.join(MODEL_DIR, "model_mi_norm.keras"), compile=False)
model_sttc_norm = load_model(os.path.join(MODEL_DIR, "model_sttc_norm.keras"), compile=False)
model_mi_sttc = load_model(os.path.join(MODEL_DIR, "model_mi_sttc.keras"), compile=False)

# --------------------------------------
# Class mapping
# --------------------------------------
classes = ['MI', 'NORM', 'STTC']
class_index = {label: i for i, label in enumerate(classes)}

# --------------------------------------
# Inference logic
# --------------------------------------
def get_final_prediction(ecg_batch):
    """
    Input: ecg_batch shape (n_samples, 1000, 1)
    Output: final_preds (n_samples,) -> 'MI' | 'NORM' | 'STTC'
    """
    preds_mi_norm = model_mi_norm.predict(ecg_batch, verbose=0)
    preds_sttc_norm = model_sttc_norm.predict(ecg_batch, verbose=0)
    preds_mi_sttc = model_mi_sttc.predict(ecg_batch, verbose=0)

    final_preds = []
    final_probs = []

    for i in range(len(ecg_batch)):
        votes = []

        # Add all 3 model predictions with associated class
        votes.append(('MI', preds_mi_norm[i][0]))
        votes.append(('NORM', preds_mi_norm[i][1]))

        votes.append(('STTC', preds_sttc_norm[i][0]))
        votes.append(('NORM', preds_sttc_norm[i][1]))

        votes.append(('MI', preds_mi_sttc[i][0]))
        votes.append(('STTC', preds_mi_sttc[i][1]))

        # Pick the class with the highest confidence
        top_class, top_conf = max(votes, key=lambda x: x[1])
        final_preds.append(top_class)

        # Optional: Construct 3-class probability vector for tracking
        probs = np.zeros(3)
        for label, conf in votes:
            probs[class_index[label]] = max(probs[class_index[label]], conf)
        probs /= probs.sum()  # normalize
        final_probs.append(probs)

    return final_preds, np.array(final_probs)


# --------------------------------------
# Example run on test set
# --------------------------------------
if __name__ == "__main__":
    print("📦 Loading test ECG samples...")
    X_test, y_test = load_dataset_from_csv(
        csv_path="data/sample_ids_test.csv",
        base_path="../ptbxl-data/",
        augment=False,
        leads=[0]
    )

    le = LabelEncoder()
    y_true = le.fit_transform(y_test)
    final_preds, final_probs = get_final_prediction(X_test)

    acc = np.mean(np.array(final_preds) == np.array(y_test))
    print(f"✅ Ensemble Accuracy: {acc * 100:.2f}%")
