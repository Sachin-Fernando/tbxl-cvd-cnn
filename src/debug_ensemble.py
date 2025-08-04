import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from extract import load_dataset_from_csv
import os

# Load the test CSV
df_all = pd.read_csv("data/sample_ids_test.csv")

# Path to the ECG signal files
BASE_PATH = "../ptbxl-data/"

# Evaluation function
def evaluate_model(model_path, df, class_1, class_2):
    # Filter for the two classes
    df = df[df['diagnostic_superclass'].isin([class_1, class_2])]

    if df.empty:
        print(f"⚠️ No samples found for {class_1} vs {class_2}")
        return

    # Save filtered CSV temporarily
    tmp_path = "tmp_eval.csv"
    df.to_csv(tmp_path, index=False)

    # Load ECG data and labels
    X, y = load_dataset_from_csv(csv_path=tmp_path, base_path=BASE_PATH, augment=False, leads=[0])

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Predict
    preds = model.predict(X, verbose=0)
    y_pred = preds.argmax(axis=1)

    # Accuracy
    acc = accuracy_score(y_enc, y_pred)
    print(f"✅ {class_1} vs {class_2} Accuracy: {acc * 100:.2f}%")

    # Cleanup temp file
    os.remove(tmp_path)

# Run evaluations
evaluate_model("models/ensemble/model_mi_norm.keras", df_all, "MI", "NORM")
evaluate_model("models/ensemble/model_sttc_norm.keras", df_all, "STTC", "NORM")
evaluate_model("models/ensemble/model_mi_sttc.keras", df_all, "MI", "STTC")
