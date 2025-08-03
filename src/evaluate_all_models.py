import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed

# ----------- Configurable Paths -------------
models = [
    "lead1_model_full_hpo_V2.keras",
    "lead1_model_full_hpo_V3.keras",
    "lead1_model_full_hpo_V4.keras",
    "lead1_model_full_hpo_V5.keras",
    "lead1_model_full_hpo.keras"
]
models_path = "models"
output_dir = "evaluation_outputs"
os.makedirs(output_dir, exist_ok=True)

# ------------- Load test data ----------------
X_test, y_test = load_dataset_from_csv(
    csv_path="data/sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=True,      # noisy test set
    leads=[0]          # Lead I
)

print("✅ Noisy test data loaded:", X_test.shape)

# ------------- Label encoding ----------------
le = LabelEncoder()
le.fit(y_test)
y_test_encoded = le.transform(y_test)
print("✅ Classes:", list(le.classes_))

# ------------- Evaluate All Models ----------
for model_file in models:
    model_path = os.path.join(models_path, model_file)
    short_name = model_file.replace(".keras", "")
    print(f"\n📦 Evaluating: {model_file}")

    # Load model with focal loss
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"focal_loss_fixed": focal_loss_fixed}
    )

    # Evaluate loss/accuracy
    results = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"✅ Accuracy: {results[1]:.4f}")

    # Predict and decode
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {short_name}")
    plt.savefig(f"{output_dir}/{short_name}_confusion_matrix.png")
    plt.close()

    # Classification report
    report_dict = classification_report(
        y_test_encoded, y_pred, target_names=le.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{output_dir}/{short_name}_classification_report.csv")

    # Accuracy summary
    with open(f"{output_dir}/{short_name}_accuracy.txt", "w") as f:
        f.write(f"Accuracy: {results[1]:.4f}\n")

    print(f"📁 Saved confusion matrix and report for: {short_name}")

print("\n✅ All model evaluations complete.")
