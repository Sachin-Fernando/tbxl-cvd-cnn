import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from extract import load_dataset_from_csv
from tensorflow.keras.models import load_model
from focal_loss import focal_loss_fixed

# --- Load test data ---
X_test, y_labels_test = load_dataset_from_csv("data/sample_ids_test.csv")

# --- Encode labels ---
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_labels_test)

# --- Load base models ---
model_mi_norm = load_model("models/ensemble/model_mi_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_sttc_norm = load_model("models/ensemble/model_sttc_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_mi_sttc = load_model("models/ensemble/model_mi_sttc.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_multiclass = load_model("models/lead1_model_full_hpo_V5.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

def get_model_probs(model, X):
    return model.predict(X, verbose=0)

# --- Get stacked features ---
probs_mi_norm = get_model_probs(model_mi_norm, X_test)
probs_sttc_norm = get_model_probs(model_sttc_norm, X_test)
probs_mi_sttc = get_model_probs(model_mi_sttc, X_test)
probs_multiclass = get_model_probs(model_multiclass, X_test)

X_stack_test = np.concatenate([
    probs_mi_norm,
    probs_sttc_norm,
    probs_mi_sttc,
    probs_multiclass
], axis=1)

# --- Load XGBoost model ---
xgb_model = xgb.Booster()
xgb_model.load_model("models/ensemble/meta_classifier_xgb.json")

# --- Prepare test data for XGBoost ---
dtest = xgb.DMatrix(X_stack_test)

# --- Predict and evaluate ---
y_pred_probs = xgb_model.predict(dtest)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Meta-Classifier Accuracy on Unseen Test Data: {acc * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
