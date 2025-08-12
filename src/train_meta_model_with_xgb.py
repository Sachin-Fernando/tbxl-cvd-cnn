import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import optuna
from extract import load_dataset_from_csv

# Load data (stacked base model probabilities)
X, y_labels = load_dataset_from_csv("data/all_batches.csv")

# Load your base models and get probabilities (same as your code)
from tensorflow.keras.models import load_model
from focal_loss import focal_loss_fixed

model_mi_norm = load_model("models/ensemble/model_mi_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_sttc_norm = load_model("models/ensemble/model_sttc_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_mi_sttc = load_model("models/ensemble/model_mi_sttc.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_multiclass = load_model("models/lead1_model_full_hpo_V5.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

def get_model_probs(model, X):
    return model.predict(X, verbose=0)

probs_mi_norm = get_model_probs(model_mi_norm, X)
probs_sttc_norm = get_model_probs(model_sttc_norm, X)
probs_mi_sttc = get_model_probs(model_mi_sttc, X)
probs_multiclass = get_model_probs(model_multiclass, X)

X_stack = np.concatenate([
    probs_mi_norm,
    probs_sttc_norm,
    probs_mi_sttc,
    probs_multiclass
], axis=1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_stack, y, test_size=0.2, random_state=42, stratify=y)

# Optuna objective function
def objective(trial):
    param = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "booster": "gbtree",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "verbosity": 0,
        "seed": 42
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        param,
        dtrain,
        evals=[(dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
        num_boost_round=200,
    )

    preds = bst.predict(dval)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_val, pred_labels)
    return acc

# Run Optuna study
import optuna

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best params on all data
best_params = trial.params
best_params.update({
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "tree_method": "hist",
    "verbosity": 0,
    "seed": 42
})

dall = xgb.DMatrix(X_stack, label=y)
final_model = xgb.train(best_params, dall, num_boost_round=study.best_trial.user_attrs.get("best_iteration", 100))

# Save model
final_model.save_model("models/ensemble/meta_classifier_xgb.json")
print("Saved final XGBoost meta-classifier to 'models/ensemble/meta_classifier_xgb.json'")

# Evaluate on validation split again
dval = xgb.DMatrix(X_val)
preds_val = final_model.predict(dval)
pred_labels_val = np.argmax(preds_val, axis=1)

print("\nClassification Report on Validation Set:")
print(classification_report(y_val, pred_labels_val, target_names=label_encoder.classes_))
