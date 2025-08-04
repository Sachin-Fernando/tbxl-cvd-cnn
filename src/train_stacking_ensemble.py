import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed
import keras_tuner as kt

# ✅ Load test set
X, y_labels = load_dataset_from_csv("data/all_batches.csv")

# 🎯 Encode true labels
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_labels)
y_cat = to_categorical(y_true, num_classes=3)  # Keras expects one-hot

# 🧠 Load binary models with custom loss
model_mi_norm = load_model("models/ensemble/model_mi_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_sttc_norm = load_model("models/ensemble/model_sttc_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_mi_sttc = load_model("models/ensemble/model_mi_sttc.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

# 🔄 Load multi-class CNN model
model_multiclass = load_model("models/lead1_model_full_hpo_V5.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

# ⚙️ Helper to get model probabilities
def get_model_probs(model, X):
    return model.predict(X, verbose=0)

# 📦 Get all model outputs
probs_mi_norm = get_model_probs(model_mi_norm, X)
probs_sttc_norm = get_model_probs(model_sttc_norm, X)
probs_mi_sttc = get_model_probs(model_mi_sttc, X)
probs_multiclass = get_model_probs(model_multiclass, X)

# 🔗 Stack features: shape = (N, 9)
X_stack = np.concatenate([
    probs_mi_norm,
    probs_sttc_norm,
    probs_mi_sttc,
    probs_multiclass
], axis=1)

# 📊 Split data
X_train, X_val, y_train, y_val = train_test_split(X_stack, y_cat, test_size=0.2, random_state=42)

# 🧠 HyperModel for tuner
def build_meta_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(Dense(
            units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16),
            activation=hp.Choice("activation", ["relu", "tanh"])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    
    model.add(Dense(3, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

# 🧪 Run tuner from scratch (no reuse!)
tuner = kt.RandomSearch(
    build_meta_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner_logs_fresh",
    project_name="meta_model_v2"
)

# 🔍 Search
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# ✅ Get best model
best_meta_model = tuner.get_best_models(num_models=1)[0]

# 💾 Save best model
best_meta_model.save("models/ensemble/meta_classifier_hpo_v2.keras")
print("✅ Saved best Keras meta-classifier to 'models/ensemble/meta_classifier_hpo_v2.keras'")

# 📈 Evaluate
y_pred_probs = best_meta_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_val_true = np.argmax(y_val, axis=1)

acc = accuracy_score(y_val_true, y_pred)
print(f"✅ Best HPO Meta Accuracy: {acc * 100:.2f}%")

print("\n📄 Classification Report:")
print(classification_report(y_val_true, y_pred, target_names=label_encoder.classes_))
