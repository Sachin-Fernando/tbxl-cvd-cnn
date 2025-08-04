import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed

# ✅ Load test set
X, y_labels = load_dataset_from_csv("data/all_batches.csv")

# 🎯 Encode true labels
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_labels)
y_cat = to_categorical(y_true, num_classes=3)  # Keras expects one-hot for softmax output

# 🧠 Load trained binary models with custom loss
model_mi_norm = load_model("models/ensemble/model_mi_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_sttc_norm = load_model("models/ensemble/model_sttc_norm.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_mi_sttc = load_model("models/ensemble/model_mi_sttc.keras", custom_objects={'focal_loss_fixed': focal_loss_fixed})

# ⚙️ Define helper to get probabilities from binary model
def get_model_probs(model, X):
    return model.predict(X, verbose=0)

# 📦 Get all model outputs
probs_mi_norm = get_model_probs(model_mi_norm, X)      # cols: MI, NORM
probs_sttc_norm = get_model_probs(model_sttc_norm, X)  # cols: STTC, NORM
probs_mi_sttc = get_model_probs(model_mi_sttc, X)      # cols: MI, STTC

# 🧱 Stack features
X_stack = np.concatenate([probs_mi_norm, probs_sttc_norm, probs_mi_sttc], axis=1)

# 📊 Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_stack, y_cat, test_size=0.2, random_state=42)

# 🧠 Define Keras meta-classifier
meta_model = Sequential([
    Input(shape=(6,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

meta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️ Train meta-classifier
meta_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# ✅ Save Keras meta-classifier
meta_model.save("models/ensemble/meta_classifier.keras")
print("✅ Saved Keras meta-classifier to models/ensemble/meta_classifier.keras")

# 📈 Evaluate
y_pred_probs = meta_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_val_true = np.argmax(y_val, axis=1)
acc = accuracy_score(y_val_true, y_pred)
print(f"✅ Stacking Ensemble Accuracy: {acc * 100:.2f}%")

print("\n📄 Classification Report:")
print(classification_report(y_val_true, y_pred, target_names=label_encoder.classes_))
