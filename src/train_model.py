import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed
import keras_tuner as kt

# ----------------------------
# Early stopping & LR schedule
# ----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ----------------------------
# Load Data
# ----------------------------
data_path = "data/all_batches.csv"
print(f"✅ Loading data from: {data_path}")

X, y = load_dataset_from_csv(
    csv_path=data_path,
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

# ----------------------------
# Shuffle & Oversample
# ----------------------------
perm = np.random.permutation(len(X))
X = X[perm]
y = np.array(y)[perm]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)

print("✅ Classes:", list(le.classes_))

X_flat = X.reshape((X.shape[0], -1))
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y_encoded)
X_resampled = X_resampled.reshape((-1, 1000, 1))

# ----------------------------
# Class Weights
# ----------------------------
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weights_dict = dict(zip(np.unique(y_resampled), weights))
print("✅ Class weights:", class_weights_dict)

# ----------------------------
# Residual Block
# ----------------------------
def residual_block(x, filters, kernel_size):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding='same')(shortcut)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# ----------------------------
# Model Builder
# ----------------------------
def build_model(hp, n_classes):
    inputs = keras.Input(shape=(1000, 1))

    filters_initial = hp.Int("filters_initial", 16, 128, step=16)
    x = layers.Conv1D(filters_initial, kernel_size=7, activation='relu', padding='same')(inputs)

    num_blocks = hp.Int("num_blocks", 1, 3)
    filters = filters_initial
    for i in range(num_blocks):
        filters = hp.Int(f"filters_block_{i}", 32, 128, step=32)
        x = residual_block(x, filters, kernel_size=5)
        if hp.Boolean(f"pool_after_block_{i}"):
            x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)

    dense_units = hp.Int("dense_units", 32, 256, step=32)
    x = layers.Dense(dense_units, activation='relu')(x)

    dropout_rate = hp.Float("dropout", 0.2, 0.5, step=0.1)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=focal_loss_fixed,
        metrics=['accuracy']
    )
    return model

# ----------------------------
# Hyperparameter Tuning
# ----------------------------
tuner = kt.RandomSearch(
    lambda hp: build_model(hp, n_classes=len(classes)),
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="hpo_logs",
    project_name="ecg_full_dataset"
)

tuner.search(
    X_resampled,
    y_resampled,
    epochs=30,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]

# ----------------------------
# Final Training
# ----------------------------
history = best_model.fit(
    X_resampled,
    y_resampled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ----------------------------
# Evaluate and Compare
# ----------------------------
X_test, y_test = load_dataset_from_csv(
    csv_path="data/sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=False,
    leads=[0]
)
y_test_encoded = le.transform(y_test)

new_results = best_model.evaluate(X_test, y_test_encoded, verbose=0)
new_acc = new_results[1]
print("\n✅ New model test accuracy:", new_acc)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
existing_models = sorted(
    glob.glob(os.path.join(model_dir, "lead1_model_full_hpo_V*.keras")),
    key=os.path.getmtime
)

previous_acc = 0
latest_model = None
if existing_models:
    latest_model = existing_models[-1]
    print("✅ Loading previous model:", latest_model)
    old_model = tf.keras.models.load_model(latest_model, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    old_results = old_model.evaluate(X_test, y_test_encoded, verbose=0)
    previous_acc = old_results[1]
    print("✅ Previous model test accuracy:", previous_acc)

# ----------------------------
# Save Decision
# ----------------------------
if new_acc > previous_acc:
    new_version = len(existing_models) + 2
    new_model_path = f"{model_dir}/lead1_model_full_hpo_V{new_version}.keras"
    print("\n📊 New model performed better.")
    print("📁 Suggested save path:", new_model_path)
    save = input("❓ Save this new model? (y/n): ").strip().lower()
    if save == 'y':
        best_model.save(new_model_path)
        print("✅ Model saved as", new_model_path)
    else:
        print("❌ Model was not saved.")
else:
    print("📉 New model did not outperform the previous one. Not saved.")
