# train_ensemble_models.py
import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, Input, Model
from focal_loss import focal_loss_fixed
from extract import load_dataset_from_csv
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------------
# Settings
# -------------------------------------
BASE_PATH = "../ptbxl-data/"
CSV_PATH = "data/all_batches.csv"
OUTPUT_DIR = "models/ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LEADS = [0]  # Lead I only

# -------------------------------------
# Utility: Residual Block
# -------------------------------------
def residual_block(x, filters, kernel_size):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding='same')(shortcut)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

# -------------------------------------
# Model Builder for Keras Tuner
# -------------------------------------
def build_model(hp):
    inputs = Input(shape=(1000, 1))
    x = layers.Conv1D(hp.Int("filters_initial", 32, 128, step=32), 7, padding='same', activation='relu')(inputs)

    for i in range(hp.Int("num_blocks", 1, 3)):
        filters = hp.Int(f"filters_block_{i}", 32, 128, step=32)
        x = residual_block(x, filters, 5)
        if hp.Boolean(f"pool_after_block_{i}", default=True):
            x = layers.MaxPooling1D(2)(x)

    if hp.Boolean("use_bilstm", default=True):
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp.Int("dense_units", 64, 256, step=64), activation='relu')(x)
    x = layers.Dropout(hp.Float("dropout", 0.3, 0.5, step=0.1))(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4])),
        loss=focal_loss_fixed,
        metrics=['accuracy']
    )
    return model

# -------------------------------------
# Helper: Train binary model
# -------------------------------------
def train_binary_model(df, class_1, class_2, name):
    print(f"\n📦 Training model: {name} [{class_1} vs {class_2}]")

    df_subset = df[df['diagnostic_superclass'].isin([class_1, class_2])]
    min_count = df_subset['diagnostic_superclass'].value_counts().min()
    df_balanced = pd.concat([
        resample(df_subset[df_subset['diagnostic_superclass'] == class_1], replace=False, n_samples=min_count),
        resample(df_subset[df_subset['diagnostic_superclass'] == class_2], replace=False, n_samples=min_count)
    ])

    tmp_csv_path = "tmp_balanced.csv"
    df_balanced.to_csv(tmp_csv_path, index=False)

    X, y = load_dataset_from_csv(
        csv_path=tmp_csv_path,
        base_path=BASE_PATH,
        augment=True,
        leads=LEADS
    )

    # Optional: Delete the temp file if you want
    os.remove(tmp_csv_path)


    perm = np.random.permutation(len(X))
    X, y = X[perm], np.array(y)[perm]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='hpo_logs',
        project_name=f'hpo_{name}'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    tuner.search(
        X, y_enc,
        validation_split=0.2,
        epochs=30,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    best_model = tuner.get_best_models(1)[0]
    best_model.fit(
        X, y_enc,
        validation_split=0.2,
        epochs=50,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    best_model.save(os.path.join(OUTPUT_DIR, f"{name}.keras"))
    print(f"✅ Saved model to {OUTPUT_DIR}/{name}.keras")

# -------------------------------------
# Main logic
# -------------------------------------
if __name__ == "__main__":
    df_all = pd.read_csv(CSV_PATH)
    train_binary_model(df_all, "MI", "NORM", "model_mi_norm")
    train_binary_model(df_all, "STTC", "NORM", "model_sttc_norm")
    train_binary_model(df_all, "MI", "STTC", "model_mi_sttc")
