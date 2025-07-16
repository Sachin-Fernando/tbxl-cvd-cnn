import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from extract import load_dataset_from_csv
import keras_tuner as kt
import os

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
# STEP 1 - Merge all CSVs
# ----------------------------

batch_files = sorted(glob.glob("batches/sample_ids_batch*.csv"))
print("✅ Found batch files:", batch_files)

dfs = [pd.read_csv(f) for f in batch_files]
df_all = pd.concat(dfs, ignore_index=True)

# Save merged CSV
df_all.to_csv("all_batches.csv", index=False)
print("✅ Merged all batches into all_batches.csv")

# ----------------------------
# STEP 2 - Load entire dataset
# ----------------------------

X, y = load_dataset_from_csv(
    csv_path="all_batches.csv",
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

# ----------------------------
# STEP 3 - Shuffle dataset
# ----------------------------

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

print("✅ Data shuffled.")
print("X shape:", X.shape)

# ----------------------------
# STEP 4 - Encode labels
# ----------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)

print("✅ Classes:", list(le.classes_))

# ----------------------------
# STEP 5 - Compute class weights
# ----------------------------

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_encoded
)
class_weights_dict = dict(zip(classes, weights))
print("✅ Class weights:", class_weights_dict)

# ----------------------------
# STEP 6 - Define Residual Block
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
# STEP 7 - Define Model Builder
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
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------------
# STEP 8 - Run Hyperparameter Tuning
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
    X,
    y_encoded,
    epochs=30,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]

# ----------------------------
# STEP 9 - Retrain Best Model
# ----------------------------

history = best_model.fit(
    X,
    y_encoded,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ----------------------------
# STEP 10 - Save Model
# ----------------------------

os.makedirs("models", exist_ok=True)
best_model.save("models/lead1_model_full_hpo.keras")
print("✅ Final model saved as models/lead1_model_full_hpo.keras")

# ----------------------------
# STEP 11 - Evaluate on Test Set
# ----------------------------

X_test, y_test = load_dataset_from_csv(
    csv_path="sample_ids_test.csv",
    base_path="../ptbxl-data/",
    augment=False,
    leads=[0]
)

y_test_encoded = le.transform(y_test)

results = best_model.evaluate(X_test, y_test_encoded, verbose=1)
print("✅ Test Loss:", results[0])
print("✅ Test Accuracy:", results[1])
