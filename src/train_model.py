import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from azure.storage.blob import BlobServiceClient
from extract import load_dataset_from_csv
from focal_loss import focal_loss_fixed
from io import StringIO

# -------------------------------------
# Ask user for mode
# -------------------------------------
mode = input("⚙️  Run mode? Enter 'local' or 'azure': ").strip().lower()
assert mode in ['local', 'azure'], "❌ Invalid mode. Use 'local' or 'azure'"

# -------------------------------------
# Load CSV depending on mode
# -------------------------------------
if mode == "azure":
    try:
        print("🔄 Connecting to Azure Blob Storage...")
        AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        CONTAINER_NAME = "ptbxl-data"
        BLOB_NAME = "all_batches.csv"

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(BLOB_NAME)

        csv_data = blob_client.download_blob().readall().decode('utf-8')
        df_all = pd.read_csv(StringIO(csv_data))
        local_csv_path = "temp_all_batches.csv"
        df_all.to_csv(local_csv_path, index=False)
        print(f"✅ Loaded and saved {BLOB_NAME} to {local_csv_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load CSV from Blob: {e}")

else:
    try:
        local_csv_path = "data/all_batches.csv"
        df_all = pd.read_csv(local_csv_path)
        print(f"✅ Loaded CSV locally from: {local_csv_path}")
    except Exception as e:
        raise FileNotFoundError(f"❌ Local CSV not found at {local_csv_path}: {e}")

# -------------------------------------
# Load ECG data
# -------------------------------------
print("📥 Loading ECG samples from CSV references...")
X, y = load_dataset_from_csv(
    csv_path="data/all_batches.csv",
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

perm = np.random.permutation(len(X))
X, y = X[perm], np.array(y)[perm]

# -------------------------------------
# Encode labels & compute weights
# -------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y_encoded)

weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
class_weights_dict = dict(zip(classes, weights))

# -------------------------------------
# Residual block
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
# Build model
# -------------------------------------
def build_model(hp, n_classes):
    inputs = keras.Input(shape=(1000, 1))
    x = layers.Conv1D(hp.Int("filters_initial", 32, 128, step=32), kernel_size=7, padding='same', activation='relu')(inputs)

    for i in range(hp.Int("num_blocks", 1, 3)):
        filters = hp.Int(f"filters_block_{i}", 32, 128, step=32)
        x = residual_block(x, filters, kernel_size=5)
        if hp.Boolean(f"pool_after_block_{i}", default=True):
            x = layers.MaxPooling1D(2)(x)

    if hp.Boolean("use_bilstm", default=True):
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp.Int("dense_units", 64, 256, step=64), activation='relu')(x)
    x = layers.Dropout(hp.Float("dropout", 0.3, 0.5, step=0.1))(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4])),
        loss=focal_loss_fixed,
        metrics=['accuracy']
    )
    return model

# -------------------------------------
# Callbacks
# -------------------------------------
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# -------------------------------------
# Hyperparameter tuning
# -------------------------------------
print("🔍 Starting hyperparameter tuning...")
tuner = kt.RandomSearch(
    lambda hp: build_model(hp, len(classes)),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='hpo_logs',
    project_name='ecg_refined'
)

tuner.search(
    X, y_encoded,
    validation_split=0.2,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# -------------------------------------
# Retrain best model
# -------------------------------------
best_model = tuner.get_best_models(1)[0]
print("📈 Retraining best model on full dataset...")
history = best_model.fit(
    X, y_encoded,
    validation_split=0.2,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# -------------------------------------
# Save model
# -------------------------------------
os.makedirs("outputs", exist_ok=True)
best_model.save("outputs/fixed_ecg_model.keras")
print("✅ Saved trained model to outputs/fixed_ecg_model.keras")
