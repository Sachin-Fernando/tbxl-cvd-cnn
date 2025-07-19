import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from extract import load_dataset_from_csv
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# ----------------------------
# Load one batch
# ----------------------------

X, y = load_dataset_from_csv(
    csv_path="batches/sample_ids_batch1.csv",
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

print("✅ Data loaded.")
print("X shape:", X.shape)
print("Single signal shape:", X[0].shape)
print("First 5 labels:", y[:5])

# ----------------------------
# Encode labels as integers
# ----------------------------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Define simple CNN model
# ----------------------------

model = keras.Sequential([
    keras.Input(shape=(1000, 1)),
    layers.Conv1D(16, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(3, activation='softmax')
])



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ----------------------------
# Train model
# ----------------------------

model.fit(
    X,
    y_encoded,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)


# ----------------------------
# Save model
# ----------------------------

model.save("models/lead1_model_v1.keras")

print("✅ Model saved!")
