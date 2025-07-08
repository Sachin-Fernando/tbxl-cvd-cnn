import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.extract import load_dataset_from_csv

# ----------------------------
# Load one batch
# ----------------------------

X, y = load_dataset_from_csv(
    csv_path="batches/sample_ids_batch1.csv",
    base_path="../ptbxl-data/records100/",
    augment=True,
    lead=0
)

print("✅ Data loaded.")
print("Shape of X:", X.shape)
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
    layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=(1000, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')
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
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# ----------------------------
# Save model
# ----------------------------

model.save("models/lead1_model_v1.h5")
print("✅ Model saved!")
