import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from extract import load_dataset_from_csv

# ----------------------------
# Model Definition
# ----------------------------

def build_model(n_classes):
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

        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ----------------------------
# Early stopping callback
# ----------------------------

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ----------------------------
# Prepare batch list
# ----------------------------

batch_files = sorted(glob.glob("batches/sample_ids_batch*.csv"))
print("✅ Found batches:")
for f in batch_files:
    print(f" - {f}")

# ----------------------------
# Prepare Label Encoder
# ----------------------------

# Load one batch to discover all classes
X_sample, y_sample = load_dataset_from_csv(
    csv_path=batch_files[0],
    base_path="../ptbxl-data/",
    augment=True,
    leads=[0]
)

le = LabelEncoder()
le.fit(y_sample)

# ----------------------------
# Load or Create Model
# ----------------------------

model_path = "models/lead1_model_current.keras"

try:
    print(f"✅ Loading existing model from {model_path}")
    model = keras.models.load_model(model_path)
except:
    print("⚠️ No saved model found. Building new model.")
    model = build_model(len(le.classes_))

# ----------------------------
# Training Loop
# ----------------------------

for csv_path in batch_files:
    print("\n🚀 Training on batch:", csv_path)

    X, y = load_dataset_from_csv(
        csv_path=csv_path,
        base_path="../ptbxl-data/",
        augment=True,
        leads=[0]
    )

    # Encode labels
    y_encoded = le.transform(y)

    # Train
    history = model.fit(
        X,
        y_encoded,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Show metrics
    last_acc = history.history['accuracy'][-1]
    last_val_acc = history.history['val_accuracy'][-1]
    last_loss = history.history['loss'][-1]
    last_val_loss = history.history['val_loss'][-1]

    print("\n✅ Training Summary for this batch:")
    print(f" - Train accuracy:      {last_acc:.4f}")
    print(f" - Validation accuracy: {last_val_acc:.4f}")
    print(f" - Train loss:          {last_loss:.4f}")
    print(f" - Validation loss:     {last_val_loss:.4f}")

    # Ask user
    answer = input("Continue training on next batch? (y/n): ").strip().lower()

    if answer == 'y':
        print(f"✅ Saving model after batch {csv_path}")
        model.save(model_path)
    else:
        print(f"🚫 Stopping. Last accepted model is saved at {model_path}.")
        break

print("✅ Training completed.")
