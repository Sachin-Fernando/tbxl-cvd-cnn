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

# def build_model(n_classes):
#     model = keras.Sequential([
#         keras.Input(shape=(1000, 1)),
#         layers.Conv1D(16, kernel_size=5, activation='relu'),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),

#         layers.Conv1D(32, kernel_size=5, activation='relu'),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),

#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.5),

#         layers.Dense(n_classes, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model

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



def build_model(n_classes):
    inputs = keras.Input(shape=(1000, 1))

    x = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(inputs)
    x = residual_block(x, 32, 7)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, 64, 5)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, 128, 3)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

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

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
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

model_path = "models/lead1_model_current1.keras"

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
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
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
