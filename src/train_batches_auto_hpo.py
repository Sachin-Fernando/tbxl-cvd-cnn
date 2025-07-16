import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from extract import load_dataset_from_csv
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# Early stopping & LR schedule
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
# Define Model for HPO
# ----------------------------

def build_model(hp, n_classes):
    inputs = keras.Input(shape=(1000, 1))

    # Tune initial Conv1D filters
    filters_initial = hp.Int("filters_initial", min_value=16, max_value=64, step=16)
    x = layers.Conv1D(filters_initial, kernel_size=7, activation='relu', padding='same')(inputs)

    # Tune number of residual blocks
    num_blocks = hp.Int("num_blocks", 1, 3)
    filters = filters_initial
    for i in range(num_blocks):
        filters = hp.Int(f"filters_block_{i}", min_value=32, max_value=128, step=32)
        x = residual_block(x, filters, kernel_size=5)

        if hp.Boolean(f"pool_after_block_{i}"):
            x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)

    dense_units = hp.Int("dense_units", min_value=32, max_value=256, step=32)
    x = layers.Dense(dense_units, activation='relu')(x)

    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------------
# Define Residual Block
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
# Load or Initialize Model Path
# ----------------------------

model_path = "models/lead1_model_current1.keras"

# ----------------------------
# Training Loop
# ----------------------------

for csv_path in batch_files:
    print("\n🚀 Training on batch:", csv_path)

    try:
        X, y = load_dataset_from_csv(
            csv_path=csv_path,
            base_path="../ptbxl-data/",
            augment=True,
            leads=[0]
        )

        # Shuffle batch
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Encode labels
        y_encoded = le.transform(y)

        # Compute class weights
        class_weights_dict = None
        if len(np.unique(y_encoded)) > 1:
            classes = np.unique(y_encoded)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_encoded
            )
            class_weights_dict = dict(zip(classes, weights))

        # Run hyperparameter tuning
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, n_classes=len(le.classes_)),
            objective="val_accuracy",
            max_trials=5,
            executions_per_trial=1,
            overwrite=True,
            directory="hpo_logs",
            project_name=f"hpo_batch_{csv_path.replace('/', '_')}"
        )

        tuner.search(
            X,
            y_encoded,
            epochs=20,
            validation_split=0.2,
            class_weight=class_weights_dict,
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        # Retrain best model on this batch
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

        last_acc = history.history['accuracy'][-1]
        last_val_acc = history.history['val_accuracy'][-1]
        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]

        print("\n✅ Training Summary for this batch:")
        print(f" - Train accuracy:      {last_acc:.4f}")
        print(f" - Validation accuracy: {last_val_acc:.4f}")
        print(f" - Train loss:          {last_loss:.4f}")
        print(f" - Validation loss:     {last_val_loss:.4f}")

        # Always save best model after each batch
        print(f"✅ Saving model after batch {csv_path}")
        best_model.save(model_path)

    except Exception as e:
        print(f"❌ Error training batch {csv_path}: {e}")
        print("⚠️ Skipping to next batch.")
        continue

print("✅ All batches processed. Training complete.")
