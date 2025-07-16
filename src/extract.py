import pandas as pd
import numpy as np
import wfdb
import os

def load_ecg_signal(path, lead=0, length=1000):
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, lead]
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)))

def add_noise(signal, noise_level=0.05):
    return signal + noise_level * np.random.randn(*signal.shape)

def add_baseline_drift(signal, max_drift=0.1):
    drift = np.linspace(
        0,
        np.random.uniform(-max_drift, max_drift),
        signal.shape[0]
    )
    return signal + drift[:, np.newaxis]

def add_powerline_noise(signal, freq=50, amplitude=0.02):
    t = np.linspace(0, 1, signal.shape[0])
    noise = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + noise[:, np.newaxis]

def random_signal_dropout(signal, max_fraction=0.1):
    mask = np.ones(signal.shape[0])
    start = np.random.randint(0, int(signal.shape[0] * (1 - max_fraction)))
    length = np.random.randint(1, int(signal.shape[0] * max_fraction))
    mask[start:start+length] = 0
    return signal * mask[:, np.newaxis]

def random_scaling(signal, min_factor=0.9, max_factor=1.1):
    scale = np.random.uniform(min_factor, max_factor)
    return signal * scale

def load_dataset_from_csv(
    csv_path="sample_ids.csv",
    base_path="../ptbxl-data/records100/",
    augment=False,
    leads=[0]
):
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        path = os.path.join(base_path, row['filename_lr'])
        record = wfdb.rdrecord(path)

        if isinstance(leads, int):
            leads = [leads]

        if len(leads) == 1:
            signal = record.p_signal[:, leads[0]]
            signal = signal[:, np.newaxis]
        else:
            signal = record.p_signal[:, leads]

        signal = signal[:1000, :]
        if signal.shape[0] < 1000:
            signal = np.pad(signal, ((0, 1000 - signal.shape[0]), (0, 0)))

        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True) + 1e-8
        signal = (signal - mean) / std

        if augment:
            if np.random.rand() < 0.7:
                signal = add_noise(signal)
            if np.random.rand() < 0.5:
                signal = add_baseline_drift(signal)
            if np.random.rand() < 0.5:
                signal = add_powerline_noise(signal)
            if np.random.rand() < 0.3:
                signal = random_signal_dropout(signal)
            if np.random.rand() < 0.5:
                signal = random_scaling(signal)

        X.append(signal)
        y.append(row['diagnostic_superclass'])

    X = np.array(X)
    y = np.array(y)
    return X, y


# X, y = load_dataset_from_csv(
#     csv_path="batches/sample_ids_batch1.csv",
#     base_path="../ptbxl-data/",
#     augment=True,
#     leads=[0]
# )

# print("✅ Data loaded.")
# print("X shape:", X.shape)
# print("Single signal shape:", X[0].shape)
