import pandas as pd
import numpy as np
import wfdb
import os

def load_ecg_signal(path, lead=0, length=1000):
    """
    Load a single ECG signal file and extract one lead.
    Pads or truncates the signal to the desired length.
    """
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, lead]
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)))

def add_noise(signal, noise_level=0.05):
    """
    Add Gaussian noise to the signal for augmentation.
    """
    return signal + noise_level * np.random.randn(len(signal))

def load_dataset_from_csv(
    csv_path="sample_ids.csv",
    base_path="../ptbxl-data/records100/",
    augment=False,
    leads=[0]
):
    """
    Load multiple ECG signals and labels based on a CSV file.
    """
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        path = os.path.join(base_path, row['filename_lr'])
        record = wfdb.rdrecord(path)

        # Load multiple leads
        if len(leads) > 1:
            signal = record.p_signal[:, leads]
        else:
            signal = record.p_signal[:, leads[0]]
            if len(signal.shape) == 1:
                signal = signal[:, np.newaxis]

        # Clip or pad signal
        signal = signal[:1000, :]
        if augment:
            signal = add_noise(signal)

        X.append(signal)
        y.append(row['diagnostic_superclass'])

    X = np.array(X)
    y = np.array(y)
    return X, y
