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
    return signal + noise_level * np.random.randn(*signal.shape)


def load_dataset_from_csv(
    csv_path="sample_ids.csv",
    base_path="../ptbxl-data/records100/",
    augment=False,
    leads=[0]
):
    import pandas as pd
    import numpy as np
    import wfdb
    import os

    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        path = os.path.join(base_path, row['filename_lr'])
        record = wfdb.rdrecord(path)

        # FIX: guarantee leads is a list
        if isinstance(leads, int):
            leads = [leads]

        if len(leads) == 1:
            # extract single lead
            signal = record.p_signal[:, leads[0]]
            # force shape (samples, 1)
            signal = signal[:, np.newaxis]
        else:
            # extract multiple leads
            signal = record.p_signal[:, leads]

        # Clip or pad
        signal = signal[:1000, :]
        if signal.shape[0] < 1000:
            signal = np.pad(signal, ((0, 1000 - signal.shape[0]), (0, 0)))

        #standardize signal
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True) + 1e-8
        signal = (signal - mean) / std


        if augment:
            signal = add_noise(signal)

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
