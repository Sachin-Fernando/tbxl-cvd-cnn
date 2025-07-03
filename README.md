# ptbxl-cvd-cnn

A deep learning pipeline for detecting cardiovascular disease from ECG signals using the PTB-XL dataset.

## 🚀 Project Overview

This project:
- Builds a 1D Convolutional Neural Network (CNN) for ECG classification.
- Compares:
  - Full 12-lead ECG performance
  - Single-lead (Lead I) ECG for wearable applications
  - Single-lead with data augmentation
- Uses the PTB-XL dataset.
- Saves and reloads trained models for deployment.

## ✅ Dataset

This project uses the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/). **The dataset is NOT included in this repository.** Please download it manually and place it here:

```
ptbxl-data/
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/
```

## 📁 Repository Structure

```
ptbxl-cvd-cnn/
├── create_sample_ids.py
├── sample_ids.csv
├── requirements.txt
├── README.md
├── /src/
│ ├── extract.py
│ └── train_model.py
├── /models/
│ ├── cnn_model_12lead.h5
│ ├── cnn_model_leadI.h5
│ ├── cnn_model_leadI_aug.h5
│ └── label_encoder.pkl
├── /plots/
  ├── confusion_matrix_exp1.png
  ├── confusion_matrix_exp2.png
  └── ...
```

## 🛠️ How to Run

1. Download PTB-XL dataset and place it in `ptbxl-data/`.
2. Run:
    ```
    python create_sample_ids.py
    ```
3. Train your model:
    ```
    python src/train_model.py
    ```

## 📜 License

MIT License.