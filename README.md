# PTB-XL Lead-I ECG Classification Service

[![CI](https://github.com/Sachin-Fernando/tbxl-cvd-cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/Sachin-Fernando/tbxl-cvd-cnn/actions/workflows/ci.yml)

A research-to-production portfolio project for classifying Lead-I ECG recordings with a
one-dimensional residual CNN. The repository contains the original PTB-XL experimentation,
trained Keras artifacts, evaluation outputs, and a containerized FastAPI inference service.

> **Important:** This project is a research demonstration. It is not a medical device and must
> not be used for diagnosis, treatment, or other clinical decisions.

## What this project demonstrates

- PTB-XL data extraction and per-record preprocessing
- Three-class classification: myocardial infarction (`MI`), normal ECG (`NORM`), and ST/T
  changes (`STTC`)
- Lead-I modeling for a wearable-oriented use case
- Residual 1D CNN experiments, optional BiLSTM layers, class weighting, focal loss, data
  augmentation, and Keras Tuner hyperparameter search
- Versioned model artifacts and evaluation reports
- A typed REST API with health/readiness checks and input validation
- Unit and API tests that do not require the full model to be loaded
- A non-root Docker image, Docker Compose configuration, and GitHub Actions CI

## Architecture

```text
1,000 Lead-I samples
        |
        v
per-record z-score normalization
        |
        v
FastAPI -> Keras model -> MI / NORM / STTC probabilities
```

The production contract is recorded in [`model_metadata.json`](model_metadata.json). The API
uses `models/lead1_model_full_hpo_V3.keras` because it has the highest recorded accuracy among
the evaluated artifacts currently in this repository.

## Recorded results

| Model | Recorded accuracy |
| --- | ---: |
| `lead1_model_full_hpo_V2.keras` | 70.28% |
| `lead1_model_full_hpo_V3.keras` | **70.54%** |
| `lead1_model_full_hpo_V4.keras` | 69.89% |
| `lead1_model_full_hpo_V5.keras` | 70.00% |
| `lead1_model_full_hpo.keras` | 69.15% |

These values are read from the committed files in `evaluation_outputs/`. The existing
evaluation script enables stochastic augmentation on the test set, so treat them as recorded
experiment results rather than a fully deterministic benchmark. Confusion matrices and
per-class reports are available in the same directory.

## Repository layout

```text
.
├── app/                    # Production API and model service
├── tests/                  # Unit and endpoint tests
├── src/                    # Current training, extraction, and evaluation code
│   └── legacy/             # Preserved earlier experiments
├── notebooks/              # Exploratory analysis
├── models/                 # Versioned trained Keras models
├── evaluation_outputs/     # Accuracy, class reports, and confusion matrices
├── hpo_logs/               # Hyperparameter-search artifacts
├── data/ and batches/      # Dataset indexes (not raw ECG records)
├── model_metadata.json     # Deployed model contract and provenance
├── Dockerfile
└── .github/workflows/ci.yml
```

The raw PTB-XL waveform dataset is intentionally not committed.

## Run the inference API locally

Python 3.11 is the tested runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive OpenAPI UI.

The service exposes:

- `GET /health` — process liveness
- `GET /ready` — model readiness
- `POST /predict` — validated inference for exactly 1,000 Lead-I samples

Example request:

```bash
python - <<'PY'
import json
import urllib.request

payload = json.dumps({"ecg_signal": [0.0] * 1000}).encode()
request = urllib.request.Request(
    "http://localhost:8000/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(request).read().decode())
PY
```

Example response:

```json
{
  "prediction": "NORM",
  "confidence": 0.91,
  "probabilities": {"MI": 0.04, "NORM": 0.91, "STTC": 0.05},
  "model_version": "V3",
  "disclaimer": "Research demonstration only; not a medical device or clinical diagnosis."
}
```

## Run with Docker

```bash
docker compose up --build
```

The container runs as a non-root user, includes a readiness health check, and copies only the
selected production model into the image.

## Test and lint

The lightweight CI dependency set deliberately mocks the neural network so routine checks do
not download TensorFlow:

```bash
python -m pip install -r requirements-ci.txt
ruff check app tests
pytest
```

GitHub Actions runs these checks on pushes and pull requests, then verifies that the production
Docker image builds successfully.

## Reproduce training and evaluation

1. Download PTB-XL 1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2. Extract it as a sibling of this repository so the existing relative paths resolve:

   ```text
   parent-directory/
   ├── ptbxl-cvd-cnn/
   └── ptbxl-data/
       ├── ptbxl_database.csv
       ├── scp_statements.csv
       └── records100/
   ```

3. Create the train/test indexes if needed:

   ```bash
   python src/generate_train_test_csv.py
   ```

4. Install the complete experiment dependencies and train:

   ```bash
   python -m pip install -r requirements.txt
   python src/train_model.py
   ```

5. Evaluate the saved models:

   ```bash
   python src/evaluate_all_models.py
   ```

Azure mode in `src/train_model.py` additionally expects `AZURE_STORAGE_CONNECTION_STRING`.
Never commit that value; supply it through the environment or a secret manager.

## Configuration

`MODEL_PATH` can override the deployed artifact. See [`.env.example`](.env.example). Any
replacement model must preserve the input shape and class ordering defined in
`model_metadata.json`, or that metadata must be versioned with the model.

## Known limitations

- The model predicts only three PTB-XL diagnostic superclasses.
- Training/evaluation paths are currently script-oriented rather than a single orchestrated
  pipeline.
- The committed metrics do not include calibration, subgroup, or external-validation studies.
- The API accepts already sampled numeric Lead-I data; waveform file parsing is outside the
  production endpoint.
- Clinical validation, regulatory review, security hardening, and production monitoring are
  outside this portfolio scope.

## Data and attribution

This project uses the PTB-XL dataset. Follow the dataset's license, access, and citation
requirements on PhysioNet. No raw PTB-XL waveform files are stored in this repository.
