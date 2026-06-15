# FastAPI Parking Ticket Prediction App

This FastAPI app serves a machine learning model (`api/model.pkl`) for predicting expired meter parking tickets in Washington, DC.

Notes (latest updates):
- The project now saves per-run model bundles under `scripts/models/` as `/{modelname}-YYYY-MM-DD/` containing the `.pkl`, `results.json`, and plots.
- An evaluation script (`scripts/evaluate_models.py`) selects the best recent model and copies it to `api/model.pkl` for serving. The most recent retrain occurred on 2026-06-14 and `api/model.pkl` was updated accordingly.

Retrain & update workflow (quick):
- Retrain locally: `python scripts/train_model.py --input_file scripts/tickets_extracted.csv --output_dir scripts/models`
- Compare and promote best model: `python scripts/evaluate_models.py` (this copies the chosen `.pkl` to `api/model.pkl` and writes a comparison bundle).

For more details on training and evaluation, see `scripts/train_model.py`, `scripts/train_model_sklearn.py`, and `scripts/evaluate_models.py`.

## Deployment

The app is deployed to **Cloud Run** using **Cloud Build** with a trigger connected to this repository. Changes pushed to the repository automatically trigger a build and deployment to Cloud Run.

### Model Training and Upload

### Serving the Model

Locally and in CI, the FastAPI app loads `api/model.pkl` from the repository. Use the evaluation script to update `api/model.pkl` after retraining so the service serves the desired model.

For historical context the original training notebook is available [here](https://nbviewer.org/github/reedmarkham/meter-made/blob/main/meter-made.ipynb).

## Ingress / Access

This service is deployed to Cloud Run with an "internal" ingress policy. That restricts incoming traffic to other resources inside the same Google Cloud project (for example, other Cloud Run services like the app in the [`ui`](./ui) subdirectory ) and disallows public internet access.