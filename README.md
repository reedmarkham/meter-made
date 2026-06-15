# meter-made

https://meter-made-ui-397485407730.us-east4.run.app/

See `scripts/` for training and evaluation scripts as well as model-training artifacts, and see `api/README.md` for more documentation specific to the API.

## 2026-06-14:
- Models are saved per-run under `scripts/models/{modelname}-YYYY-MM-DD/` and include the `.pkl`, `results.json`, and plots.
- Run `python scripts/train_model.py --input_file scripts/tickets_extracted.csv --output_dir scripts/models` to retrain LightGBM.
- Compare recent runs and promote the chosen model to the API with `python scripts/evaluate_models.py` (this copies the selected `.pkl` to `api/model.pkl`).
