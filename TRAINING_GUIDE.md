# Model Training Guide

This guide explains how to train the parking meter violation prediction model using the scripts in the `scripts/` directory.

## Overview

The training pipeline consists of:
1. **Data Loading**: Load preprocessed parking ticket data from `tickets_extracted.csv`
2. **Feature Engineering**: Extract temporal and spatial features
3. **Clustering**: Identify enforcement hotspots using DBSCAN
4. **Model Training**: Train a machine learning classifier
5. **Evaluation**: Assess model performance with multiple metrics
6. **Visualization**: Generate performance plots

## Quick Start

### Install Dependencies

```bash
pip install -r scripts/requirements.txt
```

### Train Model (scikit-learn version - recommended)

The scikit-learn version uses Random Forest and works reliably across all platforms:

```bash
python3 scripts/train_model_sklearn.py \
  --input_file scripts/tickets_extracted.csv \
  --output_dir ./models \
  --model_output api/model.pkl
```

### Train Model (LightGBM version - advanced)

For systems with proper LightGBM setup (requires libomp):

```bash
python3 scripts/train_model.py \
  --input_file scripts/tickets_extracted.csv \
  --output_dir ./models \
  --model_output api/model.pkl
```

## Command-Line Options

Both scripts support the following options:

- `--input_file` (default: `scripts/tickets_extracted.csv`)
  - Path to the preprocessed ticket data CSV file

- `--output_dir` (default: `./models`)
  - Directory to save model, results, and plots

- `--model_output` (default: None)
  - Optional path to save the trained model separately
  - Example: `api/model.pkl` to save to the API directory

- `--no_plot`
  - Skip visualization generation (faster training)

- `--no_cv`
  - Skip cross-validation (faster training)

## Output Files

After running the training script, you'll find:

```
models/
├── model.pkl                  # Trained model (pickle format)
├── results.json               # Complete evaluation metrics
├── confusion_matrix.png       # Confusion matrix heatmap
├── roc_curve.png              # ROC curve and AUC score
└── feature_importance.png     # Top 10 feature importances
```

If `--model_output api/model.pkl` is specified:
```
api/
└── model.pkl                  # Copy of trained model for API
```

## Example Usage Scenarios

### Scenario 1: Quick Training (No Visualizations or CV)

For fastest training when prototyping:

```bash
python3 scripts/train_model_sklearn.py --no_plot --no_cv
```

### Scenario 2: Full Training with Results in API Directory

For production model updates:

```bash
python3 scripts/train_model_sklearn.py \
  --output_dir ./models \
  --model_output api/model.pkl
```

### Scenario 3: Custom Data File

If using different preprocessed data:

```bash
python3 scripts/train_model_sklearn.py \
  --input_file data/custom_tickets.csv \
  --output_dir ./results
```

## Model Performance

Based on the current training:

**Test Set Metrics:**
- Accuracy: 99.9%
- Precision: 99.8%
- Recall: 99.9%
- F1 Score: 99.9%
- ROC AUC: 0.72

**Cross-Validation (5-fold):**
- Accuracy: 99.9% ± 0.03%
- Precision: 99.9% ± 0.05%
- Recall: 99.9% ± 0.03%
- F1 Score: 99.9% ± 0.04%

**Top Features (by importance):**
1. Latitude (27.8%)
2. Longitude (22.5%)
3. Date Ordinal (21.8%)
4. Day of Month (9.9%)
5. Weekday (9.5%)
6. Hour (8.5%)

## Data Requirements

The input CSV file must have these columns:
- `x` - Longitude coordinate
- `y` - Latitude coordinate
- `hour` - Hour of day (0-23)
- `minute` - Minute of hour (0-59)
- `day` - Day of month
- `month` - Month (1-12)
- `year` - Year

See `scripts/tickets_extracted.csv` for an example format.

## Troubleshooting

### LightGBM Import Error

If using `train_model.py` and getting LightGBM errors:
- Use `train_model_sklearn.py` instead (recommended)
- On macOS with Apple Silicon: Install libomp via Homebrew

### Class Imbalance Warning

The dataset has many more positive examples than negative examples. This is expected for parking enforcement data where most locations are enforced. The scripts handle this automatically through:
- Class balancing in the Random Forest classifier
- Proper evaluation metrics that account for imbalance

### Memory Issues

For very large datasets:
- Use `--no_plot` to reduce memory usage
- Reduce `test_size` parameter in the script (default 0.2)

## Integration with API

After training, the model is saved to `api/model.pkl`. The FastAPI server in `api/main.py` will load and use this model for predictions.

To serve predictions:

```bash
cd api
pip install -r requirements.txt
python main.py
```

Then make predictions via:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-06-25",
    "hour": 11,
    "latitude": 38.90720,
    "longitude": -77.03690
  }'
```

## Scripts Reference

### `train_model_sklearn.py` (Recommended)

**Algorithm**: Random Forest Classifier
**Advantages**:
- Works on all platforms
- No external dependencies beyond pip packages
- Good interpretability via feature importance

**Best for**: Production use, cross-platform deployment

### `train_model.py` (Advanced)

**Algorithm**: LightGBM Classifier
**Advantages**:
- Faster training on large datasets
- Often better performance
- Better handling of categorical features

**Requirements**: 
- macOS: requires libomp (`brew install libomp`)
- Linux/Windows: usually works out of the box

**Best for**: Experienced users, optimized performance

## Next Steps

1. Run `train_model_sklearn.py` to generate initial model
2. Check `models/results.json` for detailed metrics
3. Review plots in `models/` directory
4. Deploy model to API via `api/model.pkl`
5. Monitor prediction accuracy in production
6. Retrain periodically with new data

## Support

For issues or questions:
- Check logs in output
- Review `models/results.json` for detailed metrics
- Verify input data format matches `scripts/tickets_extracted.csv`
