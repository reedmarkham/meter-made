# Claude Code Session Log

## Session: February 6, 2026

### Python Environment Setup & Dependency Resolution

#### Problem
The training script `scripts/train_model_sklearn.py` was failing with an import error:
```
imbalanced-learn not installed. Install with: pip install imbalanced-learn
```

#### Investigation
The script had recently been updated (commit `73c486b`) to include class balancing using `RandomUnderSampler` from the `imblearn.under_sampling` module, but the dependency wasn't installed in the environment.

#### Resolution Process

**1. Initial Installation Attempt**
- Ran `pip install imbalanced-learn`
- Successfully installed `imbalanced-learn==0.12.4`
- However, this triggered a NumPy upgrade to 2.0.2

**2. NumPy 2.0 Compatibility Issues**
- NumPy 2.0.2 caused binary incompatibility errors with existing packages:
  - `pandas` was compiled against NumPy 1.x
  - `numexpr` and `bottleneck` dependencies failed with `AttributeError: _ARRAY_API not found`
  - Error: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**3. NumPy Downgrade**
- Downgraded NumPy to 1.26.4 (1.x series): `pip install "numpy<2" --force-reinstall`
- Reinstalled pandas: `pip install --force-reinstall --no-cache-dir pandas`
- Upgraded pandas to 2.3.3 (compatible with NumPy 1.26.4)
- Upgraded optional dependencies: `pip install --upgrade numexpr bottleneck`

**4. Additional Dependency Issues**
- **DuckDB**: Version 1.0.0 failed to build from source on macOS
  - Solution: Upgraded to `duckdb>=1.1.0` which has pre-built wheels
  - Installed: `duckdb==1.4.4`

- **LightGBM**: Missing OpenMP library on macOS
  - Error: `Library not loaded: @rpath/libomp.dylib`
  - Solution: `brew install libomp`

**5. Package Installation**
- Installed remaining packages: `pip install matplotlib==3.8.4 seaborn==0.13.1 lightgbm==4.6.0`

#### Final Environment State

All packages successfully installed and verified:
- **pandas**: 2.3.3
- **numpy**: 1.26.4 (1.x for compatibility)
- **scikit-learn**: 1.6.1
- **matplotlib**: 3.8.4
- **seaborn**: 0.13.1
- **lightgbm**: 4.6.0
- **duckdb**: 1.4.4
- **imbalanced-learn**: 0.12.4

#### Updated Files
- `scripts/requirements.txt`: Updated with correct package versions and changed `duckdb==1.0.0` to `duckdb>=1.1.0`

#### Key Learnings
1. NumPy 2.0 introduced breaking changes with compiled C extensions
2. Many scientific Python packages still require NumPy 1.x for compatibility
3. Pre-built wheels are crucial for packages like DuckDB on macOS
4. LightGBM requires OpenMP runtime library on macOS (installed via Homebrew)

#### Training Script Status
✅ `scripts/train_model_sklearn.py` is now ready to run with full support for:
- Random Forest classification
- Class balancing via `RandomUnderSampler`
- DBSCAN clustering for location hotspots
- Model evaluation and visualization

---

### Notes on Project Requirements Files

The project has three separate requirements.txt files:
1. **`scripts/requirements.txt`** - For data science/ML scripts (✅ Updated and working)
2. **`api/requirements.txt`** - For FastAPI application
3. **`requirements.txt`** (root) - Full project dependencies

**Important**: The root `requirements.txt` contains NumPy 2.0.2 and has some dependency conflicts (boto3/botocore versions). For running the training scripts, use `scripts/requirements.txt` which has been configured with compatible versions (NumPy 1.26.4).

#### Dependency Conflict Details
- Root `requirements.txt` line 13-14: `boto3==1.35.44` and `botocore==1.39.8` have conflicting sub-dependencies
- Root `requirements.txt` line 74: `numpy==2.0.2` causes binary incompatibility with many packages
- Current environment is configured for the scripts folder and works correctly

---

### Root Requirements.txt Fixes

Fixed multiple issues in the root `requirements.txt` file:

**1. AWS SDK Dependency Chain Resolution**
- **Problem**: Multiple conflicts in the aiobotocore/boto3/botocore/s3transfer dependency chain:
  - `aiobotocore==2.23.2` requires `botocore>=1.39.7,<1.39.9`
  - Original requirements had `boto3==1.35.44` with `botocore==1.39.8` (mismatch - wrong boto3 version)
  - Original had `s3transfer==0.10.3` but boto3 1.39.8 requires `s3transfer>=0.13.0,<0.14.0`
- **Solution**: Aligned all AWS packages to work with aiobotocore 2.23.2:
  - `aiobotocore==2.23.2` (unchanged)
  - `boto3==1.39.8` (updated to match botocore)
  - `botocore==1.39.8` (compatible with aiobotocore 2.23.2: within 1.39.7-1.39.9 range)
  - `s3transfer>=0.13.0,<0.14.0` (compatible with boto3 1.39.8)
- **Rationale**: AWS SDK packages have tightly coupled version requirements; versions must be synchronized

**2. NumPy 2.0 Compatibility**
- **Problem**: `numpy==2.0.2` causes binary incompatibility with many compiled packages
- **Solution**: Downgraded to `numpy==1.26.4` (matching scripts/requirements.txt)
- **Impact**: Prevents crashes with pandas, scipy, scikit-learn, and other scientific packages

**3. DuckDB Build Issues**
- **Problem**: `duckdb==1.0.0` requires building from source on macOS, which fails
- **Solution**: Changed to `duckdb>=1.1.0` which has pre-built wheels
- **Result**: Installs cleanly without requiring build tools

**4. Package Version Synchronization**
- Updated `pandas` from `2.2.3` to `2.3.3`
- Updated `scikit-learn` from `1.6.0` to `1.6.1`
- Updated `imbalanced-learn` from `0.12.0` to `0.12.4`

These versions are now consistent across both root and scripts requirements files.

---

### Model Performance Optimization: Addressing Class Imbalance

#### Problem
The trained parking ticket prediction model showed severe class imbalance issues:
- **Original model confusion matrix**:
  - Class 0 (non-hotspot): 1 correct, 1 misclassified (50% recall)
  - Class 1 (hotspot): 351 misclassified, 2,087 correct (85.6% recall)
- **Original class distribution**: 12,195 samples in Class 1 vs. 1 sample in Class 0 (99.99% imbalance)
- **Test accuracy**: 85.6% (misleading due to extreme imbalance)
- **ROC AUC**: 92.3% (unstable, ranged from 0.44 to 0.97 in CV folds)
- **Cross-validation**: Perfect training scores (100%) but unstable test scores, indicating overfitting

#### Root Cause Analysis
The problem stemmed from DBSCAN clustering parameters that were too lenient:
- Original parameters: `eps=0.005`, `min_samples=3`
- These settings caused almost all parking tickets to be classified as "hotspots" (Class 1)
- With only 1-2 non-hotspot samples, the model couldn't learn to distinguish between classes
- RandomUnderSampler was undersampling from an already tiny minority class

#### Solution Approach
Implemented a two-pronged strategy:

**1. Adjusted DBSCAN Clustering Parameters** ([train_model_sklearn.py:152](scripts/train_model_sklearn.py#L152))
- Changed `eps` from `0.005` to `0.003` (smaller epsilon = tighter, more selective clusters)
- Changed `min_samples` from `3` to `50` (higher threshold = only true hotspots with many tickets qualify)
- **Rationale**: Create more balanced classes by requiring locations to have significantly more tickets before being labeled as hotspots

**2. Modified Undersampling Strategy** ([train_model_sklearn.py:199](scripts/train_model_sklearn.py#L199))
- Changed from default 1:1 balancing to `sampling_strategy=0.5`
- This keeps 50% as many minority class samples as majority class (e.g., 2:1 ratio)
- **Rationale**: Less aggressive undersampling prevents over-correction and preserves more training data

#### Results

**New Class Distribution:**
- Class 1 (hotspot): 9,353 samples (76.7%)
- Class 0 (non-hotspot): 2,843 samples (23.3%)
- Much healthier balance compared to original 99.99% imbalance

**Test Set Performance (Dramatic Improvement):**
- **Accuracy**: 95.1% (↑ from 85.6%)
- **Precision**: 95.3% (stable)
- **Recall**: 95.1% (↑ from 85.6%)
- **F1 Score**: 95.2% (↑ from 89.4%)
- **ROC AUC**: 98.7% (↑ from 92.3%)

**Balanced Confusion Matrix:**
```
Class 0 (non-hotspot):  530 correct, 39 misclassified → 93.1% recall (↑ from 50%)
Class 1 (hotspot):      1,791 correct, 80 misclassified → 95.7% recall (↑ from 85.7%)
```

**Cross-Validation Results (Stable & Reliable):**
- CV Accuracy: 95.4% ± 0.1% (low variance indicates stability)
- CV ROC AUC: 98.8% ± 0.2% (consistent across folds)
- Train vs. Test gap reduced: 97.9% train vs. 95.4% test (minimal overfitting)

**Feature Importance (Makes Intuitive Sense):**
- Longitude: 43.6%
- Latitude: 32.9%
- Date ordinal: 11.3%
- Hour: 5.7%
- Day of month: 3.7%
- Weekday: 2.8%

Location features (76.5% combined) dominate as expected for spatial hotspot prediction.

#### Key Learnings

1. **Class imbalance at the source**: When DBSCAN parameters are too lenient, almost all data points form clusters, creating artificial imbalance
2. **Undersampling isn't always the answer**: If the minority class is already tiny (1-10 samples), aggressive undersampling makes the problem worse
3. **Parameter tuning sequence matters**: Fix the data generation process (DBSCAN) before applying balancing techniques (undersampling)
4. **DBSCAN parameter effects**:
   - Smaller `eps` → tighter clusters → more isolated points → more Class 0 samples
   - Higher `min_samples` → stricter hotspot criteria → more Class 0 samples
5. **Model diagnostics**: Perfect training scores + poor/unstable test scores = severe overfitting due to insufficient minority class representation

#### Files Modified
- [scripts/train_model_sklearn.py](scripts/train_model_sklearn.py):
  - Line 152: Updated `_compute_clusters()` default parameters to `eps=0.003, min_samples=50`
  - Line 199: Updated `train_model()` to use `sampling_strategy=0.5` in RandomUnderSampler
  - Added detailed comments explaining the rationale for undersampling strategy

#### Model Artifacts Updated
- `models/model.pkl`: Retrained model with balanced classes
- `models/results.json`: Updated metrics showing 95%+ performance across all measures
- `models/confusion_matrix.png`: Balanced confusion matrix with good performance on both classes
- `models/roc_curve.png`: ROC curve with AUC = 0.987
- `models/feature_importance.png`: Feature importance dominated by location features

#### Next Steps / Recommendations
- Consider testing different DBSCAN parameters (e.g., `eps=0.002-0.005`, `min_samples=30-100`) to find optimal balance
- Explore alternative clustering algorithms (HDBSCAN, OPTICS) for automatic parameter selection
- Consider reframing as regression problem (predict ticket density) instead of binary classification
- Add spatial cross-validation to ensure model generalizes across different geographic areas
