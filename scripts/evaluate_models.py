#!/usr/bin/env python3
"""Compare the most recent sklearn and LightGBM model results.

This script finds the newest `sklearn-YYYY-MM-DD` and `lightgbm-YYYY-MM-DD`
bundles under `scripts/models/`, compares their `results.json` metrics, picks the
best model, copies its pickle to `api/model.pkl`, and writes a comparison
bundle under `models/comparison-YYYY-MM-DD/`.
"""

import json, logging, math, shutil
from datetime import date
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MODELS_DIR = REPO_ROOT / 'scripts' / 'models'
API_MODEL_PATH = REPO_ROOT / 'api' / 'model.pkl'
COMPARISON_BASE_DIR = REPO_ROOT / 'scripts' / 'models'


MODEL_TYPES = ['sklearn', 'lightgbm']
DEFAULT_METRIC = 'roc_auc'
FALLBACK_METRIC = 'accuracy'


def load_json_allow_nan(path: Path) -> dict:
    text = path.read_text()
    return json.loads(text, parse_constant=lambda x: float('nan'))


def find_latest_run(model_type: str) -> Optional[Path]:
    prefix = f'{model_type}-'
    candidates = [p for p in TRAIN_MODELS_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None

    def parse_date(dir_path: Path):
        try:
            return date.fromisoformat(dir_path.name.split('-', 1)[1])
        except Exception:
            return date.min

    latest = max(candidates, key=parse_date)
    return latest


def safe_metric_value(metrics: dict, metric_name: str) -> Optional[float]:
    value = metrics.get(metric_name)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def choose_best_model(candidates: dict) -> dict:
    ranked = []
    for model_type, payload in candidates.items():
        results = payload['results']
        cv_metrics = results.get('cv_results', {})
        test_metrics = results.get('test_metrics', {})

        cv_roc = safe_metric_value(cv_metrics.get(DEFAULT_METRIC, {}), 'test_mean')
        test_roc = safe_metric_value(test_metrics, DEFAULT_METRIC)
        cv_acc = safe_metric_value(cv_metrics.get(FALLBACK_METRIC, {}), 'test_mean')
        test_acc = safe_metric_value(test_metrics, FALLBACK_METRIC)

        if cv_roc is not None:
            score = (4, cv_roc)
            chosen_metric = 'cv_roc'
            chosen_value = cv_roc
        elif test_roc is not None:
            score = (3, test_roc)
            chosen_metric = 'test_roc'
            chosen_value = test_roc
        elif cv_acc is not None:
            score = (2, cv_acc)
            chosen_metric = 'cv_accuracy'
            chosen_value = cv_acc
        elif test_acc is not None:
            score = (1, test_acc)
            chosen_metric = 'test_accuracy'
            chosen_value = test_acc
        else:
            score = (0, -1.0)
            chosen_metric = 'none'
            chosen_value = None

        ranked.append((score, model_type, payload, chosen_metric, chosen_value))

    ranked.sort(reverse=True)
    _, _, best_payload, best_metric, best_value = ranked[0]
    best_payload['chosen_metric'] = best_metric
    best_payload['chosen_metric_value'] = best_value
    return best_payload


def build_candidate(model_type: str) -> Optional[dict]:
    run_dir = find_latest_run(model_type)
    if run_dir is None:
        logger.warning('No runs found for %s', model_type)
        return None

    results_path = run_dir / 'results.json'
    model_path = run_dir / f'{run_dir.name}.pkl'
    if not results_path.exists():
        raise FileNotFoundError(f'Missing results.json for {model_type} at {results_path}')
    if not model_path.exists():
        raise FileNotFoundError(f'Missing model.pkl for {model_type} at {model_path}')

    results = load_json_allow_nan(results_path)
    return {
        'model_type': model_type,
        'run_dir': run_dir,
        'results_path': results_path,
        'model_path': model_path,
        'results': results,
    }


def prepare_comparison_bundle(best: dict, candidates: dict) -> Path:
    comparison_dir = COMPARISON_BASE_DIR / f'comparison-{date.today().isoformat()}'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'selected_model_type': best['model_type'],
        'selected_run_dir': str(best['run_dir']),
        'selected_model_path': str(best['model_path']),
        'api_model_path': str(API_MODEL_PATH),
        'metric': best.get('chosen_metric', DEFAULT_METRIC),
        'chosen_metric_value': best.get('chosen_metric_value'),
        'candidates': {},
    }

    for model_type, payload in candidates.items():
        candidate_results = payload['results']
        candidate_metrics = candidate_results.get('test_metrics', {})
        candidate_cv = candidate_results.get('cv_results', {})
        summary['candidates'][model_type] = {
            'run_dir': str(payload['run_dir']),
            'model_path': str(payload['model_path']),
            'test_metrics': {
                'accuracy': safe_metric_value(candidate_metrics, 'accuracy'),
                'precision': safe_metric_value(candidate_metrics, 'precision'),
                'recall': safe_metric_value(candidate_metrics, 'recall'),
                'f1': safe_metric_value(candidate_metrics, 'f1'),
                'roc_auc': safe_metric_value(candidate_metrics, 'roc_auc'),
            },
            'cv_metrics': {
                'accuracy': safe_metric_value(candidate_cv.get('accuracy', {}), 'test_mean'),
                'precision': safe_metric_value(candidate_cv.get('precision', {}), 'test_mean'),
                'recall': safe_metric_value(candidate_cv.get('recall', {}), 'test_mean'),
                'f1': safe_metric_value(candidate_cv.get('f1', {}), 'test_mean'),
                'roc_auc': safe_metric_value(candidate_cv.get('roc_auc', {}), 'test_mean'),
            },
        }

    results_path = comparison_dir / 'results.json'
    with results_path.open('w') as handle:
        json.dump(summary, handle, indent=2)

    return comparison_dir


def copy_model_to_api(best: dict) -> None:
    API_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best['model_path'], API_MODEL_PATH)
    logger.info('Copied best model %s to %s', best['model_path'], API_MODEL_PATH)


def main() -> int:
    candidates = {}
    for model_type in MODEL_TYPES:
        payload = build_candidate(model_type)
        if payload is not None:
            candidates[model_type] = payload

    if not candidates:
        logger.error('No model candidates found. Aborting.')
        return 1

    best = choose_best_model(candidates)
    logger.info('Selected best model: %s from %s', best['model_type'], best['run_dir'])

    copy_model_to_api(best)
    comparison_dir = prepare_comparison_bundle(best, candidates)

    logger.info('Comparison results written to: %s', comparison_dir / 'results.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
