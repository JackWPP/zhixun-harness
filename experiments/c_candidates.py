"""CPU-only C continuation experiments. Training uses public/train exclusively.

Run with Python + NumPy from the repository root. The official validation set
is evaluation-only. All candidates start at the same official checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rsi.model import ExtendedModel

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'datasets/cpu-c-guide-20261110-v1'
OUTPUT = ROOT / 'outputs/c_candidates'


def runtime_info():
    return {'numpy_version': np.__version__, 'python_version': platform.python_version(),
            'platform': platform.platform(), 'configured_blas_threads': 1}


def load_rows(path, model):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    paths = [[key for _, key in json.loads(path)] for path in model.meta['paths']]
    def feature(row, keys):
        value = row['inputs']
        for key in keys:
            value = value[key]
        return value
    x = np.array([[feature(row, keys) for keys in paths] for row in rows])
    y = np.array([row['answer'] for row in rows], dtype=float)
    groups = np.array([row['id'].rsplit('_c', 1)[0] for row in rows])
    return x, y, groups, hashlib.sha256(raw).hexdigest()


def metrics(pred, y, groups):
    err = pred - y
    penalties = np.expm1(np.where(err < 0, -err / 13.0, err / 10.0))
    unit_metrics = []
    for group in np.unique(groups):
        mask = groups == group
        unit_metrics.append({'unit': str(group), 'rows': int(mask.sum()),
                             'rmse': float(np.sqrt(np.mean(err[mask] ** 2))),
                             'nasa': float(penalties[mask].sum())})
    return {'rows': len(y), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mae': float(np.mean(abs(err))), 'nasa': float(penalties.sum()),
            'bias': float(err.mean()), 'max_absolute_error': float(abs(err).max()),
            'p99_absolute_error': float(np.quantile(abs(err), .99)),
            'nasa_underprediction': float(penalties[err < 0].sum()),
            'nasa_overprediction': float(penalties[err >= 0].sum()),
            'worst_units': sorted(unit_metrics, key=lambda row: row['nasa'], reverse=True)[:8],
            'all_units': unit_metrics}


def fit_huber(model, x, y, epochs, delta_cycles):
    normalized = model.normalized(x)
    target = (y - model.meta['target_mean']) / model.meta['target_scale']
    threshold = delta_cycles / model.meta['target_scale']
    for _ in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([model.meta['seed'], model.meta['epochs']]))
        order = rng.permutation(len(x))
        for start in range(0, len(order), model.meta['batch_size']):
            indices = order[start:start + model.meta['batch_size']]
            batch = normalized[indices]
            hidden, logits = model.forward(batch)
            error = logits[:, 0] - target[indices]
            delta = (2 * np.clip(error, -threshold, threshold) / len(indices))[:, None]
            gradients = {'w2': hidden.T @ delta, 'b2': delta.sum(axis=0)}
            hidden_delta = (delta @ model.params['w2'].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(gradient ** 2)) for gradient in gradients.values()))
            model.meta['steps'] += 1
            for name, gradient in gradients.items():
                gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * gradient
                model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
                first = model.first[name] / (1 - .9 ** model.meta['steps'])
                second = model.second[name] / (1 - .999 ** model.meta['steps'])
                model.params[name] -= model.meta['learning_rate'] * first / (np.sqrt(second) + 1e-8)
        model.meta['epochs'] += 1
    model.validate()


def paired_bootstrap(baseline_pred, first_pred, val_y, val_groups):
    def contributions(predicted):
        error = predicted - val_y
        penalty = np.expm1(np.where(error < 0, -error / 13, error / 10))
        return np.array([[np.sum(error[val_groups == group] ** 2),
                          np.sum(penalty[val_groups == group]), np.sum(val_groups == group)]
                         for group in np.unique(val_groups)])
    base = contributions(baseline_pred)
    changed = contributions(first_pred)
    rng = np.random.default_rng(20261126)
    indices = rng.integers(0, len(base), (2000, len(base)))
    base_draws = base[indices].sum(axis=1)
    changed_draws = changed[indices].sum(axis=1)
    rmse_delta = np.sqrt(changed_draws[:, 0] / changed_draws[:, 2]) - np.sqrt(base_draws[:, 0] / base_draws[:, 2])
    nasa_relative_change = changed_draws[:, 1] / base_draws[:, 1] - 1
    return {
        'resamples': 2000, 'engine_count': len(base), 'seed': 20261126,
        'rmse_delta_percentiles_2_5_50_97_5': np.quantile(rmse_delta, [.025, .5, .975]).tolist(),
        'nasa_relative_change_percentiles_2_5_50_97_5': np.quantile(nasa_relative_change, [.025, .5, .975]).tolist(),
        'resamples_rmse_improved': int(np.sum(rmse_delta < 0)),
        'resamples_nasa_improved': int(np.sum(nasa_relative_change < 0)),
        'caveat': 'Candidate was selected on this validation set. Intervals describe paired resampling sensitivity and are not an unbiased confirmatory test or hidden-set forecast.'}


def robustness(parent, train_x, train_y, val_x, val_y, val_groups):
    """Exploratory seed/subgroup/repeated-round checks, all on public data."""
    baseline_pred = parent.predict(val_x)
    baseline_values = metrics(baseline_pred, val_y, val_groups)
    result = {'scope': 'Exploratory public validation checks, not hidden-test evidence',
              'runtime': runtime_info(),
              'repeated_round_seed': 20261117,
              'baseline': baseline_values, 'seed_results': [], 'repeated_rounds': []}
    for seed in (17, 73, 91):
        candidate = parent.clone()
        candidate.meta['seed'] = seed
        candidate.meta['learning_rate'] = .0003
        start = time.process_time()
        candidate.fit(train_x, train_y, 10)
        predicted = candidate.predict(val_x)
        score = metrics(predicted, val_y, val_groups)
        result['seed_results'].append({'seed': seed, 'cpu_seconds': time.process_time() - start,
                                       'validation': score})
        checkpoint = OUTPUT / f'mse_lr0003_e10_seed{seed}.npz'
        if not checkpoint.exists():
            candidate.save(checkpoint)
        print('seed', seed, score['rmse'], score['mae'], score['nasa'], flush=True)

    candidate = parent.clone()
    candidate.meta['seed'] = 20261117
    candidate.meta['learning_rate'] = .0003
    first_pred = None
    for round_index in range(1, 11):
        start = time.process_time()
        candidate.fit(train_x, train_y, 10)
        predicted = candidate.predict(val_x)
        if round_index == 1:
            first_pred = predicted.copy()
        score = metrics(predicted, val_y, val_groups)
        result['repeated_rounds'].append({'round': round_index,
                                         'cpu_seconds': time.process_time() - start,
                                         'validation': score})
        print('round', round_index, score['rmse'], score['mae'], score['nasa'], flush=True)

    masks = {subset: np.array([group.startswith(subset) for group in val_groups])
             for subset in ('fd001', 'fd002', 'fd003', 'fd004')}
    masks.update({'RUL_0_99': val_y < 100, 'RUL_100_249': (val_y >= 100) & (val_y < 250),
                  'RUL_250_plus': val_y >= 250})
    result['first_candidate_subgroups'] = {
        label: {'baseline': metrics(baseline_pred[mask], val_y[mask], val_groups[mask]),
                'candidate': metrics(first_pred[mask], val_y[mask], val_groups[mask])}
        for label, mask in masks.items()}

    result['paired_engine_bootstrap'] = paired_bootstrap(baseline_pred, first_pred, val_y, val_groups)
    (OUTPUT / 'robustness.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))


def cost_comparison(parent, train_x, train_y, val_x, val_y, val_groups):
    from rsi.model import Model
    records = []
    configurations = [('baseline', 0, parent.meta['learning_rate'], parent.meta['seed'], False),
                      ('low_cost_default_seed', 10, .0003, 20261117, False),
                      ('old_continue40', 40, .001, 20261117, False),
                      ('old_scratch300', 300, .001, 17, True)]
    for name, epochs, lr, seed, scratch in configurations:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if scratch:
            model = Model('regression', parent.meta['paths'], [], train_x, train_y,
                          hidden=32, seed=seed, learning_rate=lr)
        else:
            model = parent.clone()
            model.meta['seed'] = seed
            model.meta['learning_rate'] = lr
        if epochs:
            model.fit(train_x, train_y, epochs)
        cpu_seconds = time.process_time() - start_cpu
        wall_seconds = time.perf_counter() - start_wall
        infer_start = time.perf_counter()
        predicted = model.predict(val_x)
        predict_seconds = time.perf_counter() - infer_start
        score = metrics(predicted, val_y, val_groups)
        subgroups = {}
        for subset in ('fd001', 'fd002', 'fd003', 'fd004'):
            mask = np.array([group.startswith(subset) for group in val_groups])
            subgroups[subset] = metrics(predicted[mask], val_y[mask], val_groups[mask])
        record = {'candidate': name, 'epochs': epochs, 'learning_rate': lr,
                  'runtime': runtime_info(),
                  'seed': seed, 'from_scratch': scratch, 'train_cpu_seconds': cpu_seconds,
                  'train_wall_seconds': wall_seconds, 'predict_seconds': predict_seconds,
                  'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'validation': score, 'subgroups': subgroups}
        if name == 'low_cost_default_seed':
            record['paired_engine_bootstrap'] = paired_bootstrap(parent.predict(val_x), predicted, val_y, val_groups)
        checkpoint = OUTPUT / f'{name}.npz'
        if not checkpoint.exists():
            model.save(checkpoint)
        assert np.array_equal(ExtendedModel.load(checkpoint).predict(val_x), predicted)
        np.savez(OUTPUT / f'{name}_predictions.npz', predicted=predicted, expected=val_y, groups=val_groups)
        records.append(record)
        (OUTPUT / 'cost_comparison.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(name, cpu_seconds, wall_seconds, score['rmse'], score['mae'], score['nasa'], flush=True)


def run():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness', action='store_true')
    parser.add_argument('--cost-comparison', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    OUTPUT.mkdir(exist_ok=True, parents=True)
    parent = ExtendedModel.load(PACKAGE / 'checkpoint.npz')
    train_x, train_y, train_groups, train_hash = load_rows(PACKAGE / 'public/train.jsonl', parent)
    val_x, val_y, val_groups, val_hash = load_rows(PACKAGE / 'public/validation.jsonl', parent)
    assert not set(train_groups) & set(val_groups)
    manifest = json.loads((PACKAGE / 'training.json').read_text())
    assert 'sha256:' + train_hash == manifest['train_sha256']
    assert 'sha256:' + val_hash == manifest['validation_sha256']
    assert 'sha256:' + hashlib.sha256((PACKAGE / 'checkpoint.npz').read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    if args.robustness:
        robustness(parent, train_x, train_y, val_x, val_y, val_groups)
        return
    if args.cost_comparison:
        cost_comparison(parent, train_x, train_y, val_x, val_y, val_groups)
        return
    definitions = [
        ('baseline', 'none', 0, .002, 0),
        ('mse_lr0003_e10', 'mse', 10, .0003, 0),
        ('mse_lr0003_e30', 'mse', 30, .0003, 0),
        ('huber80_lr0005_e10', 'huber', 10, .0005, 80),
        ('huber80_lr0005_e30', 'huber', 30, .0005, 80),
        ('huber40_lr0005_e30', 'huber', 30, .0005, 40),
    ]
    records = []
    for name, loss, epochs, lr, threshold in definitions:
        model = parent.clone()
        model.meta['learning_rate'] = lr
        wall, cpu = time.perf_counter(), time.process_time()
        if loss == 'mse':
            model.fit(train_x, train_y, epochs)
        elif loss == 'huber':
            fit_huber(model, train_x, train_y, epochs, threshold)
        train_seconds = time.perf_counter() - wall
        cpu_seconds = time.process_time() - cpu
        evaluation_start = time.perf_counter()
        predicted = model.predict(val_x)
        score = metrics(predicted, val_y, val_groups)
        record = {'candidate': name, 'loss': loss, 'epochs': epochs,
                  'learning_rate': lr, 'huber_delta_cycles': threshold,
                  'training_seconds': train_seconds, 'cpu_seconds': cpu_seconds,
                  'validation_seconds': time.perf_counter() - evaluation_start,
                  'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'validation': score}
        records.append(record)
        destination = OUTPUT / name
        destination.mkdir(exist_ok=True)
        checkpoint = destination / 'checkpoint.npz'
        if not checkpoint.exists():
            model.save(checkpoint)
        reloaded = ExtendedModel.load(checkpoint)
        assert np.array_equal(reloaded.predict(val_x), predicted)
        record['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        np.savez(destination / 'validation_predictions.npz', predicted=predicted, expected=val_y, groups=val_groups)
        (OUTPUT / 'results.json').write_text(json.dumps({'train_sha256':train_hash,'validation_sha256':val_hash,
            'numpy_version':np.__version__,'records':records},ensure_ascii=False,indent=2))
        print(json.dumps({key:value for key,value in record.items() if key != 'validation'}), flush=True)
        print(json.dumps({key:score[key] for key in ('rmse','mae','nasa','p99_absolute_error')}), flush=True)

if __name__ == '__main__':
    run()
