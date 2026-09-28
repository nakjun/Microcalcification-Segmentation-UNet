"""Train-only grouped CV and a frozen-encoder linear gate; no test evaluation."""
import argparse
import csv
import fcntl
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from calcification_gate_common import atomic_json, checksum, classification_metrics, validate_shard


def fit_head(x, y, c_value, config):
    y = np.asarray(y)
    assert set(np.unique(y)) == {0, 1} and np.isfinite(x).all()
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    ratio = float((y == 0).sum() / (y == 1).sum())
    model = LogisticRegression(C=c_value, class_weight={0: 1., 1: ratio}, solver=config['solver'],
                               max_iter=config['max_iter'], tol=config['tol'], random_state=config['seed'])
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(z, y)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    logits = model.decision_function(z)
    weight = np.where(y == 1, ratio, 1.)
    bce = float(np.average(np.logaddexp(0, logits) - y * logits, weights=weight))
    metadata = {'C': c_value, 'fit_images': len(y), 'fit_positives': int(y.sum()),
                'positive_weight': ratio, 'optimizer_iterations': int(model.n_iter_.max()),
                'weighted_BCE': bce, 'elapsed_seconds': time.monotonic() - started, 'converged': True}
    return scaler, model, metadata


def model_arrays(scaler, model):
    return {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': model.coef_[0], 'intercept': model.intercept_[0]}


def score_arrays(arrays, x):
    return ((x.astype(np.float64) - arrays['mean']) / arrays['scale']) @ arrays['coef'] + arrays['intercept']


def event(root, state, **data):
    row = {'state': state, 'updated_at_unix': time.time(), 'pid': os.getpid(), **data}
    atomic_json(root / 'status.json', row)
    with (root / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def run(args):
    config = json.loads(Path(args.config).read_text())
    out = Path(config['root']) / ('smoke_head' if args.smoke else 'head_seed42')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    feature_root = Path(config['root']) / ('smoke_features' if args.smoke else 'features')
    complete = json.loads((feature_root / 'complete.json').read_text())
    assert complete['complete'] and complete['smoke'] == args.smoke
    for path, sha in complete['signature'].items():
        assert checksum(path) == sha, f'Feature source signature mismatch: {path}'
    all_rows = json.loads(Path(config['manifest']).read_text())
    byid = {r['image_id']: r for r in all_rows}
    rows = [byid[i] for i in complete['image_ids']]
    assert len(rows) == len(set(complete['image_ids'])) == complete['rows']
    assert all(r['split'] in ('train', 'val') for r in rows)
    if not args.smoke:
        assert set(complete['image_ids']) == {r['image_id'] for r in all_rows if r['split'] in ('train', 'val')}
    chunks = []
    offset = 0
    for shard in complete['shards']:
        assert shard['offset'] == offset and checksum(shard['path']) == shard['sha256']
        subset = rows[offset:offset + shard['images']]
        chunks.append(validate_shard(shard['path'], subset, complete['signature']))
        offset += shard['images']
    assert offset == len(rows)
    x = np.concatenate(chunks).astype(np.float64)
    y = np.array([r['label'] for r in rows])
    sig = {'feature_signature': complete['signature'], 'shards': complete['shards'],
           'training_source_sha256': checksum(__file__), 'smoke': args.smoke}
    if (out / 'signature.json').exists():
        assert json.loads((out / 'signature.json').read_text()) == sig, 'Head protocol/source changed'
    else:
        atomic_json(out / 'signature.json', sig)
    if (out / 'complete.json').exists():
        done = json.loads((out / 'complete.json').read_text())
        assert checksum(out / 'classifier.npz') == done['classifier_sha256']
        print('Classifier training already complete and verified.', flush=True)
        return
    with threadpool_limits(limits=config['cpu_threads']):
        if args.smoke:
            scaler, model, metadata = fit_head(x, y, .01, config)
            arrays = model_arrays(scaler, model)
            assert np.allclose(score_arrays(arrays, x), model.decision_function(scaler.transform(x)), rtol=0, atol=1e-12)
            with (out / 'classifier.npz').open('wb') as f:
                np.savez(f, **arrays)
            assert metadata['optimizer_iterations'] > 0
            atomic_json(out / 'complete.json', {'smoke': True, 'scope': '16-image training-path test only; not a performance estimate',
                        'classifier_sha256': checksum(out / 'classifier.npz'), **metadata})
            event(out, 'smoke_complete', **metadata)
            return
        train = np.array([r['split'] == 'train' for r in rows])
        validation = np.array([r['split'] == 'val' for r in rows])
        fold_ids = np.array([r['inner_fold'] if r['inner_fold'] is not None else -1 for r in rows])
        candidates = []
        for c_value in config['C_candidates']:
            fold_results = []
            oof = np.full(len(rows), np.nan)
            for fold in range(config['inner_folds']):
                fit = train & (fold_ids != fold)
                held = train & (fold_ids == fold)
                assert not {r['component_id'] for i, r in enumerate(rows) if fit[i]} & {
                    r['component_id'] for i, r in enumerate(rows) if held[i]}
                event(out, 'cv_fit', C=c_value, fold=fold, fit_images=int(fit.sum()), held_images=int(held.sum()))
                scaler, model, metadata = fit_head(x[fit], y[fit], c_value, config)
                logits = model.decision_function(scaler.transform(x[held]))
                oof[held] = logits
                metrics = classification_metrics(y[held], logits)
                result = {'fold': fold, **metadata, 'held_metrics': metrics}
                atomic_json(out / f'cv_C{c_value:g}_fold{fold}.json', result)
                fold_results.append(result)
                event(out, 'cv_fold_complete', C=c_value, fold=fold,
                      average_precision=metrics['average_precision'], roc_auc=metrics['roc_auc'],
                      negative_rejection_at_full_sensitivity=metrics['operating_points'][0]['negative_rejection'],
                      **{k: metadata[k] for k in ['optimizer_iterations', 'weighted_BCE', 'elapsed_seconds']})
            assert np.isfinite(oof[train]).all() and np.isnan(oof[validation]).all()
            with (out / f'oof_C{c_value:g}.npz').open('wb') as f:
                np.savez(f, image_ids=np.asarray([r['image_id'] for i, r in enumerate(rows) if train[i]]),
                         labels=y[train], logits=oof[train])
            candidates.append({'C': c_value,
                               'mean_negative_rejection_at_full_sensitivity': float(np.mean([
                                   r['held_metrics']['operating_points'][0]['negative_rejection'] for r in fold_results])),
                               'mean_average_precision': float(np.mean([r['held_metrics']['average_precision'] for r in fold_results])),
                               'folds': fold_results})
            atomic_json(out / 'cv_results.json', candidates)
        selected = max(candidates, key=lambda r: (r['mean_negative_rejection_at_full_sensitivity'],
                                                  r['mean_average_precision'], -r['C']))
        atomic_json(out / 'selection.json', {'selected_C': selected['C'], 'scope': 'clean train inner 5-fold CV only',
                    'criterion': config['selection'], 'validation_used_for_C_selection': False})
        event(out, 'final_fit', selected_C=selected['C'], train_images=int(train.sum()))
        scaler, model, metadata = fit_head(x[train], y[train], selected['C'], config)
        arrays = model_arrays(scaler, model)
        logits = score_arrays(arrays, x[validation])
        assert np.allclose(logits, model.decision_function(scaler.transform(x[validation])), rtol=0, atol=1e-12)
        temporary = out / 'classifier.npz.tmp'
        with temporary.open('wb') as f:
            np.savez(f, **arrays)
        temporary.replace(out / 'classifier.npz')
        with np.load(out / 'classifier.npz', allow_pickle=False) as saved:
            assert np.array_equal(score_arrays(saved, x[validation]), logits)
        metrics = classification_metrics(y[validation], logits)
        metrics.update(scope='clean development validation; operating thresholds selected on these same labels',
                       selected_C=selected['C'], images=int(validation.sum()), positive_images=int(y[validation].sum()),
                       test_evaluated=False)
        atomic_json(out / 'validation_metrics.json', metrics)
        primary = metrics['operating_points'][0]
        thresholds = {'comparison': 'logit >= threshold_logit; include ties', 'primary': primary,
                      'secondary': metrics['operating_points'][1:], 'scope': metrics['scope'],
                      'score_interpretation': 'Class-weighted score, not calibrated disease probability'}
        atomic_json(out / 'thresholds.json', thresholds)
        validation_rows = [r for i, r in enumerate(rows) if validation[i]]
        with (out / 'validation_predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'study_id', 'label', 'logit', 'score', 'gate_pass'])
            writer.writeheader()
            for row, logit in zip(validation_rows, logits):
                writer.writerow({'image_id': row['image_id'], 'study_id': row['study_id'], 'label': row['label'],
                                 'logit': repr(float(logit)), 'score': repr(float(expit(logit))),
                                 'gate_pass': int(logit >= primary['threshold_logit'])})
        with (out / 'validation_scores.npz').open('wb') as f:
            np.savez(f, image_ids=np.asarray([r['image_id'] for r in validation_rows]), labels=y[validation], logits=logits)
        done = {'complete': True, 'smoke': False, 'finished_at_unix': time.time(),
                'classifier_sha256': checksum(out / 'classifier.npz'), 'manifest_sha256': checksum(config['manifest']),
                'selected_C': selected['C'], 'final_fit': metadata, 'validation': metrics,
                'test_evaluated': False, 'cascade_evaluated': False}
        atomic_json(out / 'complete.json', done)
        event(out, 'complete', selected_C=selected['C'], validation=metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_gate.json')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
