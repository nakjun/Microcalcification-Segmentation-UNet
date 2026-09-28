"""Annotation-limited INbreast evaluation after every 48x410 image-only prediction is complete."""
from __future__ import annotations

import fcntl
from pathlib import Path

import cv2
import numpy as np

from annotations import build as build_annotations
from common import ROOT, VARIANTS, atomic_json, checksum, context, digest, immutable, prediction_dir, read, verify
from infer import primary_mask, ungated_mask

ENDPOINTS = dict(
    primary='external_v3 fixed gate AND p>=0.9; every image nevertheless received refinement inference',
    secondary='ungated p>=0.5; reported separately from primary',
    point='np.rint ties-to-even; exact and Chebyshev radius1 native-pixel hit; not clinical sensitivity',
    contour='union of valid Calcification contours; full-image Dice/IoU against incomplete contour annotations; not exhaustive pixel accuracy',
    negative_images='missing XML and annotation absence are unknown, never negative',
    selection='none; per-fold checkpoint results and fold mean/min/max only, no ensemble')


def average(values):
    return float(np.mean(values)) if values else None


def image_metrics(mask, annotation):
    height, width = mask.shape
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    result = dict(predicted_pixels=int(mask.sum()), predicted_fraction=float(mask.mean()),
                  predicted_components=int(count - 1), image_predicted_positive=bool(mask.any()))
    point_hits = []
    for point in annotation['points']:
        x, y = point['x'], point['y']
        point_hits.append(dict(roi_id=point['roi_id'], exact=bool(mask[y, x]),
                               within1=bool(mask[max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)].any())))
    result.update(points_total=len(point_hits), points_exact=sum(x['exact'] for x in point_hits),
                  points_within1=sum(x['within1'] for x in point_hits), point_hits=point_hits)
    target = np.zeros(mask.shape, np.uint8)
    contour_hits = []
    for contour in annotation['contours']:
        xy = np.asarray(contour['raster_xy'], np.int32)
        cv2.fillPoly(target, [xy], 1)
        x0, y0 = xy.min(0); x1, y1 = xy.max(0)
        local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(local, [xy - [x0, y0]], 1)
        contour_hits.append(dict(roi_id=contour['roi_id'], any_overlap=bool(np.any(mask[y0:y1 + 1, x0:x1 + 1] & local.astype(bool)))))
    target_pixels = int(target.sum())
    overlap = int(np.count_nonzero(mask & target.astype(bool)))
    predicted = result['predicted_pixels']
    result.update(contours_total=len(contour_hits), contours_any_overlap=sum(x['any_overlap'] for x in contour_hits),
                  contour_hits=contour_hits, contour_union_pixels=target_pixels, contour_overlap_pixels=overlap,
                  partial_contour_Dice=(2 * overlap / (target_pixels + predicted)) if target_pixels else None,
                  partial_contour_IoU=(overlap / (target_pixels + predicted - overlap)) if target_pixels else None,
                  predicted_pixels_outside_contour_union_not_confirmed_FP=(predicted - overlap) if target_pixels else None)
    return result


def summarize(cases, policy):
    values = [case['policies'][policy] for case in cases]
    summary = {key: sum(item[key] for item in values) for key in ('predicted_pixels', 'predicted_components', 'image_predicted_positive',
                                                                      'points_total', 'points_exact', 'points_within1',
                                                                      'contours_total', 'contours_any_overlap', 'contour_union_pixels',
                                                                      'contour_overlap_pixels')}
    summary.update(mean_predicted_pixels=average([x['predicted_pixels'] for x in values]),
                   mean_predicted_fraction=average([x['predicted_fraction'] for x in values]),
                   point_exact_hit_rate=summary['points_exact'] / summary['points_total'] if summary['points_total'] else None,
                   point_within1_hit_rate=summary['points_within1'] / summary['points_total'] if summary['points_total'] else None,
                   contour_annotated_images=sum(x['contour_union_pixels'] > 0 for x in values),
                   partial_contour_macro_Dice=average([x['partial_contour_Dice'] for x in values if x['partial_contour_Dice'] is not None]),
                   partial_contour_macro_IoU=average([x['partial_contour_IoU'] for x in values if x['partial_contour_IoU'] is not None]),
                   predicted_pixels_outside_contour_union_not_confirmed_FP=sum(x['predicted_pixels_outside_contour_union_not_confirmed_FP'] or 0 for x in values))
    return summary


def fold_range(folds, policy):
    """Every scalar result is retained by fold; this compact view is only a descriptive range."""
    keys = [key for key, value in folds[0]['metrics'][policy].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {key: dict(mean=float(np.mean([fold['metrics'][policy][key] for fold in folds])),
                      min=float(np.min([fold['metrics'][policy][key] for fold in folds])),
                      max=float(np.max([fold['metrics'][policy][key] for fold in folds]))) for key in keys}


def _verify_all_prediction_cohorts(c):
    expected_ids = [row['image_id'] for row in c['rows']]
    for checkpoint in c['checkpoints']:
        output = prediction_dir(checkpoint) / 'predictions'
        done = verify(output / 'complete.json', c['signature'])
        assert done['image_ids'] == expected_ids and not done['smoke'] and done['GT_read'] is False
        assert done['checkpoint_sha256'] == checkpoint['checkpoint_sha256']
        expected = {}
        for row in c['rows']:
            path = output / f"{row['image_id']}.npz"
            marker = path.with_suffix('.json')
            meta = verify(marker, c['signature'])
            assert meta['image_id'] == row['image_id'] and meta['dicom_sha256'] == row['dicom_sha256']
            assert meta['refinement_executed'] is True and meta['smoke'] is False and meta['GT_read'] is False
            assert meta['checkpoint_sha256'] == checkpoint['checkpoint_sha256']
            expected[str(path)] = checksum(path); expected[str(marker)] = checksum(marker)
        assert done['output_files'] == expected


def _read_probability(path, shape):
    with np.load(path) as data:
        assert tuple(data['shape']) == shape
        flat, probability = data['flat'], data['probability']
        assert flat.dtype == np.int64 and probability.dtype == np.float32 and flat.shape == probability.shape
        assert np.all(flat[1:] > flat[:-1]) and np.all((flat >= 0) & (flat < np.prod(shape)))
        assert np.isfinite(probability).all() and np.all((probability >= np.float32(.3)) & (probability <= 1))
        dense = np.zeros(np.prod(shape), np.float32); dense[flat] = probability
        return dense.reshape(shape), float(data['gate_logit']), bool(data['gate_pass'])


def main():
    c = context()
    destination = ROOT / 'evaluation'
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This complete-cohort check is intentionally before XML parsing.
        aggregate = verify(ROOT / 'predictions/complete.json', c['signature'])
        assert aggregate['checkpoints'] == 48 and aggregate['images_per_checkpoint'] == 410
        _verify_all_prediction_cohorts(c)
        annotation_document = build_annotations(c['rows'])
        for path, sha in annotation_document['source_files'].items():
            assert checksum(path) == sha
        annotations = {item['image_id']: item for item in annotation_document['records']}
        assert set(annotations) == {row['image_id'] for row in c['rows']}
        source_hashes = {str(ROOT / 'protocol.json'): checksum(ROOT / 'protocol.json'),
                         str(ROOT / 'annotations/inbreast.json'): checksum(ROOT / 'annotations/inbreast.json'),
                         str(Path(__file__)): checksum(__file__),
                         str(Path(__file__).with_name('annotations.py')): checksum(Path(__file__).with_name('annotations.py'))}
        evaluation_protocol = dict(inference_signature=c['signature'], endpoints=ENDPOINTS, inputs=source_hashes)
        signature = digest(evaluation_protocol)
        immutable(destination / 'protocol.json', evaluation_protocol)
        output_files = dict(source_hashes); output_files[str(destination / 'protocol.json')] = checksum(destination / 'protocol.json')
        grouped = {}
        gate_threshold = float(c['protocol']['gate']['threshold_logit'])
        for checkpoint in c['checkpoints']:
            key = (checkpoint['model'], checkpoint['variant'])
            cases = []
            input_dir = prediction_dir(checkpoint) / 'predictions'
            for row in c['rows']:
                path = input_dir / f"{row['image_id']}.npz"
                case_path = destination / 'cases' / checkpoint['model'] / checkpoint['variant'] / f"fold_{checkpoint['fold']}" / f"{row['image_id']}.json"
                if case_path.exists():
                    case = read(case_path)
                    assert case['signature'] == signature and case['probability_sha256'] == checksum(path)
                else:
                    probability, logit, gate = _read_probability(path, (row['height'], row['width']))
                    assert gate == bool(logit >= gate_threshold)
                    annotation = annotations[row['image_id']]
                    case = dict(signature=signature, image_id=row['image_id'], model=checkpoint['model'], variant=checkpoint['variant'], fold=checkpoint['fold'],
                                annotation_status=annotation['status'], annotation_issues=annotation['issues'], probability_sha256=checksum(path),
                                gate_logit=logit, gate_pass=gate,
                                policies=dict(primary=image_metrics(primary_mask(probability, gate), annotation),
                                              ungated_p5=image_metrics(ungated_mask(probability), annotation)))
                    immutable(case_path, case)
                cases.append(case); output_files[str(case_path)] = checksum(case_path)
            fold_result = dict(fold=checkpoint['fold'], checkpoint=str(checkpoint['checkpoint']), checkpoint_sha256=checkpoint['checkpoint_sha256'],
                               images=len(cases), metrics={policy: summarize(cases, policy) for policy in ('primary', 'ungated_p5')})
            grouped.setdefault(key, []).append(fold_result)
        conditions = {}
        for (model, variant), folds in grouped.items():
            assert [item['fold'] for item in folds] == [0, 1, 2]
            conditions.setdefault(model, {})[variant] = dict(folds=folds, fold_mean_min_max={policy: fold_range(folds, policy) for policy in ('primary', 'ungated_p5')})
        immutable(destination / 'complete.json', dict(complete=True, signature=signature, inference_signature=c['signature'], dataset='inbreast',
                                                       images=410, checkpoints=48, models=4, variants=list(VARIANTS), annotation_summary=annotation_document['summary'],
                                                       endpoints=ENDPOINTS, selection_performed=False, ensemble=False,
                                                       all_refinement_executed_before_gate=True, conditions=conditions, output_files=output_files))


if __name__ == '__main__':
    main()
