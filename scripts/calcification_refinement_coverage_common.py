"""Fixed-gate diagnostic counts for full-image and two-region threshold policies."""
import numpy as np

from calcification_refinement_common import pixel_counts
from calcification_gate_sam_common import summarize_segmentation


def candidates(config):
    result = []
    for route in ['boxes', 'full_image']:
        for threshold in config['threshold_grid']:
            result.append({'id': f'{route}@{threshold:g}', 'route': route,
                           'inside_threshold': threshold,
                           'outside_threshold': threshold if route == 'full_image' else None})
    for inside in config['inside_thresholds']:
        for outside in config['outside_thresholds']:
            if outside >= inside:
                result.append({'id': f'two_region@{inside:g}|{outside:g}', 'route': 'two_region',
                               'inside_threshold': inside, 'outside_threshold': outside})
    assert len({r['id'] for r in result}) == len(result)
    return result


def above(values, thresholds):
    """Use explicit FP32 inclusive comparisons, including thresholds like float32(.7)."""
    ordered = np.sort(np.asarray(values, np.float32), axis=None)
    return {float(t): int(ordered.size - np.searchsorted(ordered, np.float32(t), side='left'))
            for t in thresholds}


def measure_probability(probability, fill, sam, gy, gx, passed, config):
    assert probability.dtype == np.float32 and probability.shape == fill.shape == sam.shape
    assert np.isfinite(probability).all() and fill.dtype == sam.dtype == bool
    assert not np.any(sam & ~fill)
    thresholds = sorted(set(config['threshold_grid'] + config['inside_thresholds'] + config['outside_thresholds']))
    inside_gt = fill[gy, gx]
    counts = {}
    gt = len(gy)
    inside_total_gt = int(inside_gt.sum())
    if passed:
        predicted_in = above(probability[fill], thresholds)
        predicted_out = above(probability[~fill], thresholds)
        tp_in = above(probability[gy[inside_gt], gx[inside_gt]], thresholds)
        tp_out = above(probability[gy[~inside_gt], gx[~inside_gt]], thresholds)
    else:
        assert not fill.any() and not sam.any()
        predicted_in = predicted_out = tp_in = tp_out = {t: 0 for t in thresholds}
    for candidate in candidates(config):
        ti, to = candidate['inside_threshold'], candidate['outside_threshold']
        inside_tp, outside_tp = tp_in[ti], tp_out[to] if to is not None else 0
        inside_pred, outside_pred = predicted_in[ti], predicted_out[to] if to is not None else 0
        record = pixel_counts(inside_tp + outside_tp, inside_pred + outside_pred, gt)
        record.update(TP_inside_boxes=inside_tp, TP_outside_boxes=outside_tp,
                      FP_inside_boxes=inside_pred - inside_tp, FP_outside_boxes=outside_pred - outside_tp)
        counts[candidate['id']] = record
    for name, mask in [('SAM_baseline', sam), ('box_fill_control', fill)]:
        tp = int(mask[gy, gx].sum())
        predicted = int(mask.sum())
        counts[name] = pixel_counts(tp, predicted, gt) | {
            'TP_inside_boxes': tp, 'TP_outside_boxes': 0, 'FP_inside_boxes': predicted - tp, 'FP_outside_boxes': 0}
    for record in counts.values():
        record.update(pixel_recall=record['TP'] / gt if gt else None,
                      pixel_precision=record['TP'] / record['predicted_pixels'] if record['predicted_pixels'] else None,
                      gt_inside_boxes=inside_total_gt, gt_outside_boxes=gt - inside_total_gt,
                      negative_FP_image=bool(gt == 0 and record['FP'] > 0))
        record['positive_image_with_any_GT_pixel_overlap'] = bool(gt > 0 and record['TP'] > 0)
    return counts


def summarize(records, config):
    names = [r['id'] for r in candidates(config)] + ['SAM_baseline', 'box_fill_control']
    result = {}
    for name in names:
        stats = summarize_segmentation(records, name)
        gt = stats['TP'] + stats['FN']
        pred = stats['TP'] + stats['FP']
        neg = [r[name] for r in records if r[name]['gt_pixels'] == 0]
        stats.update(pixel_recall=stats['TP'] / gt if gt else None,
                     pixel_precision=stats['TP'] / pred if pred else None,
                     negative_FP_images=sum(r['FP'] > 0 for r in neg),
                     negative_FP_pixels_total=sum(r['FP'] for r in neg),
                     positive_images_with_any_GT_pixel_overlap=sum(r[name]['positive_image_with_any_GT_pixel_overlap'] for r in records),
                     TP_inside_boxes=sum(r[name]['TP_inside_boxes'] for r in records),
                     TP_outside_boxes=sum(r[name]['TP_outside_boxes'] for r in records),
                     FP_inside_boxes=sum(r[name]['FP_inside_boxes'] for r in records),
                     FP_outside_boxes=sum(r[name]['FP_outside_boxes'] for r in records))
        result[name] = stats
    baseline = result['SAM_baseline']
    feasible = []
    for candidate in candidates(config):
        entry = result[candidate['id']]
        assert entry['TP'] + entry['FN'] == baseline['TP'] + baseline['FN']
        reference_defined = baseline['TP'] + baseline['FN'] > 0
        entry['meets_SAM_pixel_recall_reference'] = entry['TP'] >= baseline['TP'] if reference_defined else None
        if entry['meets_SAM_pixel_recall_reference']:
            feasible.append(candidate['id'])
    rank = lambda name: (-result[name]['positive_macro_Dice'], result[name]['negative_FP_pixels_per_image'], name)
    return {'metrics': result, 'relative_reference': {'SAM_TP': baseline['TP'],
              'GT_pixels': baseline['TP'] + baseline['FN'], 'SAM_pixel_recall': baseline['pixel_recall'],
              'interpretation': config['relative_condition']},
            'feasible_candidates': feasible, 'feasible_count': len(feasible),
            'diagnostic_best_feasible': min(feasible, key=rank) if feasible else None,
            'no_feasible_candidate': not bool(feasible)}


def reconstructed_mask(probability, fill, passed, candidate):
    if not passed:
        return np.zeros(probability.shape, bool)
    inside = (probability >= np.float32(candidate['inside_threshold'])) & fill
    if candidate['outside_threshold'] is None:
        return inside
    return inside | ((probability >= np.float32(candidate['outside_threshold'])) & ~fill)
