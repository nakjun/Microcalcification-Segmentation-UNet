"""Image-only fusion policies and exact count-efficient threshold evaluation."""
import numpy as np
from calcification_refinement_common import pixel_counts
from train_calcification_refinement_v2 import component_stats


def policies():
    return [{'id': f'support{s:g}_inside{i:g}_outside{o:g}', 'support': s, 'inside': i, 'outside': o}
            for s in [0., .001, .003, .01, .03, .1]
            for i in [.01, .03, .1, .3] for o in [.1, .3, .5, .7, .9]]


def fused_mask(v2, probability, fill, gate_pass, policy):
    assert v2.shape == probability.shape == fill.shape
    assert v2.dtype == fill.dtype == bool and probability.dtype == np.float32
    if not gate_pass:
        return np.zeros(v2.shape, bool)
    return ((v2 & (probability >= np.float32(policy['support']))) |
            (fill & (probability >= np.float32(policy['inside']))) |
            (~fill & (probability >= np.float32(policy['outside']))))


def evaluate_counts(v2, probability, fill, gate_pass, gy, gx, groups, choices):
    """Sort four disjoint image regions once; never materialize every candidate mask."""
    assert probability.dtype == np.float32 and np.isfinite(probability).all()
    regions = [v2 & fill, v2 & ~fill, ~v2 & fill, ~v2 & ~fill]
    sorted_regions = [np.sort(probability[r]) if gate_pass else np.empty(0, np.float32) for r in regions]
    gt_regions = [r[gy, gx] for r in regions]
    foreground = probability[gy, gx]
    values = {}
    for p in choices:
        thresholds = [min(p['support'], p['inside']), min(p['support'], p['outside']), p['inside'], p['outside']]
        hit = np.zeros(len(gy), bool)
        predicted = []
        for ordered, selected, threshold in zip(sorted_regions, gt_regions, thresholds):
            threshold = np.float32(threshold)
            predicted.append(int(len(ordered) - np.searchsorted(ordered, threshold, side='left')))
            if gate_pass:
                hit[selected] = foreground[selected] >= threshold
        tp = int(hit.sum())
        tin = int(hit[fill[gy, gx]].sum())
        pin = predicted[0] + predicted[2]
        m = pixel_counts(tp, sum(predicted), len(gy))
        m.update(TP_inside_boxes=tin, TP_outside_boxes=tp - tin, FP_inside_boxes=pin - tin,
                 FP_outside_boxes=m['FP'] - pin + tin, components=component_stats(hit, groups))
        values[p['id']] = m
    return values


def constraint_failures(metrics, reference):
    failed = {}
    for arm, ref in reference.items():
        m = metrics[arm]
        assert m['TP'] + m['FN'] == ref['TP'] + ref['FN']
        assert (m['positive_images'], m['negative_images']) == (ref['positive_images'], ref['negative_images'])
        failed[arm] = [name for name, passed in {
            'Dice': m['positive_macro_Dice'] > ref['positive_macro_Dice'],
            'pixel_recall': m['TP'] >= ref['TP'],
            'positive_image_overlap': m['positive_images_with_TP'] >= ref['positive_images_with_TP'],
            'negative_FP_area': m['negative_FP_pixels'] <= ref['negative_FP_pixels'],
            'negative_FP_images': m['negative_images_with_FP'] <= ref['negative_images_with_FP'],
        }.items() if not passed]
    return failed


def select(candidates, reference):
    feasible = [c for c in candidates if not any(constraint_failures(c['metrics'], reference).values())]
    feasible.sort(key=lambda c: (-sum(m['positive_macro_Dice'] for m in c['metrics'].values()),
                                sum(m['negative_FP_pixels'] / m['negative_images'] for m in c['metrics'].values()), c['policy']['id']))
    return feasible[0] if feasible else None, len(feasible)
