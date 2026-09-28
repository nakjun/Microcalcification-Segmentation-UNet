"""Image-only connected-component rescue of the frozen v2 mask."""
import cv2
import numpy as np
from calcification_refinement_v4_common import evaluate_counts as pixel_evaluate, fused_mask
from calcification_refinement_v5_common import failures as v5_failures
from calcification_refinement_common import pixel_counts
from train_calcification_refinement_v2 import component_stats


def policies():
    # Fixed before OOF execution. cap=0 is the matched no-rescue ablation.
    rescue = [(0, 0.)] + [(cap, peak) for cap in [32, 128, 512, 2048] for peak in [0., .003]]
    return [dict(id=f'cap{cap}_peak{peak:g}_support{s:g}_core{core:g}', cap=cap, peak=peak,
                 support=s, inside=core, outside=core)
            for cap, peak in rescue for s in [.001, .003, .01] for core in [.3, .5, .7]]


def component_maps(mask, probability):
    """8-connected predicted components; no ground truth enters this function."""
    assert mask.dtype == bool and probability.dtype == np.float32
    assert mask.shape == probability.shape and np.isfinite(probability).all()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    peaks = np.zeros(len(stats), np.float32)
    np.maximum.at(peaks, labels[mask], probability[mask])
    areas[0] = np.iinfo(np.int64).max
    peaks[0] = -1
    return labels, areas, peaks


def rescue_mask(maps, policy):
    labels, areas, peaks = maps
    keep = (areas <= policy['cap']) & (peaks >= np.float32(policy['peak']))
    keep[0] = False
    return keep[labels]


def predict(v2, probability, fill, passed, policy, maps=None):
    maps = maps if maps is not None else component_maps(v2, probability)
    return (fused_mask(v2, probability, fill, passed, policy) | rescue_mask(maps, policy)) & bool(passed)


def evaluate(v2, probability, fill, passed, gy, gx, groups, choices):
    # Sort the full image once. Rescue adds only pixels from compact v2 components.
    maps = component_maps(v2, probability)
    base_choices = list({(p['support'], p['inside'], p['outside']): p for p in choices}.values())
    base_raw = pixel_evaluate(v2, probability, fill, passed, gy, gx, groups, base_choices)
    bases = {(p['support'], p['inside'], p['outside']): base_raw[p['id']] for p in base_choices}
    pg, vg, fg = probability[gy, gx], v2[gy, gx], fill[gy, gx]
    result = {}
    for cap, peak in sorted({(p['cap'], p['peak']) for p in choices}):
        selected = [p for p in choices if (p['cap'], p['peak']) == (cap, peak)]
        rescue = rescue_mask(maps, selected[0])
        ordered_in = np.sort(probability[rescue & fill])
        ordered_out = np.sort(probability[rescue & ~fill])
        rg = rescue[gy, gx]
        for p in selected:
            base = bases[(p['support'], p['inside'], p['outside'])]
            if not passed:
                result[p['id']] = base.copy()
                continue
            # A rescued pixel already in v2 is absent from the base iff p3 is
            # strictly below both the support and its spatial core threshold.
            tin = np.float32(min(p['support'], p['inside']))
            tout = np.float32(min(p['support'], p['outside']))
            add_in = int(np.searchsorted(ordered_in, tin, side='left'))
            add_out = int(np.searchsorted(ordered_out, tout, side='left'))
            hits = ((vg & (pg >= np.float32(p['support']))) |
                    (fg & (pg >= np.float32(p['inside']))) |
                    (~fg & (pg >= np.float32(p['outside']))) | rg)
            tp = int(hits.sum()); tp_in = int(hits[fg].sum())
            predicted = base['predicted_pixels'] + add_in + add_out
            predicted_in = base['TP_inside_boxes'] + base['FP_inside_boxes'] + add_in
            m = pixel_counts(tp, predicted, len(gy))
            m.update(TP_inside_boxes=tp_in, TP_outside_boxes=tp-tp_in,
                     FP_inside_boxes=predicted_in-tp_in,
                     FP_outside_boxes=m['FP']-predicted_in+tp_in,
                     components=component_stats(hits, groups))
            result[p['id']] = m
    return result


def failures(metrics, v2, v4b):
    result = v5_failures(metrics, v2, v4b)
    for arm, m in metrics.items():
        if m['positive_macro_Dice'] <= v4b[arm]['positive_macro_Dice']:
            result[arm].append('Dice_not_above_v4b')
    return result


def select(candidates, v2, v4b):
    eligible = []
    for c in candidates:
        c['constraint_failures'] = failures(c['metrics'], v2, v4b)
        if not any(c['constraint_failures'].values()):
            eligible.append(c)
    eligible.sort(key=lambda c: (-sum(m['positive_macro_Dice'] for m in c['metrics'].values()),
                                 sum(m['negative_FP_pixels'] for m in c['metrics'].values()), c['policy']['id']))
    return (eligible[0] if eligible else None), len(eligible)
