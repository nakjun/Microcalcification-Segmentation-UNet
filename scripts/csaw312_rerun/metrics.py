"""Native-image binary counts and single-class, no-crowd detection AP.

No checkpoint or threshold selection occurs here. Detection inputs are the
caller's already postprocessed native boxes; this module does not silently NMS,
truncate, resize, or filter them. Boxes use floating half-open xyxy coordinates.
"""
from __future__ import annotations

import numpy as np


def segmentation_counts(prob, mask, threshold=0.5):
    prob, mask = np.asarray(prob), np.asarray(mask)
    if prob.ndim != 2 or prob.shape != mask.shape or not prob.size:
        raise ValueError("Require nonempty, identically shaped 2D probability and mask arrays")
    if not np.isfinite(prob).all() or not np.isfinite(mask).all():
        raise ValueError("Nonfinite segmentation input")
    if np.any(prob < 0) or np.any(prob > 1) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probability and threshold must be within [0, 1]")
    truth, predicted = mask > 0, prob >= threshold
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(prob.size - tp - fp - fn)
    positive = bool(truth.any())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, positive=positive,
                dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                negative_fp=fp if not positive else 0,
                gt_pixels=tp + fn, predicted_pixels=tp + fp)


def summarize_segmentation(records):
    records = list(records)
    positives = [row for row in records if row["positive"]]
    negatives = [row for row in records if not row["positive"]]
    totals = {key: sum(int(row[key]) for row in records) for key in ("TP", "FP", "FN", "TN")}
    for row in records:
        if any(row[key] < 0 or int(row[key]) != row[key] for key in totals):
            raise ValueError("Invalid segmentation counts")
        if bool(row["positive"]) != (row["TP"] + row["FN"] > 0):
            raise ValueError("Positive flag disagrees with GT counts")
    negative_fp = sum(int(row["FP"]) for row in negatives)
    return dict(images=len(records), positive_images=len(positives), negative_images=len(negatives),
                positive_macro_dice=float(np.mean([row["dice"] for row in positives])) if positives else None,
                positive_macro_iou=float(np.mean([row["iou"] for row in positives])) if positives else None,
                pixel_recall=totals["TP"] / (totals["TP"] + totals["FN"]) if totals["TP"] + totals["FN"] else None,
                negative_fp_per_image=negative_fp / len(negatives) if negatives else None,
                negative_fp_total=negative_fp,
                negative_fp_images=sum(row["FP"] > 0 for row in negatives),
                negative_fp_image_fraction=sum(row["FP"] > 0 for row in negatives) / len(negatives) if negatives else None,
                gt_pixels=totals["TP"] + totals["FN"], predicted_pixels=totals["TP"] + totals["FP"], **totals)


def _boxes(value):
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Boxes must be a finite Nx4 xyxy array")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("Half-open boxes must have positive width and height")
    return boxes


def _iou(box, truths):
    if not len(truths):
        return np.empty(0, dtype=np.float64)
    extent = np.maximum(np.minimum(box[2:], truths[:, 2:]) - np.maximum(box[:2], truths[:, :2]), 0)
    intersection = extent[:, 0] * extent[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    gt_area = (truths[:, 2] - truths[:, 0]) * (truths[:, 3] - truths[:, 1])
    return intersection / (area + gt_area - intersection)


def detection_ap(records):
    """COCO-style 101-recall-point AP, IoU .50:.05:.95, single class/no crowd.

    Matching is greedy within each image in descending detection-score order,
    allowing one detection per GT. Tied scores use image_id then original box
    index; equal eligible IoUs use the first GT index. GT-empty images still add
    their false positives to dataset-pooled precision. No-GT cohort AP is None.
    """
    ground_truth, predictions = {}, []
    for record in records:
        ident = str(record["image_id"])
        if ident in ground_truth:
            raise ValueError(f"Duplicate detection image_id: {ident}")
        boxes, truths = _boxes(record["boxes"]), _boxes(record["gt_boxes"])
        scores = np.asarray(record["scores"], dtype=np.float64)
        if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Detection scores must be one finite [0, 1] value per box")
        ground_truth[ident] = truths
        predictions.extend((float(score), ident, index, box) for index, (score, box) in enumerate(zip(scores, boxes)))
    predictions.sort(key=lambda row: (-row[0], row[1], row[2]))
    total_gt = sum(len(truths) for truths in ground_truth.values())
    overlaps = [_iou(box, ground_truth[ident]) for _, ident, _, box in predictions]
    ap_values, details = {}, {}
    for percent in range(50, 100, 5):
        threshold = percent / 100.0
        matched = {ident: np.zeros(len(truths), dtype=bool) for ident, truths in ground_truth.items()}
        hits = np.zeros(len(predictions), dtype=np.int64)
        for index, (_, ident, _, _) in enumerate(predictions):
            eligible = np.flatnonzero((overlaps[index] >= threshold) & ~matched[ident])
            if len(eligible):
                chosen = int(eligible[np.argmax(overlaps[index][eligible])])
                matched[ident][chosen] = True
                hits[index] = 1
        tp = np.cumsum(hits)
        fp = np.cumsum(1 - hits)
        key = f"{threshold:.2f}"
        if total_gt:
            recall = tp / total_gt
            precision = tp / np.maximum(tp + fp, 1)
            if len(precision):
                precision = np.maximum.accumulate(precision[::-1])[::-1]
            indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
            interpolated = np.zeros(101, dtype=np.float64)
            available = indices < len(precision)
            interpolated[available] = precision[indices[available]]
            ap_values[key] = float(interpolated.mean())
        else:
            ap_values[key] = None
        matched_count = int(hits.sum())
        details[key] = dict(AP=ap_values[key], TP=matched_count,
                            FP=len(predictions) - matched_count, FN=total_gt - matched_count)
    return dict(AP50_95=float(np.mean(list(ap_values.values()))) if total_gt else None,
                AP50=ap_values["0.50"], AP_per_IoU=ap_values, per_iou=details,
                images=len(ground_truth), gt_boxes=total_gt, predicted_boxes=len(predictions))
