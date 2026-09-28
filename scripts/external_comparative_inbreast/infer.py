"""GT-free, non-ensemble full-native inference for every comparative checkpoint."""
from __future__ import annotations

import argparse
import fcntl
import gc
import time

import cv2
import numpy as np
import pydicom
import torch

from common import (GATE_ROOT, ROOT, atomic_npz, checksum, context, import_models, immutable,
                    prediction_dir, verify)

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from calcification_gate_sam_common import gate_logits, load_gate, native_dicom
from calcification_refinement_common import tiled_probability
from prepare_retinanet_roi import breast_roi


def gate_dicom_roi(path, shape):
    """The unchanged external_v3 image-only DICOM gate preprocessing."""
    data = pydicom.dcmread(path)
    raw = data.pixel_array
    assert raw.shape == tuple(shape)
    image = raw[5:-5, 5:-5].astype(np.float32)
    assert data.PhotometricInterpretation in ('MONOCHROME1', 'MONOCHROME2')
    if data.PhotometricInterpretation == 'MONOCHROME1':
        image = image.max() - image
    image -= image.min()
    assert image.max() > 0
    image = (image / image.max() * 255).astype(np.uint8)
    x0, y0, x1, y1 = breast_roi(image)
    assert x1 > x0 and y1 > y0
    padded = np.zeros((1536, 928), np.uint8)
    padded[:1520, :912] = cv2.resize(image[y0:y1, x0:x1], (912, 1520), interpolation=cv2.INTER_AREA)
    return padded, [x0 + 5, y0 + 5, x1 + 5, y1 + 5]


def primary_mask(probability, gate_pass):
    return bool(gate_pass) & (probability >= np.float32(.9))


def ungated_mask(probability):
    return probability >= np.float32(.5)


def _load_model(checkpoint, device):
    StudyModel, UNeXtAuthor2ch = import_models()
    state = torch.load(checkpoint['checkpoint'], map_location='cpu', weights_only=False)
    assert state['epoch'] == 16
    model = UNeXtAuthor2ch(checkpoint['model_config']) if checkpoint['kind'] == 'unext' else StudyModel(checkpoint['model_config'])
    model.load_state_dict(state['model'], strict=True)
    return model.requires_grad_(False).eval().to(device)


def run(checkpoint, rows, signature, gate_threshold, device, smoke=False):
    output = prediction_dir(checkpoint) / ('smoke' if smoke else 'predictions')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
     fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
     gate_encoder, gate, gate_config = load_gate(GATE_ROOT / 'classifier.pt', device)
     model = _load_model(checkpoint, device)
     active_rows = rows[:1] if smoke else rows
     outputs, start = {}, time.time()
     for index, row in enumerate(active_rows):
      assert checksum(row['dicom_path']) == row['dicom_sha256']
      path = output / f"{row['image_id']}.npz"
      marker = path.with_suffix('.json')
      if marker.exists():
       existing = verify(marker, signature)
       assert existing['checkpoint_sha256'] == checkpoint['checkpoint_sha256']
       assert existing['dicom_sha256'] == row['dicom_sha256'] and existing['smoke'] == smoke
      else:
       shape = (row['height'], row['width'])
       padded, roi = gate_dicom_roi(row['dicom_path'], shape)
       logit = float(gate_logits(gate_encoder, gate, torch.from_numpy(padded[None, None]), gate_config, device)[0])
       gate_pass = bool(logit >= gate_threshold)
       # Refinement deliberately runs regardless of gate pass; labels and XML are not read here.
       native = native_dicom(row['dicom_path'], shape)
       probability, tiles = tiled_probability(model, native, checkpoint['model_config'], device)
       assert probability.dtype == np.float32 and probability.shape == shape and np.isfinite(probability).all()
       assert ((probability >= 0) & (probability <= 1)).all()
       flat = np.flatnonzero(probability.ravel() >= np.float32(.3)).astype(np.int64)
       atomic_npz(path, shape=np.asarray(shape), flat=flat, probability=probability.ravel()[flat],
                  gate_logit=np.asarray(logit), gate_pass=np.asarray(gate_pass))
       immutable(marker, dict(complete=True, signature=signature, model=checkpoint['model'], variant=checkpoint['variant'],
                              fold=checkpoint['fold'], checkpoint_sha256=checkpoint['checkpoint_sha256'],
                              checkpoint_complete_sha256=checkpoint['complete_sha256'], image_id=row['image_id'],
                              dicom_sha256=row['dicom_sha256'], GT_read=False, smoke=smoke,
                              refinement_executed=True, gate_threshold_logit=gate_threshold, gate_roi=roi, tiles=tiles,
                              output_files={str(path): checksum(path)}))
      outputs[str(path)] = checksum(path)
      outputs[str(marker)] = checksum(marker)
      print(f"{checkpoint['model']}/{checkpoint['variant']}/fold{checkpoint['fold']} {index + 1}/{len(active_rows)} elapsed={time.time() - start:.1f}s", flush=True)
     immutable(output / 'complete.json', dict(complete=True, signature=signature, model=checkpoint['model'], variant=checkpoint['variant'],
                                               fold=checkpoint['fold'], checkpoint_sha256=checkpoint['checkpoint_sha256'],
                                               checkpoint_complete_sha256=checkpoint['complete_sha256'],
                                               image_ids=[row['image_id'] for row in active_rows], GT_read=False, smoke=smoke,
                                               refinement_executed_all_images=True, output_files=outputs))
    del model, gate_encoder, gate
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model', choices=('proposed', 'unext', 'unet2d', 'utnet'))
    parser.add_argument('--variant', choices=('C', 'L', 'H', 'LH'))
    parser.add_argument('--fold', type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    c = context()
    selected = [q for q in c['checkpoints'] if (args.model is None or q['model'] == args.model)
                and (args.variant is None or q['variant'] == args.variant) and (args.fold is None or q['fold'] == args.fold)]
    if args.smoke:
        assert len(selected) == 1, 'smoke requires exactly one --model --variant --fold checkpoint'
    else:
        assert selected, 'no checkpoint matched the requested disjoint subset'
    assert torch.cuda.is_available(), 'CUDA is required for the fixed external full-native inference runtime'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    torch.cuda.set_per_process_memory_fraction(.45, args.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f'cuda:{args.gpu}')
    ROOT.mkdir(parents=True, exist_ok=True)
    for checkpoint in selected:
        run(checkpoint, c['rows'], c['signature'], float(c['protocol']['gate']['threshold_logit']), device, args.smoke)


if __name__ == '__main__':
    main()
