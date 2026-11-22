"""Run the frozen external evaluation on the two final FP rerun models."""
import argparse
import fcntl
import os
import tempfile
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

import export_external as external
import export_internal as internal
from common import ROOT, context, digest, disk_guard, read, sha, write
from refit_records import verify_external


OUT = ROOT / 'runs/fp_external_20261122'
SOURCE = ROOT / 'runs/CSAW312_FP_Rerun_20261119'


def reverse_predictions(records, cache, dest, provenance, predict, plan):
    # All generation happens in a private directory. Hard links publish complete
    # files atomically and refuse to replace anything another process wrote.
    evidence = OUT/'shards'/dest.parents[2].name/dest.parent.parent.name/dest.name
    evidence.mkdir(parents=True, exist_ok=True)
    protocol = read(dest/'protocol.json')
    assert protocol['pipeline'] == provenance and not protocol['smoke']
    assert protocol['threshold'] == .5 and protocol['gate'] is None
    assert protocol['inventory_sha256'] == sha(ROOT/f'runs/external_v3/inputs/{dest.parents[2].name}.json')
    signature = digest(protocol)
    published = 0
    with (evidence/'run.lock').open('a') as lock, tempfile.TemporaryDirectory(dir=evidence) as temp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temp = Path(temp)
        mask = temp/'mask.npz'
        marker = temp/'marker.json'

        def prepare(row):
            disk_guard(plan, OUT)
            prob = predict(external.native(row, cache))
            assert prob.shape == (row['height'], row['width']) and external.np.isfinite(prob).all()
            external.pack(mask, prob)
            value = sha(mask)
            write(marker, dict(signature=signature, image_id=row['image_id'], prediction_sha256=value,
                               GT_read=False, source_dicom_sha256=row['dicom_sha256']))
            return value

        reference = records[0]
        h = prepare(reference)
        ref_mask = dest/(reference['image_id']+'.npz')
        ref_marker = ref_mask.with_suffix('.json')
        assert h == sha(ref_mask) and sha(marker) == sha(ref_marker)
        write(evidence/'equivalence.json', dict(image_id=reference['image_id'], mask_sha256=h,
                                               marker_sha256=sha(marker), signature=signature))
        print('Matched original prediction and marker', reference['image_id'], flush=True)
        for idx in reversed(range(len(records))):
            # Leave a gap in front of the forward evaluator. Recheck just before
            # publishing because large mammograms have varying inference times.
            if idx < read(dest/'status.json')['images'] + 8:
                break
            row = records[idx]
            target = dest/(row['image_id']+'.npz')
            target_marker = target.with_suffix('.json')
            if target_marker.exists():
                continue
            prepare(row)
            if idx < read(dest/'status.json')['images'] + 8:
                break
            try:
                os.link(mask, target)
            except FileExistsError:
                continue
            try:
                os.link(marker, target_marker)
            except FileExistsError:
                assert sha(marker) == sha(target_marker)
            published += 1
            write(evidence/'status.json', dict(published=published, last_index=idx))
        write(evidence/'finished.json', dict(published=published, signature=signature))


def evaluate(model, device, fold_index, dataset_name, reverse=False):
    plan, lookup, splits, original = context()
    variant, seed, root = ('LH', 43, SOURCE/'utnet') if model == 'utnet' else (
        'native', 42, SOURCE/'yolo/candidate')
    internal.torch.cuda.set_device(device)
    internal.torch.set_num_threads(4)
    internal.cv2.setNumThreads(1)
    internal.torch.use_deterministic_algorithms(True)
    internal.torch.backends.cudnn.benchmark = False
    internal.torch.backends.cudnn.deterministic = True
    cache = original/'cache/external'
    config = dict(model=model, variant=variant, seed=seed)
    summary = {}
    for fold in range(3) if fold_index is None else [fold_index]:
        folder = root/'evaluation/internal'/f'{model}_{variant}'/f'seed_{seed}'/f'fold_{fold}'/'test'
        marker = read(folder/'complete.json')
        assert marker['complete'] and marker['binding_sha256'] == sha(folder/'binding.json')
        expected = read(folder/'binding.json')['provenance']
        jobs = []
        signatures = {}
        for record in (expected['refit'], expected['detector_refit']):
            if record is None:
                continue
            stage = root/'jobs'/record['job_id']/'complete.json'
            assert sha(stage) == record['complete_sha256']
            job = read(stage)['job']
            assert all(job[k] == v for k, v in config.items()) and job['fold'] == fold
            jobs.append(job)
            signatures[job['job_id']] = record['signature']
        targets = [dataset_name] if dataset_name else ['inbreast', 'cbis']
        external_jobs = [dict(config, dataset=d, fold=fold) for d in targets]
        active = dict(plan, jobs=jobs, configurations=[config], external_jobs=external_jobs)
        with patch.object(internal, 'context', lambda: (active, lookup, splits, root)), \
                patch.object(internal, 'source_signature', lambda j: (signatures[j['job_id']], {})):
            predict, provenance = internal.load_pipeline(model, variant, seed, fold, device)
        assert provenance == expected
        for dataset in targets:
            actual = 'cbis_test' if dataset == 'cbis' else dataset
            size = 284 if dataset == 'cbis' else 410
            records = read(ROOT/f'runs/external_v3/inputs/{actual}.json')
            assert len(records) == size and all((cache/actual/(r['image_id']+'.json')).is_file()
                                                for r in records)
            dest = OUT/'evaluation/external'/actual/f'{model}_{variant}'/f'seed_{seed}'/f'fold_{fold}'
            print(f'{model} fold {fold} {dataset}', flush=True)
            if reverse:
                reverse_predictions(records, cache/actual, dest, provenance, predict, active)
                return
            if not (dest/'complete.json').exists():
                load_native = external.native
                with patch.object(external, 'context', lambda: (active, lookup, splits, OUT)), \
                        patch.object(internal, 'load_pipeline', lambda *args: (predict, provenance)), \
                        patch.object(external, 'native', lambda row, folder: load_native(row, cache/actual)):
                    external.run(SimpleNamespace(**config, fold=fold, dataset=dataset, device=device, limit=None))
            summary[f'{fold}/{actual}'] = verify_external(dest, provenance, size)
        del predict
    suffix = '' if fold_index is None else f'_fold_{fold_index}'
    if dataset_name:
        suffix += f'_{dataset_name}'
    write(OUT/f'{model}{suffix}_complete.json', summary)


def await_yolo_folds():
    summary = {}
    for fold in range(3):
        subpath = Path('yolo_native/seed_42')/f'fold_{fold}'
        binding = SOURCE/'yolo/candidate/evaluation/internal'/subpath/'test/binding.json'
        expected = read(binding)['provenance']
        for dataset, count in [('inbreast', 410), ('cbis_test', 284)]:
            dest = OUT/'evaluation/external'/dataset/subpath
            while not (dest/'complete.json').exists():
                sleep(30)
            summary[f'{fold}/{dataset}'] = verify_external(dest, expected, count)
    write(OUT/'yolo_complete.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['utnet', 'yolo'])
    parser.add_argument('device')
    parser.add_argument('--fold', choices=range(3), type=int)
    parser.add_argument('--dataset', choices=['inbreast', 'cbis'])
    parser.add_argument('--after-yolo-fold1', action='store_true')
    parser.add_argument('--after-utnet', action='store_true')
    parser.add_argument('--reverse-after-fold2', action='store_true')
    parser.add_argument('--aggregate-yolo', action='store_true')
    args = parser.parse_args()
    if args.aggregate_yolo:
        await_yolo_folds()
        raise SystemExit(0)
    if args.after_utnet:
        while not (OUT/'utnet_complete.json').is_file():
            sleep(30)
    if args.after_yolo_fold1:
        while not (OUT/'yolo_fold_1_complete.json').is_file():
            sleep(30)
    if args.reverse_after_fold2:
        assert args.model == 'yolo' and args.fold == 0 and args.dataset is not None
        while not (OUT/'yolo_fold_2_complete.json').is_file():
            sleep(30)
    evaluate(args.model, args.device, args.fold, args.dataset, args.reverse_after_fold2)
