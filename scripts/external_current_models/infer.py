"""Resume-safe image-only inference for the current 21 fixed fold checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import fcntl
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import (CONDITIONS, DATASETS, REPO, RUN, SCRIPTS, context, digest, load_module,
                    prediction_dir, prepare, read, require_disk, save_mask, sha, verify, write)
from calcification_gate_sam_common import native_dicom


def native_image(row, dataset):
    """Shared image-only cache. DICOM hash is checked even when the cache exists."""
    assert sha(row['dicom_path']) == row['dicom_sha256']
    folder = RUN/'native_cache'/dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(row['image_id']+'.npy')
    marker = path.with_suffix('.json')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            meta = read(marker)
            assert meta['dicom_sha256'] == row['dicom_sha256'] and meta['preprocessor_sha256'] == sha(SCRIPTS/'calcification_gate_sam_common.py')
            assert meta['cache_sha256'] == sha(path)
        else:
            require_disk(row['height']*row['width']*2)
            image = native_dicom(row['dicom_path'], (row['height'], row['width']))
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.save(stream, image, allow_pickle=False)
            temp.replace(path)
            write(marker, dict(GT_read=False, dicom_sha256=row['dicom_sha256'], cache_sha256=sha(path),
                               preprocessor_sha256=sha(SCRIPTS/'calcification_gate_sam_common.py')))
    image = np.load(path, allow_pickle=False)
    assert image.shape == (row['height'],row['width']) and image.dtype == np.uint8
    return image, sha(path)


def starts(length):
    return sorted(set(range(0,max(length-256,0)+1,192)) | {max(length-256,0)})


def load_predictor(record, device):
    files = list(record['checkpoints'])
    if record['kind'] in ('study','unext'):
        original = load_module('external_current_original_models', SCRIPTS/'external_comparative_inbreast/common.py')
        StudyModel, UNeXt = original.import_models()
        state = torch.load(files[0], map_location='cpu', weights_only=False)
        assert state['epoch'] == 16 and state['config'] == record['config']
        model = (UNeXt if record['kind'] == 'unext' else StudyModel)(record['config'])
        model.load_state_dict(state['model'], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        from calcification_refinement_common import tiled_probability
        def predict(image):
            return tiled_probability(model, image, record['config'], device)[0]
        return predict
    if record['kind'] == 'deepmica':
        module = load_module('external_current_deepmica', REPO/'DeepMiCa/pipeline.py')
        model, meta = module.load_checkpoint(files[0], str(device))
        assert meta['channels'] == 1 and meta['scale'] == 255 and meta['patch_size'] == 256
        def predict(image):
            return module.tiled_predict(model, torch.from_numpy(image.astype(np.float32)/255.)[None], 256, 192, str(device))
        return predict
    sys.path.insert(0, str(REPO/'YOLO-HLSAM'))
    module = load_module('external_current_yolo', REPO/'YOLO-HLSAM/model.py')
    # Serialized JSON key sorting must never decide checkpoint role.
    detector_path = next(p for p in files if '/detector/' in p)
    segmentor_path = next(p for p in files if '/segmentor/' in p)
    sam_path = next(p for p in files if p.endswith('/sam_vit_b.pth'))
    detector = module.Detector(detector_path, str(device), 640, .25, .45, 100)
    segmentor = module.Segmentor('vit_b', sam_path, segmentor_path).to(device).eval()
    def predict(image):
        total, count = np.zeros(image.shape,np.float32), np.zeros(image.shape,np.uint16)
        for y in starts(image.shape[0]):
            for x in starts(image.shape[1]):
                tile = np.repeat(image[y:y+256,x:x+256,None],3,axis=2)
                probability, _, _ = module.predict(tile,detector,segmentor)
                h,w = probability.shape
                total[y:y+h,x:x+w] += probability
                count[y:y+h,x:x+w] += 1
        assert (count > 0).all()
        return total/count
    return predict


def run(dataset, record, signature, protocol, device, smoke=False):
    rows = read(protocol['inventory'][dataset])
    active = rows[:1] if smoke else rows
    output = prediction_dir(dataset,record['condition'],record['fold'],smoke)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists():
            done = verify(output/'complete.json',signature)
            assert done['image_ids'] == [r['image_id'] for r in active] and done['smoke'] == smoke
            return
        predictor = load_predictor(record,device)
        outputs = {}
        for index,row in enumerate(active):
            dest = output/(row['image_id']+'.npz')
            marker = dest.with_suffix('.json')
            if marker.exists():
                done = verify(marker,signature)
                assert done['dicom_sha256'] == row['dicom_sha256'] and done['checkpoints'] == record['checkpoints']
                assert done['GT_read'] is False and done['smoke'] == smoke
            else:
                require_disk(row['height']*row['width']//8 + 4096)
                image, image_sha = native_image(row,dataset)
                start = time.perf_counter()
                with torch.inference_mode():
                    probability = predictor(image)
                assert probability.shape == image.shape and np.isfinite(probability).all()
                assert probability.min() >= 0 and probability.max() <= 1
                save_mask(dest,probability >= np.float32(.5))
                elapsed = time.perf_counter()-start
                write(marker,dict(complete=True,signature=signature,dataset=dataset,image_id=row['image_id'],
                                  condition=record['condition'],fold=record['fold'],checkpoints=record['checkpoints'],
                                  dicom_sha256=row['dicom_sha256'],native_cache_sha256=image_sha,threshold=.5,
                                  native_shape=list(image.shape),tiles=len(starts(image.shape[0]))*len(starts(image.shape[1])),
                                  GT_read=False,smoke=smoke,gate=None,predicted_boxes_only=record['kind']=='yolo',
                                  elapsed_inference_and_write_seconds=elapsed,output_files={str(dest):sha(dest)}))
            outputs[str(dest)] = sha(dest)
            outputs[str(marker)] = sha(marker)
            print(dataset,record['condition'],record['fold'],f'{index+1}/{len(active)}',flush=True)
        write(output/'complete.json',dict(complete=True,signature=signature,dataset=dataset,condition=record['condition'],
                                         fold=record['fold'],checkpoints=record['checkpoints'],image_ids=[r['image_id'] for r in active],
                                         GT_read=False,smoke=smoke,gate=None,threshold=.5,output_files=outputs))
        del predictor
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--cache-only',action='store_true')
    p.add_argument('--dataset',choices=tuple(DATASETS))
    p.add_argument('--condition',choices=CONDITIONS)
    p.add_argument('--fold',type=int,choices=(0,1,2))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.prepare:
        prepare()
        print('Frozen protocol prepared',RUN/'protocol.json')
        return
    protocol, records, signature=context(a.condition,a.fold)
    datasets=[a.dataset] if a.dataset else list(DATASETS)
    if a.cache_only:
        for dataset in datasets:
            rows=read(protocol['inventory'][dataset])
            for index,row in enumerate(rows):
                native_image(row,dataset)
                print('cache',dataset,index+1,len(rows),flush=True)
        return
    assert records
    if a.smoke:
        assert len(records)==1 and len(datasets)==1,'Smoke requires dataset, condition, fold'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
    for record in records:
        for dataset in datasets:
            run(dataset,record,signature,protocol,device,a.smoke)


if __name__=='__main__':
    main()
