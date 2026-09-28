"""Synthetic CPU checks only; no dataset performance estimates."""
import argparse
import ast
import cv2
import importlib.util
import csv
import gc
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from pipeline import (build_model, load_checkpoint, metadata, read_manifest,
                      save_checkpoint, tiled_predict, topk_loss, train, image_tensor, Patches, ROOT)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    checks = []
    with tempfile.TemporaryDirectory(prefix='deepmica-smoke-') as directory:
        root = Path(directory)
        # Execute only the established pure preprocessing functions as an oracle.
        source = ast.parse((ROOT/'scripts/calcification_refinement_common.py').read_text())
        selected = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ('reflected_indices', 'input_patch')], type_ignores=[])
        oracle = {'np': np, 'cv2': cv2}
        exec(compile(selected, '<existing-input-patch>', 'exec'), oracle)
        config = dict(patch_size=32, contrast_kernel=19, contrast_sigma=3., contrast_gain=16.)
        native = np.random.default_rng(42).integers(0, 256, (57, 73), dtype=np.uint8)
        np.save(root/'native.npy', native)
        features = image_tensor(root/'native.npy', 2, 255.)
        for y, x in ((0, 0), (9, 12), (25, 41)):
            expected_patch = oracle['input_patch'](native, y, x, config)
            np.testing.assert_allclose(features[:, y:y+32, x:x+32].numpy(), expected_patch, atol=1e-6)
            ys = oracle['reflected_indices'](y-9, 50, native.shape[0])
            xs = oracle['reflected_indices'](x-9, 50, native.shape[1])
            np.save(root/'halo.npy', native[np.ix_(ys, xs)])
            np.save(root/'halo_mask.npy', np.zeros((32, 32), np.uint8))
            observed, _ = Patches([{'image': root/'halo.npy', 'mask': root/'halo_mask.npy'}], 2, 255., 32)[0]
            np.testing.assert_allclose(observed.numpy(), expected_patch, atol=1e-6)
        checks.append('paper Gaussian19 sigma3 gain16 reflect101 preprocessing matches existing input_patch, including halo training at native edges')
        spec = importlib.util.spec_from_file_location('official_deepmica_loss', ROOT/'external/DeepMiCa/Step2_Segmentation/_03_Train/Losses.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        logits = torch.randn(2, 1, 8, 8)
        truth = torch.zeros_like(logits)
        truth[0, 0, :2, :2] = 1
        truth[1, 0, :3, :3] = 1
        torch.testing.assert_close(topk_loss(logits, truth), official.topk_CE()(logits, truth))
        checks.append('nonempty-target top-k loss matches official concatenation then global mean pooling')
        for channels in (1, 2):
            model = build_model(channels)
            x = torch.rand(2, channels, 32, 32)
            y = model(x)
            target = torch.zeros_like(y)
            target[0, 0, 8, 8] = 1
            loss = topk_loss(y, target)
            loss.backward()
            assert y.shape == (2, 1, 32, 32)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(topk_loss(y.detach(), torch.zeros_like(target)))
            checkpoint = root / 'checkpoint.pt'
            save_checkpoint(checkpoint, model, metadata(channels, 255., 32), 0)
            model.eval()
            with torch.no_grad():
                expected = model(x)
            del model, y, loss
            gc.collect()
            loaded, meta = load_checkpoint(checkpoint)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), expected)
            prob = tiled_predict(loaded, torch.rand(channels, 19, 41), size=32, stride=16)
            assert prob.shape == (19, 41) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
            checks.append(f'{channels}ch: finite gradients, empty-mask loss, strict checkpoint roundtrip + metadata rejection, native overlap inference')
            del loaded, x, target, expected
            corrupted = torch.load(checkpoint, weights_only=True)
            corrupted['metadata']['source_sha256'] = 'invalid'
            torch.save(corrupted, checkpoint)
            del corrupted
            try:
                load_checkpoint(checkpoint)
            except ValueError as exc:
                assert 'metadata' in str(exc)
            else:
                raise AssertionError('Corrupt metadata accepted')
            gc.collect()
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 1, *x.shape[-2:])
        assert np.all(tiled_predict(Constant(), torch.rand(1, 45, 73), 32, 13) == .5)
        checks.append('overlap/border coverage agrees with constant-probability oracle')
        for index in range(3):
            np.save(root / f'x{index}.npy', np.ones((32, 32), dtype=np.float32)*128)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[10, 10] = 1
            np.save(root / f'y{index}.npy', mask)
        manifest = root / 'manifest.csv'
        rows = [[f'x{i}.npy', f'y{i}.npy', f'p{i}', 'train' if i < 2 else 'val'] for i in range(3)]
        def write_rows():
            with manifest.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image', 'mask', 'patient_id', 'split'])
                writer.writerows(rows)
        write_rows()
        assert len(read_manifest(manifest)) == 3
        rows[2][2] = 'p0'
        write_rows()
        try:
            read_manifest(manifest)
        except ValueError as exc:
            assert 'Patient crosses' in str(exc)
        else:
            raise AssertionError('Patient split leakage accepted')
        rows[2][2] = 'p2'
        write_rows()
        train(argparse.Namespace(command='train', manifest=str(manifest), output=str(root/'run'), channels=1, scale=255., patch_size=32, batch_size=2, epochs=1, lr=.001, seed=42, device='cpu'))
        assert (root/'run/training_complete.json').is_file()
        trained, meta = load_checkpoint(root/'run/best.pt')
        checks.append('patient leakage rejected; one synthetic CPU training epoch + validation + best checkpoint loading')
    result = {'scope': 'synthetic CPU smoke only; no clinical metrics or full training', 'checks': checks, 'torch': str(torch.__version__)}
    (Path(__file__).parent/'smoke_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
