# Upstream provenance

This directory vendors the unmodified upstream `archs.py` and `LICENSE` from
the authors' official UNeXt PyTorch repository:

- repository: <https://github.com/jeya-maria-jose/UNeXt-pytorch>
- pinned commit: `6ad0855114a35afbf81decf5dc912cd8de70476a`
- retrieved: 2026-09-15
- `archs.py` SHA-256: `7486a70d6398484da496727f49f7b0d0c14aac0b49bda0cd34b4f877737831ba`
- `LICENSE` SHA-256: `937c2c490f55b11c20af518ba4c094506041125553b7b42f290c495ba07cff74`

The upstream code is MIT licensed; its complete license is retained beside the
source. `../model.py` is a dependency-minimal port of upstream `UNext`, not an
import of this file: upstream imports historical `timm==0.3.2`, `mmcv`,
torchvision utilities, and project-local `utils`, although the UNext forward
path does not use those utilities. The port retains the UNext standard block
order, channel widths, BatchNorm, ReLU, shifted-MLP, bilinear decoder, and
default zero dropout/drop-path. Its only interface adaptations are listed in
`../model.py`.

This is an author-UNeXt family comparator. It is not an exact Pro_UNeXt
implementation: the Pro_UNeXt paper does not provide a source implementation
or enough fused-MBConv/SE details to reproduce that architecture exactly.
