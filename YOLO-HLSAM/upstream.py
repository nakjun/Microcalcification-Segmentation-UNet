"""Access pinned official sources; no downloads or implicit checkpoint loading."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/yolo-hlsam/yolo-hlsam'


def activate():
    # Upstream uses absolute imports named models/utils; fail on collisions.
    for name in ('models', 'utils', 'segment_anything_training'):
        module = sys.modules.get(name)
        if module is not None:
            file = getattr(module, '__file__', None)
            paths = [Path(file).resolve()] if file else [Path(p).resolve() for p in getattr(module, '__path__', [])]
            if not paths or any(not p.is_relative_to(SOURCE.resolve()) for p in paths):
                raise RuntimeError(f'Conflicting {name} import; run YOLO-HLSAM in its own process')
    for index, path in enumerate((SOURCE, SOURCE/'train'), start=1):
        if str(path) not in sys.path:
            sys.path.insert(index, str(path))
