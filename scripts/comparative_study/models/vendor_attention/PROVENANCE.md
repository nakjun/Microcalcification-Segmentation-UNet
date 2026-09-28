# Attention-Gated-Networks `unet_2D` vendor provenance

* Upstream: https://github.com/ozan-oktay/Attention-Gated-Networks
* Pinned commit: `eee4881fdc31920efd873773e0b744df8dacbfb6`
* License: MIT, copied verbatim as `LICENSE`.
* Raw upstream files, copied byte-for-byte into `original/`:
  * `unet_2D.py` — SHA-256 `76026e667358c060ae1909cd629ca0da8cda9796557eb63b1d568b1c85e71f55`
  * `utils.py` — SHA-256 `d6b151725939154d5e2eefe7ebd0e399797788c859185938f3284f1805634947`
  * `networks_other.py` — SHA-256 `9fdcecf29b9767c3853200925566f8d4b447c6e64e2b4aa77ac891d459e299e0`

`compat/` contains the exact same definitions for the required files.  Its
two `from models.networks_other import init_weights` imports were rewritten to
the relative `.networks_other` form so the source can be imported without
installing the upstream repository as a top-level package.  No network
operation, module name, constructor argument, or initialization call changed.
