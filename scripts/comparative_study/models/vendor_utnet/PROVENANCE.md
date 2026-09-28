# UTNet vendor provenance

* Upstream: https://github.com/yhygao/UTNet
* Pinned commit: `6bdcc97e2364d127e1d2d20e307f935c8fffbff0`
* License: MIT, copied verbatim as `LICENSE`.
* Raw upstream files, copied byte-for-byte into `original/`:
  * `utnet.py` — SHA-256 `9d9ed2cb3e2b91c2376ca3be5247a49502effcde47675d40cc837466393a6535`
  * `unet_utils.py` — SHA-256 `a98b51cbe7e0a888afc001264f7f87cb87f4e3e24c6b01b6efc270d8aa09993d`
  * `conv_trans_utils.py` — SHA-256 `c312e6347a77091dba2cd613f6c27b3e86e7cf299f375c8795eae561849319b0`

`compat/` is a package copy of these three files, retaining their existing
relative imports and all model operations unchanged.  It requires the
upstream dependency `einops`, installed in the locked environment.
