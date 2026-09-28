#!/usr/bin/env bash
# Download an authorized archive without embedding credentials or redistributing data.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  csaw|inbreast|cbis) dataset="$1" ;;
  *) echo 'Usage: DATA_URL=... DATA_SHA256=... bash workflows/download_data.sh {csaw|inbreast|cbis}'; exit 2 ;;
esac
if [[ -z "${DATA_URL:-}" ]]; then
  echo 'Obtain access / an original dataset archive first:'
  echo 'CSAW-S: https://zenodo.org/records/4030660 (restricted files; request access)'
  echo 'INbreast: use the original DICOM/XML release obtained from the data provider.'
  echo 'CBIS-DDSM: https://www.cancerimagingarchive.net/collection/cbis-ddsm/ (NBIA download workflow)'
  echo 'No direct unrestricted archive URL is assumed. Already downloaded data can use load_data.sh.'
  exit 2
fi
: "${DATA_SHA256:?Supply the expected archive SHA-256 from your trusted acquisition record}"
export DATASET_NAME="$dataset"
# Do not print the URL: it may contain a temporary access credential.
python3 - <<'PY'
import hashlib,os,re,urllib.request
from pathlib import Path
url=os.environ['DATA_URL'];expected=os.environ['DATA_SHA256'].lower()
if not url.startswith('https://'):raise SystemExit('Only HTTPS downloads are supported')
if not re.fullmatch('[0-9a-f]{64}',expected):raise SystemExit('Invalid SHA-256')
p=Path('datasets/downloads')/(os.environ['DATASET_NAME']+'.archive');p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():raise SystemExit('Archive already exists; refusing to overwrite')
tmp=p.with_suffix('.part');h=hashlib.sha256()
try:
 with urllib.request.urlopen(url,timeout=120) as src,tmp.open('xb') as dst:
  while True:
   chunk=src.read(8*1024*1024)
   if not chunk:break
   h.update(chunk);dst.write(chunk)
 if h.hexdigest()!=expected:raise ValueError('Downloaded archive checksum mismatch')
 tmp.rename(p);print('Verified archive:',p)
except Exception:
 # Keep the partial file for inspection; never mark a failed download complete.
 raise SystemExit('Download failed or checksum mismatched; inspect/remove .part before retrying') from None
PY
