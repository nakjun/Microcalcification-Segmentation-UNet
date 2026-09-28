"""INbreast image-only inventory; no annotation lookup."""
from common import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['inbreast','cbis_test'],required=True);p.add_argument('--inventory-only',action='store_true');a=p.parse_args()
 if a.dataset=='inbreast':
  rows=[]
  for path in sorted(Path('/data/nas/Breast/Public_Mammo/InBreast/INbreast/AllDICOMs').glob('*.dcm')):
   d=pydicom.dcmread(path,stop_before_pixels=True);spacing=d.get('PixelSpacing',d.get('ImagerPixelSpacing'))
   rows.append(dict(image_id=path.stem.split('_')[0],dataset=a.dataset,dicom_path=str(path),dicom_sha256=checksum(path),height=int(d.Rows),width=int(d.Columns),spacing=None if spacing is None else list(map(float,spacing)),study_id=path.stem.split('_')[1]))
  assert len(rows)==410;immutable(ROOT/'inputs/inbreast.json',rows)
 if a.inventory_only:return
 c=context(a.dataset);immutable(ROOT/a.dataset/'protocol.json',c['protocol']);print(c['signature'],len(c['rows']))
if __name__=='__main__':main()
