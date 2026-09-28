"""Read-only external annotation audit; GT never determines inference regions."""
import argparse,ast,csv,hashlib,json,plistlib,re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,pydicom
ROOT=Path('runs/external_v3')
IB=Path('/data/nas/Breast/Public_Mammo/InBreast/INbreast')
CB=Path('/data/nas/Breast/Public_Mammo/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)
 if p.exists():assert json.loads(p.read_text())==x,p
 else:t=p.with_suffix(p.suffix+'.tmp');t.write_text(s);t.replace(p)
def input_row(task):
 ident,p,dataset=task;d=pydicom.dcmread(p,stop_before_pixels=True);spacing=d.get('PixelSpacing',d.get('ImagerPixelSpacing'))
 return dict(image_id=ident,dataset=dataset,dicom_path=str(p),dicom_sha256=sha(p),height=int(d.Rows),width=int(d.Columns),spacing=[float(x) for x in spacing] if spacing else None,study_id=ident.split('_')[2] if dataset=='cbis_test' else p.stem.split('_')[1])
def inbreast():
 inputs={p.stem.split('_')[0]:p for p in (IB/'AllDICOMs').glob('*') if p.suffix.lower() in ['.dcm','.dicom']};records=[];sources={str(IB/'README.txt'):sha(IB/'README.txt')};tot=Counter()
 for ident,p in sorted(inputs.items()):
  d=pydicom.dcmread(p,stop_before_pixels=True);h,w=int(d.Rows),int(d.Columns);xp=IB/'AllXML'/f'{ident}.xml';r=dict(image_id=ident,height=h,width=w,status='missing_xml',points=[],contours=[],issues=[])
  if xp.exists():
   sources[str(xp)]=sha(xp);r.update(status='annotated_xml_partial_boundary',xml_path=str(xp),xml_sha256=sources[str(xp)])
   for im in plistlib.loads(xp.read_bytes()).get('Images',[]):
    for i,roi in enumerate(im.get('ROIs',[])):
     if roi.get('Name')!='Calcification':continue
     pts=np.asarray([ast.literal_eval(x) for x in roi.get('Point_px',[])],dtype=np.float64);rid=str(roi.get('IndexInImage',i));typ='point' if len(pts)==1 else 'contour' if len(pts)>=3 else 'degenerate';tot['raw_'+typ]+=1
     valid=pts.ndim==2 and pts.shape[1]==2 and np.isfinite(pts).all() and len(pts)==roi.get('NumberOfPoints') and np.all((pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h))
     pix=np.rint(pts).astype(np.int64) if valid else np.zeros((0,2),np.int64)
     valid=valid and np.all((pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h))
     if not valid or typ=='degenerate':r['issues'].append(dict(roi_id=rid,reason='invalid_coordinates_or_point_count'));tot['excluded']+=1;continue
     if typ=='point':r['points'].append(dict(roi_id=rid,x=int(pix[0,0]),y=int(pix[0,1]),original_xy=pts[0].tolist()))
     else:
      if abs(cv2.contourArea(pts.astype(np.float32)))==0:r['issues'].append(dict(roi_id=rid,reason='zero_area_contour'));tot['excluded']+=1;continue
      r['contours'].append(dict(roi_id=rid,points=pts.tolist(),raster_xy=pix.tolist()))
   tot['xml_images']+=1
  else:tot['missing_xml']+=1
  tot['valid_points']+=len(r['points']);tot['valid_contours']+=len(r['contours']);records.append(r)
 write(ROOT/'annotations/inbreast.json',dict(complete=True,dataset='inbreast',images=len(records),records=records,summary=dict(tot),coordinate_rule='original DICOM orientation;np.rint ties-to-even;no clipping',scope='points are not boundary GT;contour union is partial-annotation agreement;missing XML is unknown',source_files=sources));print('INbreast',dict(tot),flush=True)
def cbis():
 metadata=list(csv.DictReader(open(CB/'metadata.csv')));full=[r for r in metadata if r['Subject ID'].startswith('Calc-Test_') and r['Series Description']=='full mammogram images'];tasks=[]
 for r in full:
  p=CB/r['File Location'].removeprefix('./');files=sorted(p.glob('*.dcm'));assert len(files)==1,(p,len(files));tasks.append((r['Subject ID'],files[0],'cbis_test'))
 assert len({x[0] for x in tasks})==len(tasks)==284
 with ThreadPoolExecutor(max_workers=4) as pool:inputs=sorted(pool.map(input_row,tasks),key=lambda r:r['image_id'])
 # Membership comes exclusively from official split and full-mammogram metadata, before any ROI is decoded.
 write(ROOT/'inputs/cbis_test.json',inputs);lookup={r['image_id']:r for r in inputs};records={i:dict(image_id=i,height=r['height'],width=r['width'],status='ROI_annotation_pending',rois=[],issues=[]) for i,r in lookup.items()};sources={str(CB/'metadata.csv'):sha(CB/'metadata.csv')};tot=Counter()
 for r in metadata:
  if not r['Subject ID'].startswith('Calc-Test_') or r['Series Description']!='ROI mask images':continue
  rid=r['Subject ID'];ident=rid.rsplit('_',1)[0];tot['ROI_series']+=1
  if ident not in records:raise AssertionError(('ROI without official full image',rid))
  rec=records[ident];shape=(rec['height'],rec['width']);p=CB/r['File Location'].removeprefix('./');candidates=[]
  for f in sorted(p.glob('*.dcm')):
   d=pydicom.dcmread(f,stop_before_pixels=True)
   if (int(d.Rows),int(d.Columns))!=shape:continue
   a=pydicom.dcmread(f).pixel_array;u=np.unique(a)
   if len(u)<=2 and u.min()==0 and u.max()>0:candidates.append((f,a>0))
  if len(candidates)!=1:rec['issues'].append(dict(roi_id=rid,reason='ambiguous_or_no_full_size_binary_mask',candidate_count=len(candidates)));tot['excluded_ROI_series']+=1;continue
  f,mask=candidates[0];sources[str(f)]=sha(f);dest=ROOT/'annotations/cbis_masks'/f'{rid}.npz';dest.parent.mkdir(parents=True,exist_ok=True);flat=np.flatnonzero(mask).astype(np.int64)
  if dest.exists():
   with np.load(dest) as z:np.testing.assert_array_equal(z['flat'],flat);assert tuple(z['shape'])==shape
  else:
   with dest.open('wb') as handle:np.savez_compressed(handle,flat=flat,shape=np.asarray(shape))
  rec['rois'].append(dict(roi_id=rid,mask_dicom_path=str(f),mask_dicom_sha256=sources[str(f)],mask_path=str(dest),mask_sha256=sha(dest),pixels=int(len(flat))));tot['valid_ROIs']+=1
  print('CBIS ROI',tot['ROI_series'],flush=True)
 for rec in records.values():rec['status']='ROI_localization_only' if rec['rois'] else 'no_valid_ROI_unknown'
 write(ROOT/'annotations/cbis_test.json',dict(complete=True,dataset='cbis_test',images=len(records),records=list(records.values()),summary=dict(tot),scope='official Calc-Test full-image cohort;ROI localization,not individual microcalcification segmentation;no confirmed negatives',source_files=sources));print('CBIS',dict(tot),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['inbreast','cbis_test','all'],default='all');a=p.parse_args();cv2.setNumThreads(1)
 if a.dataset in ['all','inbreast']:inbreast()
 if a.dataset in ['all','cbis_test']:cbis()
