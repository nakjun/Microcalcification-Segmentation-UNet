"""Aggregate completion guard; evaluator must not open held GT before this succeeds."""
from common import ROOT,checksum,context,immutable,prediction_dir,verify
def main():
 c=context();ids=[r['image_id'] for r in c['rows']];out={}
 for cp in c['checkpoints']:
  p=prediction_dir(cp)/'predictions/complete.json';m=verify(p,c['signature']);assert m['image_ids']==ids and not m['smoke'] and m['GT_read'] is False and m['checkpoint_sha256']==cp['checkpoint_sha256'];out[str(p)]=checksum(p)
 immutable(ROOT/'predictions/complete.json',dict(complete=True,signature=c['signature'],checkpoints=48,images_per_checkpoint=32,GT_read=False,output_files=out));print('verified 48x32 GT-free prediction cohorts')
if __name__=='__main__':main()
