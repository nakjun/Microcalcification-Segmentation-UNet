"""Wait for each committed dataset, then execute its fixed evaluator once."""
from pathlib import Path
import subprocess,sys,time,fcntl,json
r=Path('runs/external_v3');runtime=r/'runtime';runtime.mkdir(exist_ok=True)
with (runtime/'evaluation_dispatch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pending=['cbis_test','inbreast']
 while pending:
  ready=[ds for ds in pending if (r/ds/'predictions/complete.json').exists()]
  for ds in ready:
   dest=r/ds/'evaluation/complete.json'
   if not dest.exists():
    with (runtime/f'evaluation_{ds}.log').open('a') as log:
     code=subprocess.call([sys.executable,'scripts/external_v3/evaluate.py','--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
    (runtime/f'evaluation_{ds}.exit').write_text(str(code)+'\n')
    if code:raise SystemExit(code)
   print(ds,'evaluation completed',flush=True);pending.remove(ds)
  if pending:time.sleep(30)
 print('Both fixed external evaluations completed',flush=True)
