"""Noncumulative bounded export; retains one complete learner plus four inference policies."""
import hashlib,json,shutil,sys,zipfile,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'runs/temporal-credit-export-v2'
assert not DEST.exists();DEST.mkdir()
manifest={'scope':'NEW V13 temporal experiment; no promotion; existing main/backup preserved','resumable':'long_credit_regularized/latest.pt','policies':[],'files':{},'existing_main_commit':'c151310b5c36996aeb1fe192be5eea7cc328e846'}
for cell in json.loads((Path(__file__).with_name('protocol.json')).read_text())['cells']:
 name=cell['name'];src=ROOT/'runs/temporal-credit-v1'/name;out=DEST/name;out.mkdir()
 for f in ['metrics.json','summary.json']+(['latest.pt'] if name=='long_credit_regularized' else ['candidate.pt']):shutil.copy2(src/f,out/f)
 manifest['policies'].append(name+('/latest.pt' if name=='long_credit_regularized' else '/candidate.pt'))
for src in [ROOT/'runs/temporal-credit-evaluation-v1',ROOT/'runs/temporal-credit-sensitivity']:
 shutil.copytree(src,DEST/src.name)
shutil.copy2(ROOT/'runs/temporal-frozen-sensitivity.json',DEST/'frozen-sensitivity.json')
for name in ['temporal-mode-margins.json','temporal-credit-states.json','temporal-credit-states.npz']:
 shutil.copy2(ROOT/'runs'/name,DEST/name)
sys.path.insert(0,str(ROOT/'src'));from policy import Policy
manifest['inference_sha256']={}
torch.set_num_threads(1)
for name in manifest['policies']:
 a=Policy().eval();a.load_state_dict(torch.load(DEST/name,map_location='cpu',weights_only=True)['state_dict'],strict=True)
 with torch.inference_mode():out=a(torch.linspace(-1,1,52*13).reshape(52,13))
 manifest['inference_sha256'][name]=hashlib.sha256(b''.join(t.numpy().tobytes() for t in out)).hexdigest()
for p in DEST.rglob('*'):
 if p.is_file():manifest['files'][str(p.relative_to(DEST))]=hashlib.sha256(p.read_bytes()).hexdigest()
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(DEST)
