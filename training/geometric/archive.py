"""One new supervised learner, compact records and original data; no cumulative ZIPs."""
from pathlib import Path
import json,hashlib,shutil,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));from policy import Policy
out=ROOT/'runs/geometric-retained';assert not out.exists();out.mkdir();src=ROOT/'runs/geometric-supervised-v1'
for n in ['latest.pt','data.npz','data.json','summary.json','history.json','independent-label-check.json']:shutil.copy2(src/n,out/n)
shutil.copytree(ROOT/'runs/geometric-evaluation-v1',out/'evaluation')
p=Policy().eval();p.load_state_dict(torch.load(out/'latest.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);torch.set_num_threads(1)
with torch.inference_mode():outputs=p(torch.tensor(np.load(out/'data.npz')['observations'][0,:256]))
manifest={'kind':'bounded experimental supervised candidate; NOT promoted','main_preserved':'c151310b5c36996aeb1fe192be5eea7cc328e846','candidate_model_sha256':hashlib.sha256((ROOT/'model/V13.pt').read_bytes()).hexdigest(),'resumable':'latest.pt','data':'data.npz','full_optimizer':'supervised Adam, not PPO','files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},'inference_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in outputs)).hexdigest(),'existing_original_PPO_states':'unchanged runs/retained roles and existing verified private backup','promotion':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
