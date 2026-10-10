"""Post-fit diagnostics only; no gradients or training."""
import json,hashlib,sys,random
from pathlib import Path
import torch,numpy as np
from campaign import ROOT,GEO,BASE,A,load,batch,diagnostics,aggregate
out=ROOT/'runs/dagger-corrective-v1';torch.set_num_threads(1)
# Expand the diagnostic-only factorial to all five starts and both sides.
tasks=[];names=[]
for name,rate,dt,u in [('unchanged60',60,None,1),('60_dt_half',60,.5,1),('60_double_gru',60,None,2),('unchanged120',120,None,1),('120_dt_one',120,1.,1)]:
 for k in range(5):
  for side in [0,1]:tasks.append((GEO,'B',9560011+k*1009,side,k,rate,20,dt,u));names.append(name)
rows=batch(tasks);(out/'timing-all-five.json').write_text(json.dumps([dict(configuration=n,**r) for n,r in zip(names,rows)],indent=2));timing={n:aggregate([r for r,k in zip(rows,names) if k==n]) for n in set(names)}
# Identical real-state prefixes: alter only delta field, holding all state and sequence updates fixed.
data=np.load(out/'data.npz');p=load(GEO);q=load(out/'candidate.pt');prefix=[]
with torch.inference_mode():
 for key in sorted(k for k in data.files if k.startswith('x')):
  x=torch.tensor(data[key]);one=x.clone();half=x.clone();one[:,12]=1;half[:,12]=.5;a=p(one);b=p(half);c=q(x);d=p(x)
  prefix.append(dict(sequence=key,dt_mode_change=float((a[0].argmax(-1)!=b[0].argmax(-1)).float().mean()),dt_steer_abs_delta=float((a[1]-b[1]).abs().mean()),trained_mode_change=float((c[0].argmax(-1)!=d[0].argmax(-1)).float().mean()),trained_steer_abs_delta=float((c[1]-d[1]).abs().mean())))
controls={}
for name,path in [('original',BASE),('A',A)]:
 rs=diagnostics(path);(out/f'{name}-diagnostics.json').write_text(json.dumps(rs,indent=2));controls[name]=aggregate(rs)
result=dict(timing=timing,fixed_prefix_mean={k:float(np.mean([r[k] for r in prefix])) for k in prefix[0] if k!='sequence'},controls=controls);(out/'analysis.json').write_text(json.dumps(result,indent=2));(out/'fixed-prefix.json').write_text(json.dumps(prefix,indent=2));print(json.dumps(result,indent=2))
# Complete reproducibility metadata, keeping epoch checkpoint unchanged.
s=torch.load(out/'latest.pt',weights_only=True);np.random.seed(9510011);s['numpy_rng']=(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]);s['numpy_rng_note']='Parent RNG seeded9510011; unused after seeding during campaign; workers own seeded generators.';s['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/policy.py',ROOT/'src/runtime.py',ROOT/'training/sim_contract.py',ROOT/'training/geometric/world.py',ROOT/'training/anchored_final/world.py',ROOT/'training/memory_selective/common.py',Path(__file__).with_name('campaign.py')]};torch.save(s,out/'complete-resumable.pt')
