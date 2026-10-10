"""Unused ground geometries; frozen actor diagnostics only."""
from proof import *
p=Policy();p.load_state_dict(torch.load(ROOT/'runs/flip-learning-v1/candidate.pt',weights_only=True)['state_dict']);p.eval();g=load();rows=[];out=ROOT/'runs/flip-fresh-ground-v1';assert not out.exists();out.mkdir()
for name,model in [('geometric',g),('candidate',p)]:
 for rate in [60,120]:
  for side in [0,1]:
   for speed,gap,lat in [(800,1700,-120),(1300,2400,120)]:
    r,*_=trial(rate,side,'nearball','geometric',seed=9890011,model=model,initial_speed=speed,ball_gap=gap,lateral=lat);r['model']=name;r['initial_speed']=speed;r['ball_gap']=gap;r['lateral']=lat;r.pop('trace');rows.append(r)
(out/'records.json').write_text(json.dumps(rows,indent=2));print({name:dict(contacts=sum(r['first_contact'] is not None for r in rows if r['model']==name),flips=sum(len(r['flip_events']) for r in rows if r['model']==name),stationary_seconds=sum(r['stationary_seconds'] for r in rows if r['model']==name)) for name in ['geometric','candidate']})
