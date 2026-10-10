"""Replay the exact failed development cases; this is diagnosis, not new training."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'training/dagger_corrective'),str(ROOT/'src')]
from campaign import load,GEO,BASE,World
from runtime import decode
FAILED=ROOT/'runs/dagger-corrective-v1/candidate.pt'
def trace(path,side):
 p=load(path);op=load(BASE);w=World(9520011,'M',side);h=None;oh=None;xs=[];rows=[]
 with torch.inference_mode():
  try:
   for n in range(1200):
    x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=op(torch.tensor(ox[None]),oh);mode=int(l.argmax());xs.append(x);old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)));speed=float(np.linalg.norm(w.state.cars[w.our].physics.linear_velocity));rows.append(dict(t=n/60,mode=mode,speed=speed,steer=float(s),distance=float(x[6]*6000),bearing=float(np.arctan2(x[1],x[0])),touch=bool(touch),goal=goal,mode_margin=float(l[0,1]-l[0,0]),car_z=float(w.state.cars[w.our].physics.position[2]),on_ground=bool(w.state.cars[w.our].on_ground),yaw=float(w.state.cars[w.our].physics.euler_angles[1])))
    if goal is not None:w.reset('E',side,w.state.tick_count)
   return np.array(xs),rows
  finally:w.close()
def main():
 torch.set_num_threads(1);out=ROOT/'runs/interception-readout-audit';assert not out.exists();out.mkdir();data=np.load(ROOT/'runs/dagger-corrective-v1/data.npz');targets=np.concatenate([data[f'm{i}'] for i in range(44)]);xall=np.concatenate([data[f'x{i}'] for i in range(44)]);summary=dict(training_target_neutral=int((targets==0).sum()),training_targets=len(targets),neutral_target_speed_min=float((xall[targets==0,7]*2300).min()),neutral_target_speed_max=float((xall[targets==0,7]*2300).max()),pairs=[])
 for side in [0,1]:
  gx,gr=trace(GEO,side);fx,fr=trace(FAILED,side);np.savez_compressed(out/f'failed-side{side}.npz',x=fx);(out/f'failed-side{side}.json').write_text(json.dumps(fr));models={'failed':load(FAILED),'geometric':load(GEO)}
  # Parameter-block diagnostic ablations only; never exported as playing models.
  a=load(FAILED);g=load(GEO)
  with torch.no_grad():a.head.weight[:4]=g.head.weight[:4];a.head.bias[:4]=g.head.bias[:4]
  models['failed_with_original_mode_head']=a;a=load(FAILED);a.encoder.load_state_dict(g.encoder.state_dict());a.gru.load_state_dict(g.gru.state_dict());models['failed_with_original_representation']=a
  stopped=np.array([r['speed']<100 for r in fr]);neutral=np.array([r['mode']==0 for r in fr]);fixed={}
  with torch.inference_mode():
   for name,p in models.items():
    l,s,h=p(torch.tensor(fx));m=l.argmax(-1).numpy();fixed[name]=dict(neutral_states=int((m==0).sum()),neutral_on_stationary=int(((m==0)&stopped).sum()),neutral_on_failed_neutral=int(((m==0)&neutral).sum()))
  summary['pairs'].append(dict(side=side,control_stationary_seconds=sum(r['speed']<100 for r in gr)/60,failed_stationary_seconds=stopped.sum()/60,neutral_first_second=next((r['t'] for r in fr if r['mode']==0),None),stopped_first_second_after_initial=next((r['t'] for r in fr if r['t']>1 and r['speed']<100),None),fixed_prefix_ablations=fixed))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
