"""Training-only mirror/recording audit. No optimizer, checkpoint writes or runtime edits."""
from pathlib import Path
import sys,copy,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/outcome_flips'));import world as ow
OUT=ROOT/'runs/mirror-recording-audit-v1'
def mirror_x(x):
 y=np.array(x,copy=True);y[...,1]*=-1;y[...,4]*=-1;return y
def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1)
 M=np.diag([-1.,1.,1.]).astype(np.float32);D=np.diag([1.,-1.,1.]).astype(np.float32)
 physical=[]
 for side in [0,1]:
  for kind in ['kickoff','curve']:
   scene=dict(kind=kind,side=side,index=side,enemy_index=4,seed=10641011+side,gap=1600,speed=600,yaw_error=.4,ball_vx=120,ball_vy=100)
   a=ow.setup(scene);b=ow.setup(scene)
   try:
    state=copy.deepcopy(a.state)
    for p in [state.ball]+[c.physics for c in state.cars.values()]:
     p.position=M@p.position;p.linear_velocity=M@p.linear_velocity;p.angular_velocity=-M@p.angular_velocity;p.rotation_mtx=M@p.rotation_mtx@D
    b.state=b.engine.set_state(state,{})
    maxfeature=maxposition=0.
    for n in range(60):
     for aid in a.ids:
      xa=a.observe(aid);xb=b.observe(aid);maxfeature=max(maxfeature,float(np.max(np.abs(xb-mirror_x(xa)))))
     steer=.3 if n<30 else -.2
     a.step(ow.decode(1,steer),ow.decode(1,-.15));b.step(ow.decode(1,-steer),ow.decode(1,.15))
     for aid in a.ids:maxposition=max(maxposition,float(np.max(np.abs(b.state.cars[aid].physics.position-M@a.state.cars[aid].physics.position))))
    assert maxfeature<1e-4,(scene,maxfeature)
    assert maxposition<.5,(scene,maxposition)
    physical.append(dict(side=side,kind=kind,callbacks=60,max_feature_error=maxfeature,max_car_position_UU_error=maxposition))
   finally:a.close();b.close()
 for mode in range(4):
  for s in [-1.,-.2,0.,.2,1.]:
   a=np.asarray(ow.decode(mode,s),float);b=np.asarray(ow.decode(mode,-s),float);expected=a.copy();expected[[1,3,4]]*=-1;assert np.array_equal(b,expected)
 from rlgym.rocket_league.state_mutators import KickoffMutator
 kickmap=[]
 for pos in [KickoffMutator.SPAWN_BLUE_POS,KickoffMutator.SPAWN_ORANGE_POS]:kickmap.append([int(np.argmin(np.linalg.norm(pos-(M@p),axis=1))) for p in pos])
 datasets={};examples=[]
 for name in ['geometric-retained','outcome-flip-learning-v1','ground-trajectory-learning-v1']:
  path=ROOT/'runs'/name/'data.npz';data=np.load(path,allow_pickle=False)
  if 'observations' in data:
   seq=[(x,m,s) for x,m,s in zip(data['observations'],data['modes'],data['steering'])];roles={'controller_demonstration_or_missing_ball_augmentation':len(seq)}
  else:
   meta=json.loads((path.parent/'data-records.json').read_text());seq=[(data[f'x{i}'],data[f'm{i}'],data[f's{i}']) for i in range(len(meta))];roles={r:sum(v['role']==r for v in meta) for r in {v['role'] for v in meta}}
  for x,m,s in seq:
   assert x.ndim==2 and x.shape[1]==13 and len(x)==len(m)==len(s) and np.isfinite(x).all() and np.isfinite(s).all()
   assert ((m>=0)&(m<4)).all() and (np.abs(s)<=1.000001).all();assert np.array_equal(mirror_x(mirror_x(x)),x)
  datasets[name]=dict(sequences=len(seq),rows=sum(len(x) for x,_,_ in seq),roles=roles,SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),chronological_arrays=True,midsequence_zero_dt=sum(int((x[1:,12]==0).sum()) for x,_,_ in seq))
  examples.extend(seq[:2])
 models={'geometric':ROOT/'model/V13.pt','outcome128':ROOT/'runs/outcome-flip-learning-v1/candidate.pt','ground64':ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt'};parity={}
 with torch.inference_mode():
  for name,path in models.items():
   p=ow.load(path);maxchunk=0.;asym=[];modes=[]
   for x,_,_ in examples:
    x=x[:480];y=mirror_x(x);t=torch.tensor(y);l,s,h=p(t);pieces=[];steers=[];hh=None
    for chunk in t.split(37):
     ll,ss,hh=p(chunk,hh);pieces.append(ll);steers.append(ss)
    maxchunk=max(maxchunk,float((l-torch.cat(pieces)).abs().max()),float((s-torch.cat(steers)).abs().max()),float((h-hh).abs().max()))
    original,os,_=p(torch.tensor(x));asym.extend((s+os).abs().tolist());modes.extend((l.argmax(-1)!=original.argmax(-1)).tolist())
   assert maxchunk<1e-5;parity[name]=dict(mirrored_whole_vs_carried_chunk_max_error=maxchunk,mean_steering_non_equivariance=float(np.mean(asym)),mirror_mode_change_fraction=float(np.mean(modes)),scope='Six old dataset prefixes; diagnostic only, not fresh gameplay improvement.')
 report=dict(physical=physical,decoder_mirror_pass=True,kickoff_index_mapping_blue_orange=kickmap,datasets=datasets,GRU=parity,mirror='Global x reflection, same team/goals: negate existing local-right columns1,4 and steering; unchanged modes/time/masks/otherfeatures. Angular velocity is an axial vector. No arbitrary transform of64Dhidden: rerun mirrored full chronological prefix from h=None at original match start.',new_optimizer_updates=0,checkpoints_written=False,full_match_records_have_chronological_observations=False,native_acceptance='UNTESTED')
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
