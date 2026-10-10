"""Frozen real-state diagnostics and explicitly counterfactual forecast sensitivity."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch.distributions import Categorical
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'training'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from world import World
from evaluate import load
from runtime import decode
COLS=[3,4,5,9,10,11]

@torch.inference_mode()
def real_states(output):
 assert not output.exists();p=load(ROOT/'model/V13.pt');o=load(ROOT/'runs/retained/baseline/V13.pt');sequences=[];raw=[];metadata=[]
 scenarios=[(stage,side,None,1715011+i*1009) for i,stage in enumerate(['A','C','D','E']) for side in [0,1]]+ [('B',side,index,1716011+index*1009) for index in range(5) for side in [0,1]]
 for stage,side,index,seed in scenarios:
  w=World(seed,stage,side,kickoff_index=index);h=None;oh=None;obs=[];positions=[]
  try:
   for t in range(512):
    x=w.observe(w.our);ox=w.observe(w.enemy);obs.append(x);positions.append(np.r_[w.state.cars[w.our].physics.position,w.state.ball.position])
    l,s,h=p(torch.tensor(x[None,:]),h);ol,os,oh=o(torch.tensor(ox[None,:]),oh)
    old,touch,goal=w.step(decode(int(l.argmax()),float(s)),decode(int(ol.argmax()),float(os)))
    if goal is not None:w.reset('E',side,w.state.tick_count)
   sequences.append(np.asarray(obs));raw.append(np.asarray(positions));metadata.append(dict(stage=stage,team=side,kickoff_index=index,seed=seed,seconds=512/60))
  finally:w.close()
 output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,observations=np.asarray(sequences),real_car_ball_positions=np.asarray(raw))
 output.with_suffix('.json').write_text(json.dumps({'source':'real RocketSim validation states, no action/teacher labels','native_RLBot_recordings_available':False,'states':len(sequences)*512,'sequences':metadata},indent=2)+'\n')

@torch.inference_mode()
def sensitivity(model,data):
 p=load(model);sequences=np.load(data)['observations'];rng=np.random.default_rng(1717011);rows=[]
 for name in ['availability_only','counterfactual_six_features']:
  mode_changes=[];steer_changes=[];joint_kls=[];mode_kls=[]
  for seq in sequences:
   x=torch.tensor(seq);y=x.clone();y[:,9]=1;y[:,10]=1;y[:,11]=0
   if name=='counterfactual_six_features':
    y[:,3:6]=torch.tensor(rng.uniform(-1.,1.,(len(y),3)),dtype=torch.float32);y[:,11]=torch.tensor(rng.uniform(-2.,2.,len(y)),dtype=torch.float32)
   l,s,h=p(x);q,t,h2=p(y);cat=Categorical(logits=l);other=Categorical(logits=q);kl=torch.distributions.kl_divergence(cat,other)
   mu=torch.atanh(s.clamp(-1+1e-6,1-1e-6));nu=torch.atanh(t.clamp(-1+1e-6,1-1e-6));joint=kl+cat.probs[:,1]*(mu-nu).square()/(2*.35**2)
   mode_changes.extend((l.argmax(-1)!=q.argmax(-1)).tolist());steer_changes.extend((s-t).abs().tolist());mode_kls.extend(kl.tolist());joint_kls.extend(joint.tolist())
  rows.append(dict(stress=name,states=len(mode_changes),mode_change_fraction=float(np.mean(mode_changes)),mean_absolute_steer_change=float(np.mean(steer_changes)),mode_KL_mean=float(np.mean(mode_kls)),behavior_joint_KL_mean=float(np.mean(joint_kls)),counterfactual_NOT_native_forecast=True))
 return dict(model=str(model),forecast_column_norm=float(p.encoder[0].weight[:,COLS].square().mean().sqrt()),stress=rows,interpretation='Open-loop sensitivity on identical real-state sequences; fabricated stress inputs are not native recordings, labels, training data or gameplay evaluation.')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--model',type=Path);parser.add_argument('--output',type=Path);a=parser.parse_args();torch.set_num_threads(1)
 if not a.dataset.exists():real_states(a.dataset)
 if a.model:
  result=sensitivity(a.model,a.dataset);assert a.output and not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
