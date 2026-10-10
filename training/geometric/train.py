"""One bounded supervised fit, exact actor and independent geometric labels."""
import sys,json,time,hashlib,random
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training'),str(Path(__file__).parent)]
from policy import Policy
from runtime import decode
from world import World
from labels import targets
COLS=[3,4,5,9,10,11]

def generate(output,cfg):
 p=Policy().eval();p.load_state_dict(torch.load(ROOT/'runs/retained/baseline/V13.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);sequences=[];metadata=[]
 scenarios=[('G',i%2,None) for i in range(40)]+[('B',s,k) for k in range(5) for s in [0,1]]+[('M',i%2,None) for i in range(20)]+[('E',i%2,None) for i in range(10)]
 for i,(stage,side,index) in enumerate(scenarios):
  seed=cfg['data_seed']+i*1009;w=World(seed,stage,side,index);oh=None;rows=[];contacts=0
  try:
   for tick in range(cfg['sequence_decisions']):
    x=w.observe(w.our);ox=w.observe(w.enemy);m,s=targets(x);rows.append(x.copy())
    with torch.inference_mode():l,st,oh=p(torch.tensor(ox[None,:]),oh)
    old,touch,goal=w.step(decode(int(m),float(s)),decode(int(l.argmax()),float(st)));contacts+=int(touch)
    if goal is not None:w.reset('E',side,w.state.tick_count)
   sequences.append(np.asarray(rows));metadata.append(dict(stage=stage,team=side,kickoff_index=index,seed=seed,touches=contacts,source='real RocketSim; labels computed from official current-ball columns only'))
  finally:w.close()
 # Honest training-only missing-ball augmentation; no invented native forecasts.
 for i in range(8):
  x=sequences[i].copy();x[:,list(range(7))+[8,9,10,11]]=0;sequences.append(x);metadata.append({'source':'counterfactual unavailable-ball augmentation of real sequence','source_sequence':i})
 data=np.asarray(sequences);m,s=targets(data);np.savez_compressed(output/'data.npz',observations=data,modes=m,steering=s)
 (output/'data.json').write_text(json.dumps(metadata,indent=2)+'\n');return data,m,s

def main(output):
 assert not output.exists();output.mkdir(parents=True);cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['data_seed']);np.random.seed(cfg['data_seed']);random.seed(cfg['data_seed']);start=time.monotonic()
 x,m,s=generate(output,cfg);generated=time.monotonic();actor=Policy();initial=ROOT/'runs/retained/selected/latest.pt';saved=torch.load(initial,map_location='cpu',weights_only=True);actor.load_state_dict(saved['state_dict'],strict=True)
 with torch.no_grad():actor.encoder[0].weight[:,COLS]=0
 optimizer=torch.optim.Adam(actor.parameters(),lr=.001);data=torch.tensor(x);modes=torch.tensor(m);steering=torch.tensor(s);history=[];updates=0;epoch_count=0
 for epoch in range(cfg['epochs']):
  assert datetime.now(timezone.utc).timestamp()<datetime.fromisoformat(cfg['training_stop']).timestamp()-120,'Training deadline reached'
  order=torch.randperm(len(data));stats=[]
  for offset in range(0,len(order),cfg['sequences_per_update']):
   ids=order[offset:offset+cfg['sequences_per_update']];optimizer.zero_grad(set_to_none=True);losses=[]
   for index in ids:
    # Complete sequence from actual trial start: no shuffled timesteps, no hidden resets inside.
    l,st,h=actor(data[index]);ce=nn.functional.cross_entropy(l,modes[index]);mask=modes[index]==1;se=nn.functional.mse_loss(st[mask],steering[index][mask]) if mask.any() else st.sum()*0
    loss=(ce+2*se)/len(ids);loss.backward();losses.append((float(ce.detach()),float(se.detach())))
   actor.encoder[0].weight.grad[:,COLS]=0;norm=nn.utils.clip_grad_norm_(actor.parameters(),1);assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());optimizer.step();updates+=1
   assert torch.count_nonzero(actor.encoder[0].weight[:,COLS]).item()==0
   stats.append({'classification_loss':float(np.mean([v[0] for v in losses])),'steer_mse':float(np.mean([v[1] for v in losses])),'gradient_norm':float(norm)})
  epoch_count+=1;history.append({'epoch':epoch_count,'updates':updates,'classification_loss':float(np.mean([v['classification_loss'] for v in stats])),'steer_mse':float(np.mean([v['steer_mse'] for v in stats])),'gradient_norm':float(np.mean([v['gradient_norm'] for v in stats]))});print('SUPERVISED',history[-1],flush=True)
  state={'state_dict':actor.state_dict(),'supervised_optimizer':optimizer.state_dict(),'configuration':cfg,'completed_epochs':epoch_count,'updates':updates,'initial_ppo_lineage_decisions':saved['decisions'],'new_ppo_decisions':0,'real_simulation_decisions':cfg['real_sequences']*cfg['sequence_decisions'],'supervised_sample_presentations':epoch_count*len(data)*data.shape[1],'torch_rng':torch.get_rng_state(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),'python_rng':random.getstate(),'history':history,'data_sha256':hashlib.sha256((output/'data.npz').read_bytes()).hexdigest(),'initial_checkpoint_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('world.py'),Path(__file__).with_name('labels.py'),Path(__file__).with_name('protocol.json'),ROOT/'training/sim_contract.py',ROOT/'src/policy.py']}}
  tmp=output/'latest.tmp.pt';torch.save(state,tmp);tmp.replace(output/'latest.pt');(output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 torch.save({'state_dict':actor.state_dict()},output/'candidate.pt');result={'epochs':epoch_count,'updates':updates,'real_simulation_decisions':38400,'augmented_samples':3840,'sample_presentations':epoch_count*len(data)*480,'data_generation_seconds':generated-start,'fitting_seconds':time.monotonic()-generated,'elapsed_seconds':time.monotonic()-start,'new_ppo_decisions':0,'initial_ppo_lineage_decisions':saved['decisions'],'configuration':cfg};(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
