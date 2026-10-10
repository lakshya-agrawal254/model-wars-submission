"""Bounded equal-budget factorial. No mutation of original PPO or playing checkpoint."""
import argparse,hashlib,json,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.distributions import Categorical
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import ppo
from world import World
ppo.World=World
COLS=[3,4,5,9,10,11]
PROTOCOL=Path(__file__).with_name('protocol.json')

def tensors(rows,gamma,lam):
 x=torch.tensor(np.asarray([r[0] for r in rows]));m=torch.tensor([r[1] for r in rows]);z=torch.tensor([r[2] for r in rows]);old=torch.tensor([r[3] for r in rows]);values=np.asarray([r[4] for r in rows]);rewards=np.asarray([r[5] for r in rows]);adv=np.zeros(len(rows),np.float32);running=0
 for i in range(len(rows)-1,-1,-1):
  next_value=values[i+1] if i+1<len(rows) else rows[-1][6]
  running=rewards[i]+gamma*next_value-values[i]+gamma*lam*running;adv[i]=running
 return x,m,z,old,torch.tensor(adv),torch.tensor(adv+values,dtype=torch.float32)

def penalty(actor):return actor.encoder[0].weight[:,COLS].square().mean()

def update(actor,critic,ao,co,batches,cfg):
 data=[tensors(rows,cfg['gamma'],cfg['gae_lambda']) for rows in batches];n=sum(len(d[0]) for d in data);advantages=torch.cat([d[4] for d in data]);mean=advantages.mean();std=advantages.std().clamp_min(1e-5);metrics=[];maximum_error=0.
 with torch.no_grad():
  for x,m,z,old,a,ret in data:
   l,s,h=actor(x);new=ppo.log_probability(l,s,m,z);maximum_error=max(maximum_error,float((new-old).abs().max()));torch.testing.assert_close(new,old,rtol=1e-4,atol=1e-4)
 for epoch in range(4):
  ao.zero_grad(set_to_none=True);co.zero_grad(set_to_none=True);stats=[]
  for x,m,z,old,a,ret in data:
   l,s,h=actor(x);lp=ppo.log_probability(l,s,m,z);ratio=(lp-old).exp();adv=(a-mean)/std
   loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean();entropy=Categorical(logits=l).entropy().mean();weight=len(x)/n
   ((loss-.025*entropy)*weight).backward()
   vl=.5*(critic(x)-ret).square().mean();(vl*weight).backward()
   stats.append((weight,float(loss.detach()),float(vl.detach()),float(entropy.detach()),float(((ratio-1)-(lp-old)).mean().detach()),float(((ratio-1).abs()>.2).float().mean())))
  regularizer=cfg['forecast_regularization']*penalty(actor)
  if cfg['forecast_regularization']:regularizer.backward()
  norm=nn.utils.clip_grad_norm_(actor.parameters(),1.);vnorm=nn.utils.clip_grad_norm_(critic.parameters(),1.)
  assert torch.isfinite(norm) and torch.isfinite(vnorm)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
  ao.step();co.step()
  names=['policy_loss','value_loss','entropy','approx_kl','clip_fraction'];d={k:sum(s[0]*s[i+1] for s in stats) for i,k in enumerate(names)}
  d.update(gradient_norm=float(norm),critic_gradient_norm=float(vnorm),regularization_loss=float(regularizer.detach()),forecast_column_norm=float(penalty(actor).sqrt().detach()),old_logp_max_error=maximum_error);metrics.append(d)
  if d['approx_kl']>.03:break
 return {k:float(np.mean([d[k] for d in metrics])) for k in metrics[0]}|{'epochs':len(metrics)}

def run(cfg,output,decisions=16384,resume=None):
 assert decisions%4096==0 and not output.exists();output.mkdir(parents=True)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 initial=resume or ROOT/'runs/retained/selected/latest.pt';saved=torch.load(initial,map_location='cpu',weights_only=True)
 actor=ppo.Policy();critic=ppo.Critic();opponent=ppo.Policy().eval()
 actor.load_state_dict(saved['state_dict'],strict=True);critic.load_state_dict(saved['critic'],strict=True)
 opponent.load_state_dict(torch.load(ROOT/'runs/retained/baseline/V13.pt',weights_only=True)['state_dict'],strict=True)
 ao=torch.optim.Adam(actor.parameters(),lr=3e-4);co=torch.optim.Adam(critic.parameters(),lr=1e-3);ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer'])
 torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);rng=saved['numpy_rng'];np.random.set_state((rng[0],np.array(rng[1],np.uint32),*rng[2:]))
 count=saved['decisions'];segment=saved['episode'];target=count+decisions;history=[];start=time.monotonic();initial_parameters=torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
 while count<target:
  if datetime.now(timezone.utc).timestamp()>=datetime.fromisoformat('2026-10-11T05:30:00+05:30').timestamp()-120:
   print('DEADLINE GUARD: preserve last completed update',flush=True);break
  actor.eval();batches=[];summaries=[];collected=0
  while collected<4096:
   steps=min(cfg['steps'],4096-collected);rows,summary=ppo.collect(actor,critic,opponent,3101011+segment*1009,'A',segment%2,steps)
   batches.append(rows);summaries.append(summary);collected+=steps;segment+=1;count+=steps
  actor.train();stats=update(actor,critic,ao,co,batches,cfg)
  history.append(dict(decisions=count,new_decisions=count-saved['decisions'],episodes=summaries,ppo=stats,elapsed_seconds=time.monotonic()-start))
  npstate=np.random.get_state();payload=dict(state_dict=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),decisions=count,episode=segment,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(npstate[0],npstate[1].tolist(),*npstate[2:]),history=saved['history']+history,experiment_history=history,configuration=cfg|{'stage':'A','update_decisions':4096,'weighted_gradient_accumulation':True,'optimizer_reset':False,'initial_checkpoint_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'prediction':'original 13D, explicitly unavailable; no synthetic forecast training','actor_parameters':26181,'time_limits_bootstrapped':True,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('world.py'),PROTOCOL,ROOT/'training/ppo.py',ROOT/'training/sim_contract.py']}})
  temp=output/'latest.tmp.pt';torch.save(payload,temp);temp.replace(output/'latest.pt');(output/'metrics.json').write_text(json.dumps(history,indent=2)+'\n')
  print('TEMPORAL',cfg['name'],count,stats,'touches',sum(e['touches'] for e in summaries),flush=True)
 final_parameters=torch.cat([p.detach().flatten() for p in actor.parameters()]);torch.save({'state_dict':actor.state_dict()},output/'candidate.pt')
 result=dict(name=cfg['name'],new_decisions=count-saved['decisions'],requested_decisions=decisions,complete_budget=count==target,lineage_decisions=count,elapsed_seconds=time.monotonic()-start,parameter_l2_change=float((final_parameters-initial_parameters).norm()),configuration=cfg)
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cell',required=True);parser.add_argument('--decisions',type=int,default=16384);parser.add_argument('--resume',type=Path);a=parser.parse_args();cfg=next(c for c in json.loads(PROTOCOL.read_text())['cells'] if c['name']==a.cell);run(cfg,a.output,a.decisions,a.resume)
