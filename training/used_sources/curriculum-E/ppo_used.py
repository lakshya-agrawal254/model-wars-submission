"""Exact V13 actor PPO with joint categorical/tanh-Normal behavior probabilities.

Value network is training-only. No new actor tensors, expert weights, labels,
scripted playing overrides, or synthetic native BallPrediction.
"""
import argparse,copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.distributions import Categorical,Normal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from policy import Policy
from runtime import decode
from sim_contract import World

class Critic(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(13,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def distributions(logits,steer,std=.35):
 return Categorical(logits=logits),Normal(torch.atanh(steer.clamp(-1+1e-6,1-1e-6)),std)
def log_probability(logits,steer,modes,z,std=.35):
 cat,norm=distributions(logits,steer,std)
 correction=2*(np.log(2)-z-nn.functional.softplus(-2*z))
 return cat.log_prob(modes)+(norm.log_prob(z)-correction)*(modes==1)

@torch.inference_mode()
def collect(actor,critic,opponent,seed,stage,team,steps=256):
 w=World(seed,stage,team);h=None;oh=None;rows=[];modes=[];latency=[];touches=0;first=None;last=0
 try:
  for t in range(steps):
   x=w.observe(w.our);ox=w.observe(w.enemy);began=time.perf_counter()
   logits,steer,h=actor(torch.tensor(x[None,:]),h);latency.append((time.perf_counter()-began)*1000)
   cat,norm=distributions(logits,steer);mode=cat.sample();z=norm.sample();control=decode(int(mode),float(torch.tanh(z)))
   logp=log_probability(logits,steer,mode,z).item();value=critic(torch.tensor(x)).item()
   ol,os,oh=opponent(torch.tensor(ox[None,:]),oh);oc=decode(int(ol.argmax()),float(os))
   old,touch,goal=w.step(control,oc);reward=w.reward(old,touch,goal,int(mode),last)
   rows.append((x,int(mode),float(z),logp,value,reward));modes.append(int(mode));last=int(mode)
   if touch:touches+=1;first=(t*2/120 if first is None else first)
   if goal is not None:w.reset('E',team,w.state.tick_count) # same match: hidden/clock history preserved
  next_value=critic(torch.tensor(w.observe(w.our))).item()
  rows[-1]=(*rows[-1],next_value)
  return rows,dict(time_limit_bootstrap=next_value,seed=seed,stage=stage,team=team,touches=touches,first_touch_seconds=first,goals_for=w.score[team],goals_against=w.score[1-team],action_counts=np.bincount(modes,minlength=4).tolist(),latency_ms_mean=float(np.mean(latency)))
 finally:w.close()

def tensors(rows):
 x=torch.tensor(np.asarray([r[0] for r in rows]));mode=torch.tensor([r[1] for r in rows]);z=torch.tensor([r[2] for r in rows]);oldp=torch.tensor([r[3] for r in rows]);v=np.array([r[4] for r in rows]);reward=np.array([r[5] for r in rows]);adv=np.zeros(len(rows),np.float32);running=0
 # Time limits bootstrap the next-state value; no hidden restart inside rollout.
 for i in range(len(rows)-1,-1,-1):
  delta=reward[i]+(.99*v[i+1] if i+1<len(rows) else .99*rows[-1][6])-v[i]
  running=delta+(.99*.95*running);adv[i]=running
 return x,mode,z,oldp,torch.tensor(adv),torch.tensor(adv+v,dtype=torch.float32)

def update(actor,critic,ao,co,batches,epochs=4):
 values=[tensors(r) for r in batches];alladv=torch.cat([v[4] for v in values]);mean=alladv.mean();std=alladv.std().clamp_min(1e-5);stats=[]
 # Recompute the FULL chronological actor prefix on every PPO epoch, h=None only at trial start.
 with torch.no_grad():
  for x,m,z,old,a,ret in values:
   l,s,h=actor(x);np.testing.assert_allclose(log_probability(l,s,m,z).numpy(),old.numpy(),rtol=1e-4,atol=1e-4)
 for epoch in range(epochs):
  for x,m,z,old,a,ret in values:
   l,s,h=actor(x);lp=log_probability(l,s,m,z);ratio=(lp-old).exp();adv=(a-mean)/std
   pg=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
   entropy=Categorical(logits=l).entropy().mean();loss=pg-.025*entropy
   ao.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(actor.parameters(),1.);assert torch.isfinite(norm);ao.step()
   value=critic(x);vl=.5*(value-ret).square().mean();co.zero_grad(set_to_none=True);vl.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
   kl=((ratio-1)-(lp-old)).mean().detach().item();stats.append(dict(policy_loss=float(pg.detach()),value_loss=float(vl.detach()),entropy=float(entropy.detach()),approx_kl=kl,clip_fraction=float(((ratio-1).abs()>.2).float().mean()),gradient_norm=float(norm)))
  if stats[-1]['approx_kl']>.03:break
 return {k:float(np.mean([r[k] for r in stats])) for k in stats[0]}

def main(output,decisions,stage,starting=None,resume=None,steps=256,scratch=False,opponents=None):
 output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20261011);np.random.seed(20261011);random.seed(20261011)
 actor=Policy();critic=Critic();baseline=Policy().eval();base=ROOT/'runs/baseline/V13.pt';baseline.load_state_dict(torch.load(base,weights_only=True)['state_dict'])
 if not scratch:actor.load_state_dict(torch.load(starting or base,weights_only=True)['state_dict'],strict=True)
 ao=torch.optim.Adam(actor.parameters(),lr=3e-4);co=torch.optim.Adam(critic.parameters(),lr=1e-3)
 count=0;episode=0;history=[]
 if resume:
  saved=torch.load(resume,weights_only=True,map_location='cpu');actor.load_state_dict(saved['state_dict']);critic.load_state_dict(saved['critic']);ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer']);count=saved['decisions'];episode=saved['episode'];torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state((saved['numpy_rng'][0],np.array(saved['numpy_rng'][1],np.uint32),*saved['numpy_rng'][2:]));history=saved['history']
 pool=[baseline]
 for path in opponents or []:
  p=Policy().eval();p.load_state_dict(torch.load(path,weights_only=True)['state_dict'],strict=True);pool.append(p)
 target=count+decisions;began=time.monotonic()
 while count<target:
  batches=[];summaries=[]
  for _ in range(2):
   if count>=target:break
   # Later stages preserve a 30% normal-match starting-state mixture.
   chosen='E' if stage in ['C','D'] and episode%3==0 else stage
   actor.eval();rows,summary=collect(actor,critic,pool[episode%len(pool)],20261011+episode*1009,chosen,episode%2,min(steps,target-count));batches.append(rows);summaries.append(summary);count+=len(rows);episode+=1
  actor.train();stats=update(actor,critic,ao,co,batches)
  history.append(dict(decisions=count,stage=stage,episodes=summaries,ppo=stats,elapsed_seconds=time.monotonic()-began))
  rng=np.random.get_state();saved=dict(state_dict=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),decisions=count,episode=episode,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(rng[0],rng[1].tolist(),*rng[2:]),history=history,configuration={'stage':stage,'actor_parameters':26181,'prediction':'explicitly absent; NOT native RLBot prediction','mode_entropy':.025,'steer_std':.35,'learning_rate':3e-4,'gamma':.99,'gae_lambda':.95,'ppo_clip':.2,'ppo_epochs':4,'max_kl':.03,'steps':steps,'seed':20261011,'opponents':[str(base)]+[str(p) for p in opponents or []],'reward_version':'ground-idle-v2','time_limits_bootstrapped':True,'initialization':'fresh exact Policy' if scratch else 'official starter or resumed actor','source_sha256':{n:hashlib.sha256((ROOT/'training'/n).read_bytes()).hexdigest() for n in ['ppo.py','sim_contract.py']} })
  temp=output/'latest.tmp.pt';torch.save(saved,temp);temp.replace(output/'latest.pt')
  print('PPO',stage,count,stats['entropy'],stats['approx_kl'],sum(s['touches'] for s in summaries),flush=True)
 torch.save({'state_dict':actor.state_dict()},output/'candidate.pt');(output/'metrics.json').write_text(json.dumps(history,indent=2))
 print('COMPLETE',count,'SECONDS',time.monotonic()-began,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--decisions',type=int,default=8192);p.add_argument('--stage',choices=list('ABCDE'),default='A');p.add_argument('--opponents',nargs='*',type=Path);p.add_argument('--scratch',action='store_true');p.add_argument('--steps',type=int,default=256);p.add_argument('--starting',type=Path);p.add_argument('--resume',type=Path);a=p.parse_args();main(a.output,a.decisions,a.stage,a.starting,a.resume,a.steps,a.scratch,a.opponents)
