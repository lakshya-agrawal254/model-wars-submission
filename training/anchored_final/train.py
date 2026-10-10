"""One constrained PPO continuation, no runtime changes or external policies."""
import argparse,hashlib,json,math,random,sys,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.distributions import Categorical,Normal
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src'),str(ROOT/'training')]
from policy import Policy
from ppo import Critic
from runtime import decode
from world import World
COLS=[3,4,5,9,10,11];CFG=json.loads(Path(__file__).with_name('protocol.json').read_text())
POOL=[ROOT/'runs/retained/baseline/V13.pt',ROOT/'runs/retained/selected/candidate.pt',ROOT/'runs/geometric-retained/latest.pt']

def logp(logits,steer,mode,z):
 mu=torch.atanh(steer.clamp(-1+1e-6,1-1e-6));return Categorical(logits=logits).log_prob(mode)+(mode==1)*(Normal(mu,CFG['steer_std']).log_prob(z)-torch.log(1-torch.tanh(z).square()+1e-6))
def joint_kl(ref_l,ref_s,l,s):
 ref=Categorical(logits=ref_l);now=Categorical(logits=l);mu=torch.atanh(ref_s.clamp(-1+1e-6,1-1e-6));nu=torch.atanh(s.clamp(-1+1e-6,1-1e-6));return torch.distributions.kl_divergence(ref,now)+ref.probs[:,1]*(mu-nu).square()/(2*CFG['steer_std']**2)
def advantages(rows):
 values=np.asarray([r[4] for r in rows]);rewards=np.asarray([r[5] for r in rows]);a=np.zeros(len(rows),np.float32);last=0
 for i in range(len(rows)-1,-1,-1):
  g=CFG['gamma_at_60Hz']**rows[i][6];lam=CFG['gae_lambda_at_60Hz']**rows[i][6];nv=values[i+1] if i+1<len(rows) else rows[-1][7];last=rewards[i]+g*nv-values[i]+g*lam*last;a[i]=last
 return torch.tensor(a),torch.tensor(a+values,dtype=torch.float32)

def collect(state,critic_state,index):
 torch.set_num_threads(1);seed=CFG['training_seed']+index*1009;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
 p=Policy().eval();p.load_state_dict(state,strict=True);v=Critic().eval();v.load_state_dict(critic_state,strict=True);op=Policy().eval();which=int(rng.integers(3));op.load_state_dict(torch.load(POOL[which],map_location='cpu',weights_only=True)['state_dict'],strict=True)
 phase=index%16;stage='E' if phase<8 else 'O' if phase<11 else 'F' if phase<14 else 'B' if phase==14 else 'G';ticks=1 if (index//4)%2 else 2;side=index%2;w=World(seed,stage,side);h=None;oh=None;rows=[];counts=np.zeros(4,int);stats={'goals_for':0,'goals_against':0,'productive_contacts':0,'nonproductive_contacts':0,'touches':0,'stationary_decisions':0};first=None;distance=[]
 try:
  with torch.inference_mode():
   for step in range(CFG['rollout_decisions']):
    x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None,:]),h);mode=Categorical(logits=l).sample().squeeze(0);z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).sample().squeeze(0);lp=float(logp(l.squeeze(0),s.squeeze(0),mode,z));value=float(v(torch.tensor(x)));ol,os,oh=op(torch.tensor(ox[None,:]),oh)
    old,touch,goal=w.step(decode(int(mode),float(torch.tanh(z))),decode(int(ol.argmax()),float(os)),ticks=ticks);gamma=CFG['gamma_at_60Hz']**(ticks/2);reward,detail=w.competitive_reward(old,touch,goal,ticks,gamma);phi=w.potential(w.state)
    if goal is not None:
     w.reset('E',side,w.state.tick_count);reward+=gamma*(w.potential(w.state)-phi) # potential uses actual next state after kickoff reset; hidden persists
    rows.append((x,int(mode),float(z),lp,value,reward,ticks/2));counts[int(mode)]+=1
    for k,n in detail.items():stats[k]+=n
    stats['touches']+=int(touch);stats['stationary_decisions']+=int(np.linalg.norm(w.state.cars[w.our].physics.linear_velocity)<100)
    if touch and first is None:first=step*ticks/120
    distance.append(float(np.linalg.norm(w.state.ball.position-w.state.cars[w.our].physics.position)))
   rows[-1]=(*rows[-1],float(v(torch.tensor(w.observe(w.our)))))
  return rows,stats|{'index':index,'seed':seed,'stage':stage,'team':side,'callback_Hz':120//ticks,'opponent':str(POOL[which].relative_to(ROOT)),'first_contact_seconds':first,'counts':counts.tolist(),'mean_car_ball_distance':float(np.mean(distance))}
 finally:w.close()

def unpack(batch):
 result=[]
 for rows in batch:
  a,r=advantages(rows);result.append((torch.tensor(np.asarray([row[0] for row in rows])),torch.tensor([row[1] for row in rows]),torch.tensor([row[2] for row in rows]),torch.tensor([row[3] for row in rows]),a,r))
 return result

def update(actor,critic,ao,co,reference,batch,anchor_data,warm=False):
 data=unpack(batch);count=sum(len(v[0]) for v in data);alladv=torch.cat([d[4] for d in data]);mean=alladv.mean();std=alladv.std().clamp_min(1e-5);scale=max(1.,float(torch.cat([d[5] for d in data]).std()));metrics=[]
 with torch.no_grad():
  for x,m,z,old,a,r in data:torch.testing.assert_close(logp(*actor(x)[:2],m,z),old,rtol=1e-4,atol=1e-4)
 if warm:
  for i in range(4):
   co.zero_grad(set_to_none=True)
   for x,m,z,old,a,r in data:(.5*(critic(x)-r).square().mean()/scale**2*len(x)/count).backward()
   assert torch.isfinite(nn.utils.clip_grad_norm_(critic.parameters(),1));co.step()
 for epoch in range(CFG['PPO_epochs']):
  ao.zero_grad(set_to_none=True);co.zero_grad(set_to_none=True);stats=[]
  for x,m,z,old,a,r in data:
   l,s,h=actor(x);lp=logp(l,s,m,z);ratio=(lp-old).exp();adv=(a-mean)/std;loss=-torch.minimum(ratio*adv,ratio.clamp(1-CFG['clip'],1+CFG['clip'])*adv).mean()
   with torch.no_grad():rl,rs,rh=reference(x)
   kl=joint_kl(rl,rs,l,s).mean();weight=len(x)/count;((loss+CFG['reference_joint_KL_weight']*kl)*weight).backward();vl=.5*(critic(x)-r).square().mean();(vl/scale**2*weight).backward();stats.append((weight,float(loss.detach()),float(kl.detach()),float(vl.detach()),float(((ratio-1)-(lp-old)).mean().detach())))
  # Independent geometric labels are a training-only movement anchor, not policy control overrides.
  indices=torch.randperm(len(anchor_data['observations']))[:4]
  for index in indices:
   x=torch.tensor(anchor_data['observations'][index]);m=torch.tensor(anchor_data['modes'][index]);target=torch.tensor(anchor_data['steering'][index]);l,s,h=actor(x);mask=m==1;ce=nn.functional.cross_entropy(l,m);se=nn.functional.mse_loss(s[mask],target[mask]) if mask.any() else s.sum()*0
   (CFG['supervised_anchor_weight']*(ce+se)/len(indices)).backward()
  actor.encoder[0].weight.grad[:,COLS]=0;gn=nn.utils.clip_grad_norm_(actor.parameters(),1);vn=nn.utils.clip_grad_norm_(critic.parameters(),1);assert torch.isfinite(gn) and torch.isfinite(vn) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());ao.step();co.step();assert torch.count_nonzero(actor.encoder[0].weight[:,COLS]).item()==0
  names=['policy_loss','reference_joint_KL','critic_loss','PPO_approx_KL'];d={k:sum(v[0]*v[i+1] for v in stats) for i,k in enumerate(names)};d.update(actor_gradient_norm=float(gn),critic_gradient_norm=float(vn),return_loss_scale=scale);metrics.append(d)
  if d['PPO_approx_KL']>.02 or d['reference_joint_KL']>.15:break
 return {k:float(np.mean([d[k] for d in metrics])) for k in metrics[0]}|{'epochs':len(metrics),'warm_critic':warm}

def main(output,budget,resume=None):
 assert budget%16384==0 and budget<=CFG['maximum_budget'];assert not output.exists();output.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(CFG['training_seed']);np.random.seed(CFG['training_seed']);random.seed(CFG['training_seed'])
 reference=Policy().eval();reference.load_state_dict(torch.load(ROOT/'runs/geometric-retained/latest.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
 for p in reference.parameters():p.requires_grad_(False)
 actor=Policy();actor.load_state_dict(reference.state_dict(),strict=True);critic=Critic();ao=torch.optim.Adam(actor.parameters(),lr=CFG['actor_lr']);co=torch.optim.Adam(critic.parameters(),lr=CFG['critic_lr']);decisions=0;index=0;history=[]
 if resume:
  saved=torch.load(resume,map_location='cpu',weights_only=True);actor.load_state_dict(saved['state_dict'],strict=True);critic.load_state_dict(saved['critic'],strict=True);ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer']);decisions=saved['additional_PPO_decisions'];index=saved['rollout_index'];history=saved['history'];torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);n=saved['numpy_rng'];np.random.set_state((n[0],np.array(n[1],np.uint32),*n[2:]))
 data=np.load(ROOT/'runs/geometric-retained/data.npz');anchor_data={k:data[k] for k in data.files};start=time.monotonic();target=decisions+budget;assert target<=CFG['maximum_budget']
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  while decisions<target:
   if datetime.now(timezone.utc).timestamp()>=datetime.fromisoformat(CFG['training_stop_IST']).timestamp()-120:print('DEADLINE GUARD STOP',flush=True);break
   actor.eval();state={k:v.detach().clone() for k,v in actor.state_dict().items()};vs={k:v.detach().clone() for k,v in critic.state_dict().items()};futures=[pool.submit(collect,state,vs,index+i) for i in range(4)];result=[f.result() for f in futures];index+=4;actor.train();stats=update(actor,critic,ao,co,reference,[r[0] for r in result],anchor_data,warm=decisions==0);decisions+=16384;history.append({'additional_PPO_decisions':decisions,'rollouts':[r[1] for r in result],'training':stats,'segment_elapsed_seconds':time.monotonic()-start});print('ANCHORED',decisions,stats,[(r[1]['goals_for'],r[1]['goals_against']) for r in result],flush=True)
   n=np.random.get_state();saved={'state_dict':actor.state_dict(),'critic':critic.state_dict(),'actor_optimizer':ao.state_dict(),'critic_optimizer':co.state_dict(),'configuration':CFG,'additional_PPO_decisions':decisions,'initial_PPO_lineage_decisions':16384,'lineage_PPO_decisions':16384+decisions,'initial_supervised_presentations':337920,'rollout_index':index,'history':history,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':(n[0],n[1].tolist(),*n[2:]),'opponent_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POOL},'anchor_data_sha256':hashlib.sha256((ROOT/'runs/geometric-retained/data.npz').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('world.py'),Path(__file__).with_name('protocol.json'),ROOT/'training/geometric/world.py',ROOT/'training/sim_contract.py',ROOT/'src/policy.py']}}
   temp=output/'latest.tmp.pt';torch.save(saved,temp);temp.replace(output/'latest.pt');torch.save({'state_dict':actor.state_dict()},output/'candidate.pt');(output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 summary={'additional_PPO_decisions':decisions,'segment_decisions':decisions-(target-budget),'elapsed_seconds':time.monotonic()-start,'lineage_PPO_decisions':16384+decisions,'budget_requested':budget,'complete_budget':decisions==target};(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--decisions',type=int,default=32768);p.add_argument('--resume',type=Path);a=p.parse_args();main(a.output,a.decisions,a.resume)
