"""Bounded near-greedy multi-batch on-policy PPO; immutable V13 actor contract."""
import sys,json,hashlib,time,random,multiprocessing
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.distributions import Categorical,Normal,kl_divergence
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'training/interception_readout'),str(ROOT/'training/memory_selective'),str(ROOT/'src'),str(ROOT/'training')]
from forecast_world import ForecastWorld
from common import Policy,Critic,sequence
from runtime import decode
OUT=ROOT/'runs/near-greedy-final-v1'
START=ROOT/'runs/outcome-flip-learning-v1/candidate.pt'
POOL=[START,ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',ROOT/'model/V13.pt',ROOT/'runs/retained/baseline/V13.pt']
TAGS=[['E','E','O','F','M','M','R','E'],['E','E','B0','B1','B2','B3','B4','M'],['E','E','E','E','O','F','M','R']]
CFG=dict(seed=11301011,max_batches=3,trajectories=8,length=8192,Hz=60,workers=4,epochs=12,actor_lr=.0001,critic_lr=.002,gamma=.999,gae_lambda=.999,steer_std=.10,mode_temperature=.35,clip=.15,reference_joint_KL_coefficient=.02,training_cutoff_UTC='2026-10-11T00:55:00+00:00',max_wall_seconds=900,initializer=str(START.relative_to(ROOT)),changes='Training-only categorical temperature reduces sampled-vs-greedy mismatch; steering exploration .10;8192step complete histories;3fresh policy/data cycles rather than4epochs onone batch. Original actor/argmax decoder/runtime unchanged.',reward='Unchanged goal+10/-10, potential-based shaping and small productive delta-velocity terms; no new coefficient tuning',selection='Last complete checkpoint is candidate. Do not select on reward. Reject if matched competitive or movement screen fails. No more campaigns.')
def density(l,s,m,z):
 return Categorical(logits=l/CFG['mode_temperature']).log_prob(m)+(m==1)*(Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).log_prob(z)-torch.log(1-torch.tanh(z).square()+1e-6))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(state,cs,index,batch):
 torch.set_num_threads(1);seed=CFG['seed']+1009*index+batch*100003;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 p=Policy().eval();p.load_state_dict(state);c=Critic().eval();c.load_state_dict(cs);o=Policy().eval();op=POOL[index//2];o.load_state_dict(torch.load(op,weights_only=True)['state_dict'])
 tag=TAGS[batch][index];stage='B' if tag.startswith('B') else tag;side=index%2;w=ForecastWorld(seed,stage,side,int(tag[1:]) if stage=='B' else None);h=oh=None;xs=[];hs=[];acts=[];vals=[];rewards=[];parts=[];changes=0;enemy_touches=0;goals=[];contacts={'touch':0,'goalward':0,'backward':0,'meaningless':0}
 try:
  with torch.inference_mode():
   for step in range(CFG['length']):
    if step%256==0 and datetime.now(timezone.utc)>=datetime.fromisoformat(CFG['training_cutoff_UTC']):break
    x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh)
    m=Categorical(logits=l/CFG['mode_temperature']).sample().squeeze();z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).sample().squeeze();lp=float(density(l.squeeze(0),s.squeeze(0),m,z));v=float(c(torch.tensor(x)))
    old,touch,goal=w.step(decode(int(m),float(torch.tanh(z))),decode(int(ol.argmax()),float(os)),ticks=2)
    component,diag=w.components(old,touch,goal,2,CFG['gamma']);changes+=int(np.linalg.norm(w.state.ball.linear_velocity-old.ball.linear_velocity)>300);enemy_touches+=int(bool(w.state.cars[w.enemy].ball_touches))
    if goal is not None:
     phi=w.potential(w.state);w.reset('E',side,w.state.tick_count);component['goal_reset_potential']=CFG['gamma']*(w.potential(w.state)-phi);goals.append(dict(step=step,goal_for=goal==side))
    xs.append(x);hs.append(h[0,0].numpy().copy());acts.append([int(m),float(z),lp]);vals.append(v);rewards.append(sum(component.values()));parts.append(list(component.values()))
    for k in contacts:contacts[k]+=diag[k]
   boot=float(c(torch.tensor(w.observe(w.our))))
   x=torch.tensor(np.array(xs));l,s,last,mem=sequence(p,x);a=torch.tensor(acts)
   torch.testing.assert_close(mem,torch.tensor(np.array(hs)),rtol=1e-5,atol=1e-6);torch.testing.assert_close(last,h,rtol=1e-5,atol=1e-6);torch.testing.assert_close(density(l,s,a[:,0].long(),a[:,1]),a[:,2],rtol=1e-4,atol=1e-4)
  return dict(x=np.array(xs),memory=np.array(hs),actions=np.array(acts,np.float32),values=np.array(vals,np.float32),rewards=np.array(rewards,np.float32),components=np.array(parts,np.float32),bootstrap=boot,meta=dict(batch=batch,index=index,seed=seed,side=side,scenario=tag,opponent=str(op.relative_to(ROOT)),goals=goals,contacts=contacts,abrupt_ball_velocity_changes=changes,opponent_touch_callbacks=enemy_touches,reward_component_order=list(component),reward_sums=np.sum(parts,axis=0).tolist(),modes=np.bincount(np.asarray(acts)[:,0].astype(int),minlength=4).tolist(),forecast_valid_fraction=float(x[:,9].mean()),forecast_difference_mean=float((x[:,3:6]-x[:,:3]).norm(dim=1).mean()),callback_delta_range=[float(x[1:,12].min()),float(x[1:,12].max())],nonterminal_bootstrap=boot,recurrent_and_behavior_replay='PASS'))
 finally:w.close()
def gae(r):
 a=np.zeros(len(r['rewards']),np.float32);last=0
 for i in range(len(a)-1,-1,-1):
  nxt=r['values'][i+1] if i+1<len(a) else r['bootstrap'];last=r['rewards'][i]+CFG['gamma']*nxt-r['values'][i]+CFG['gamma']*CFG['gae_lambda']*last;a[i]=last
 return torch.tensor(a),torch.tensor(a+r['values'])
def expired(buffer=0):return datetime.now(timezone.utc).timestamp()+buffer>=datetime.fromisoformat(CFG['training_cutoff_UTC']).timestamp()
def main():
 assert not OUT.exists();OUT.mkdir();begin=time.monotonic();torch.set_num_threads(2);torch.manual_seed(CFG['seed']);random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.use_deterministic_algorithms(True)
 preserved={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('runs/**/*.pt')};preserved['model/V13.pt']=sha(ROOT/'model/V13.pt');(OUT/'preservation-before.json').write_text(json.dumps(preserved));(OUT/'protocol.json').write_text(json.dumps(CFG,indent=2))
 actor=Policy();actor.load_state_dict(torch.load(START,weights_only=True)['state_dict'],strict=True);ref=Policy().eval();ref.load_state_dict(actor.state_dict());ref.requires_grad_(False);critic=Critic();prior=torch.load(ROOT/'runs/anchored-final-32768/latest.pt',weights_only=True);critic.load_state_dict(prior['critic']);ao=torch.optim.Adam(actor.parameters(),lr=CFG['actor_lr']);co=torch.optim.Adam(critic.parameters(),lr=CFG['critic_lr']);co.load_state_dict(prior['critic_optimizer'])
 for g in co.param_groups:g['lr']=CFG['critic_lr']
 assert sum(p.numel() for p in actor.parameters())==26181;history=[];rollouts=[];decisions=0;critic_updates=0;data_hashes={};torch.save({'state_dict':actor.state_dict()},OUT/'initial.pt');saved=None
 for batch in range(CFG['max_batches']):
  if expired(120) or time.monotonic()-begin>730:break
  # Every batch starts new physical trajectories/h=None and uses the CURRENT policy.
  behavior={k:v.detach().clone() for k,v in actor.state_dict().items()};torch.save({'state_dict':behavior},OUT/f'batch-{batch}-behavior.pt')
  with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(collect,[behavior]*8,[critic.state_dict()]*8,range(8),[batch]*8))
  arrays={f'{k}_{i}':r[k] for i,r in enumerate(rows) for k in ['x','memory','actions','values','rewards','components']};arrays['bootstrap']=np.array([r['bootstrap'] for r in rows]);path=OUT/f'batch-{batch}.npz';np.savez_compressed(path,**arrays);data_hashes[path.name]=sha(path);new_decisions=sum(len(r['x']) for r in rows);decisions+=new_decisions;rollouts.extend([r['meta'] for r in rows]);print('BATCH',batch,'decisions',new_decisions,'goals',sum(len(r['meta']['goals']) for r in rows),'contacts',sum(r['meta']['contacts']['touch'] for r in rows),'modes',np.sum([r['meta']['modes'] for r in rows],axis=0).tolist(),flush=True)
  data=[(torch.tensor(r['x']),torch.tensor(r['actions']),*gae(r)) for r in rows];aa=torch.cat([d[2] for d in data]);mean=aa.mean();std=aa.std().clamp_min(1e-5);scale=max(.1,float(torch.cat([d[3] for d in data]).std()))
  for warm in range(4):
   if expired(20):break
   co.zero_grad()
   for x,act,a,ret in data:(.5*((critic(x)-ret)/scale).square().mean()/len(data)).backward()
   torch.nn.utils.clip_grad_norm_(critic.parameters(),1);co.step();critic_updates+=1
  for epoch in range(CFG['epochs']):
   if expired(20) or time.monotonic()-begin>870:break
   ao.zero_grad();co.zero_grad();metrics=[];aborted=False
   for x,act,a,ret in data:
    if expired(5):aborted=True;break
    l,s,_,_=sequence(actor,x);lp=density(l,s,act[:,0].long(),act[:,1]);ratio=(lp-act[:,2]).exp();adv=(a-mean)/std;pl=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
    with torch.no_grad():rl,rs,_,_=sequence(ref,x)
    catkl=kl_divergence(Categorical(logits=rl/.35),Categorical(logits=l/.35));skl=(torch.atanh(s.clamp(-1+1e-6,1-1e-6))-torch.atanh(rs.clamp(-1+1e-6,1-1e-6))).square()/(2*.10**2);joint=(catkl+(rl/.35).softmax(-1)[:,1]*skl).mean()
    ((pl+.02*joint)/len(data)).backward();vl=.5*((critic(x)-ret)/scale).square().mean();(vl/len(data)).backward();metrics.append(dict(loss=float(pl.detach()),reference_joint_KL=float(joint.detach()),ppo_KL=float(((ratio-1)-(lp-act[:,2])).mean().detach()),value_loss=float(vl.detach())))
   if aborted:break
   fg=float(actor.encoder[0].weight.grad[:,3:6].norm());ag=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1));cg=float(torch.nn.utils.clip_grad_norm_(critic.parameters(),1));assert np.isfinite(ag+cg) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());ao.step();co.step();critic_updates+=1
   rec={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]};rec.update(batch=batch,epoch=epoch+1,update=len(history)+1,forecast_gradient_norm=fg,gradient_norm=ag);history.append(rec)
   n=np.random.get_state();saved=dict(state_dict=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),configuration=CFG,additional_PPO_decisions=decisions,actor_updates=len(history),critic_updates=critic_updates,history=history,rollouts=rollouts,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(n[0],n[1].tolist(),*n[2:]),initial_sha256=sha(START),batch_data_sha256=data_hashes,completed_batch=batch,epoch=epoch+1,actor_optimizer_note='Fresh PPO optimizer from originalOutcome supervised actor; originaloptimizer preserved.')
   torch.save(saved,OUT/f'update-{len(history)}.pt');torch.save({'state_dict':actor.state_dict()},OUT/f'actor-{len(history)}.pt');print('UPDATE',rec,flush=True)
   if rec['ppo_KL']>.02 or rec['reference_joint_KL']>.12:break
  if history and history[-1]['reference_joint_KL']>.12:break
 assert saved is not None;torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':actor.state_dict()},OUT/'candidate.pt');assert all(sha(ROOT/p)==v for p,v in preserved.items())
 report=dict(configuration=CFG,additional_decisions=decisions,decisions_at_saved_checkpoint=saved['additional_PPO_decisions'],optimizer_updates=len(history),critic_updates=critic_updates,elapsed_seconds=time.monotonic()-begin,finished_UTC=datetime.now(timezone.utc).isoformat(),candidate_SHA256=sha(OUT/'candidate.pt'),history=history,preserved_prior_checkpoints=len(preserved),promoted=False,native='UNTESTED');(OUT/'result.json').write_text(json.dumps(report,indent=2));(OUT/'rollouts.json').write_text(json.dumps(rollouts,indent=2));print('COMPLETE',report,flush=True)
if __name__=='__main__':main()
