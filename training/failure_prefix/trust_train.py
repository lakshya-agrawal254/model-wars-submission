"""Fresh on-policy PPO suffixes after complete deterministic failure-seeking prefixes.
No rollout controls or privileged features are added to the exported V13 actor.
"""
import sys,json,hashlib,time,random,multiprocessing,argparse
from trust_region import bounded_actor_step
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
OUT=ROOT/'runs/failure-prefix-trust-v1';START=ROOT/'runs/outcome-flip-learning-v1/candidate.pt'
POOL=[START,ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',ROOT/'model/V13.pt',ROOT/'runs/retained/baseline/V13.pt']
CFG=dict(seed=12001011,workers=4,learner_threads=2,max_batches=4,trajectories=8,suffix_length=8192,prefix_cap=10800,Hz=60,epochs=8,actor_lr=.00005,critic_lr=.002,gamma=.999,gae_lambda=.999,steer_std=.02,mode_temperature=.10,clip=.15,mode_reference_KL=.02,steer_reference_squared_latent=1.,maximum_wall_seconds=1800,initializer_SHA256='8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043',selection='Screen batch2 and final batch4 on independent full matches; stop for material competitive/movement regression. No main replacement or submission. Prefixes are not PPO samples; all prefix observations remain in GRU replay.',post_step_mean_KL_limit=.01,post_step_trajectory_KL_limit=.025,max_backtracking_attempts=9,reason='Corrected optimizer guard and sustained-stall roll-ins. Prior sampled data missed sustained stalls seen by the greedy player. New failure-seeking greedy roll-in exposes these histories before fresh low-noise on-policy recovery learning. Reward coefficients unchanged.')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def density(l,s,m,z):
 return Categorical(logits=l/CFG['mode_temperature']).log_prob(m)+(m==1)*(Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).log_prob(z)-torch.log(1-torch.tanh(z).square()+1e-6))
def failure(w,trace,stuck,stall_only=False):
 c=w.state.cars[w.our].physics;ball=w.state.ball.position;dist=float(np.linalg.norm(c.position-ball));other=float(np.linalg.norm(c.position-w.state.cars[w.enemy].physics.position));speed=float(np.linalg.norm(c.linear_velocity))
 if stuck>=180 and dist>180:return 'near_car_stall' if other<250 else 'stationary'
 raw=w.engine._cars[w.our].get_state()
 # Transient kickoff bumps are not sustained stalls; keep driving greedily through them.
 if not stall_only and len(trace)>=300:
  a=np.asarray(trace[-300:]);travel=np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).sum();delta=np.linalg.norm(a[-1,:2]-a[0,:2])
  if travel>1800 and delta<400:return 'circling_proxy'
 return None

def collect(state,cs,index,batch,suffix_length=None,prefix_cap=None):
 torch.set_num_threads(1);seed=CFG['seed']+1009*index+100003*batch;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 p=Policy().eval();p.load_state_dict(state);c=Critic().eval();c.load_state_dict(cs);o=Policy().eval();op=POOL[(index//2+batch)%4];o.load_state_dict(torch.load(op,weights_only=True)['state_dict'])
 seek=index%4 in [2,3];side=index%2;stage='E' if seek or index<2 else ['O','F','M','R'][batch%4]
 # Each batch has ordinary play and failure-seeking matches on both sides; role/opponent rotates.
 w=ForecastWorld(seed,stage,side);h=oh=None;rows={k:[] for k in ['x','memory','actions','values','rewards','components','ppo_mask']};goals=[];trace=[];stuck=0;prefix=0;trigger=None;trigger_tick=None;active=not seek;count=0
 contacts={'touch':0,'goalward':0,'backward':0,'meaningless':0};stationary=nearcar=0.;recovered_after=None
 cap=CFG['prefix_cap'] if prefix_cap is None else prefix_cap;length=CFG['suffix_length'] if suffix_length is None else suffix_length
 try:
  with torch.inference_mode():
   while count<length:
    if not active and (len(rows['x'])>=cap or (len(rows['x'])>=60 and failure(w,trace,stuck,stall_only=index in [2,3]))):
     trigger=failure(w,trace,stuck,stall_only=index in [2,3]) or 'prefix_cap';active=True;prefix=len(rows['x']);trigger_tick=w.state.tick_count
    x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh)
    if active:
     m=Categorical(logits=l/CFG['mode_temperature']).sample().squeeze();z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).sample().squeeze();lp=float(density(l.squeeze(0),s.squeeze(0),m,z));count+=1
    else:m=l.argmax().squeeze();z=torch.atanh(s.squeeze().clamp(-1+1e-6,1-1e-6));lp=0.
    v=float(c(torch.tensor(x)));executed_steer=float(torch.tanh(z)) if active else float(s.squeeze());old,touch,goal=w.step(decode(int(m),executed_steer),decode(int(ol.argmax()),float(os)),ticks=2)
    component,diag=w.components(old,touch,goal,2,CFG['gamma']);car=w.state.cars[w.our].physics;speed=float(np.linalg.norm(car.linear_velocity));stuck=stuck+1 if speed<100 else 0;trace.append(car.position.copy());stationary+=int(speed<100)/60;nearcar+=int(speed<100 and np.linalg.norm(car.position-w.state.cars[w.enemy].physics.position)<250)/60
    if trigger_tick is not None and recovered_after is None and speed>400 and bool(w.state.cars[w.our].on_ground):recovered_after=(w.state.tick_count-trigger_tick)/120
    if goal is not None:
     phi=w.potential(w.state);w.reset('E',side,w.state.tick_count);component['goal_reset_potential']=CFG['gamma']*(w.potential(w.state)-phi);goals.append(dict(step=len(rows['x']),goal_for=goal==side));trace=[];stuck=0
    rows['x'].append(x);rows['memory'].append(h[0,0].numpy().copy());rows['actions'].append([int(m),float(z),lp,executed_steer]);rows['values'].append(v);rows['rewards'].append(sum(component.values()));rows['components'].append(list(component.values()));rows['ppo_mask'].append(active)
    for k in contacts:contacts[k]+=diag[k]
   bootstrap=float(c(torch.tensor(w.observe(w.our))))
   rows={k:np.asarray(v,np.float32) for k,v in rows.items()};x=torch.tensor(rows['x']);l,s,last,mem=sequence(p,x);a=torch.tensor(rows['actions']);mask=torch.tensor(rows['ppo_mask']).bool()
   assert mask.sum()==length and not mask[:prefix].any() and mask[prefix:].all();assert np.isfinite(rows['x']).all()
   torch.testing.assert_close(mem,torch.tensor(rows['memory']),rtol=1e-5,atol=1e-6);torch.testing.assert_close(last,h,rtol=1e-5,atol=1e-6);torch.testing.assert_close(density(l[mask],s[mask],a[mask,0].long(),a[mask,1]),a[mask,2],rtol=1e-4,atol=1e-4)
   if prefix:
    assert torch.equal(l[:prefix].argmax(-1),a[:prefix,0].long());torch.testing.assert_close(s[:prefix],torch.tanh(a[:prefix,1]),rtol=1e-5,atol=1e-6)
  rows.update(bootstrap=bootstrap,meta=dict(index=index,batch=batch,seed=seed,side=side,stage=stage,opponent=str(op.relative_to(ROOT)),greedy_prefix=prefix,ppo_decisions=count,trigger=trigger,recovery_speed400_seconds=recovered_after,stationary_seconds=stationary,stationary_near_other_car_seconds=nearcar,contacts=contacts,goals=goals,modes=np.bincount(rows['actions'][mask.numpy(),0].astype(int),minlength=4).tolist(),reward_component_order=list(component),reward_sums=rows['components'][mask.numpy()].sum(axis=0).tolist(),forecast_valid_fraction=float(x[:,9].mean()),recurrent_and_actual_behavior_replay='PASS',prefix_excluded_from_PPO=True))
  return rows
 finally:w.close()

def gae(r):
 a=np.zeros(len(r['rewards']),np.float32);last=0
 for i in range(len(a)-1,-1,-1):
  nxt=r['values'][i+1] if i+1<len(a) else r['bootstrap'];last=r['rewards'][i]+CFG['gamma']*nxt-r['values'][i]+CFG['gamma']*CFG['gae_lambda']*last;a[i]=last
 return torch.tensor(a),torch.tensor(a+r['values'])

def save(actor,critic,ao,co,history,metas,hashes,decisions,all_decisions,batch,epoch,critic_updates):
 n=np.random.get_state();d=dict(state_dict=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),configuration=CFG,additional_PPO_decisions=decisions,all_new_learner_decisions=all_decisions,actor_updates=len(history),critic_updates=critic_updates,history=history,rollouts=metas,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(n[0],n[1].tolist(),*n[2:]),batch_data_sha256=hashes,completed_batch=batch,epoch=epoch,initial_sha256=sha(START),post_deadline=True)
 
 if not (OUT/f'update-{len(history):03d}.pt').exists():torch.save(d,OUT/f'update-{len(history):03d}.pt')
 torch.save(d,OUT/'latest.staging.pt');(OUT/'latest.staging.pt').replace(OUT/'latest.pt');
 if not (OUT/f'actor-{len(history):03d}.pt').exists():torch.save({'state_dict':actor.state_dict()},OUT/f'actor-{len(history):03d}.pt')
 return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batches',type=int,default=2);ap.add_argument('--resume',action='store_true');args=ap.parse_args();assert 1<=args.batches<=CFG['max_batches'];torch.set_num_threads(2);begin=time.monotonic();torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed']);torch.use_deterministic_algorithms(True)
 actor=Policy();actor.load_state_dict(torch.load(START,weights_only=True)['state_dict'],strict=True);ref=Policy().eval();ref.load_state_dict(actor.state_dict());ref.requires_grad_(False);critic=Critic();old=torch.load(ROOT/'runs/near-greedy-final-v1/latest.pt',weights_only=True);critic.load_state_dict(old['critic']);ao=torch.optim.Adam(actor.parameters(),lr=CFG['actor_lr']);co=torch.optim.Adam(critic.parameters(),lr=CFG['critic_lr']);co.load_state_dict(old['critic_optimizer']);history=[];metas=[];hashes={};decisions=all_decisions=critic_updates=0;first_batch=0
 assert sha(START)==CFG['initializer_SHA256'];assert sum(p.numel() for p in actor.parameters())==26181
 if args.resume:
  assert OUT.exists();d=torch.load(OUT/'latest.pt',weights_only=True);assert d['epoch']==CFG['epochs'] or d.get('batch_complete',False),'Resume only a complete rollout/update stage';actor.load_state_dict(d['state_dict']);critic.load_state_dict(d['critic']);ao.load_state_dict(d['actor_optimizer']);co.load_state_dict(d['critic_optimizer']);history=d['history'];metas=d['rollouts'];hashes=d['batch_data_sha256'];decisions=d['additional_PPO_decisions'];all_decisions=d['all_new_learner_decisions'];critic_updates=d['critic_updates'];first_batch=d['completed_batch']+1;torch.set_rng_state(d['torch_rng']);random.setstate(d['python_rng']);n=d['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),*n[2:]));preserved=json.loads((OUT/'preservation-before.json').read_text())
 else:
  assert not OUT.exists();OUT.mkdir();preserved={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('runs/**/*.pt')};preserved['model/V13.pt']=sha(ROOT/'model/V13.pt');(OUT/'preservation-before.json').write_text(json.dumps(preserved));(OUT/'protocol.json').write_text(json.dumps(CFG,indent=2));torch.save({'state_dict':actor.state_dict()},OUT/'initial.pt')
 for batch in range(first_batch,args.batches):
  if time.monotonic()-begin>CFG['maximum_wall_seconds']-180:break
  behavior={k:v.detach().clone() for k,v in actor.state_dict().items()};torch.save({'state_dict':behavior},OUT/f'batch-{batch}-behavior.pt')
  with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(collect,[behavior]*8,[critic.state_dict()]*8,range(8),[batch]*8))
  path=OUT/f'batch-{batch}.npz';arrays={f'{k}_{i}':r[k] for i,r in enumerate(rows) for k in ['x','memory','actions','values','rewards','components','ppo_mask']};arrays['bootstrap']=np.array([r['bootstrap'] for r in rows]);np.savez_compressed(path,**arrays);hashes[path.name]=sha(path);decisions+=sum(int(r['ppo_mask'].sum()) for r in rows);all_decisions+=sum(len(r['x']) for r in rows);metas.extend(r['meta'] for r in rows);(OUT/'rollouts.json').write_text(json.dumps(metas,indent=2));print('BATCH',batch,json.dumps([r['meta'] for r in rows]),flush=True)
  if batch==0 and not any(r['meta']['trigger'] not in [None,'prefix_cap'] for r in rows):
   raise RuntimeError('No targeted failure reached in first batch; preserve data, do not optimize an unchanged coverage regime')
  # Complete chronological replay includes deterministic history; losses use fresh stochastic suffixes only.
  data=[(torch.tensor(r['x']),torch.tensor(r['actions']),torch.tensor(r['ppo_mask']).bool(),*gae(r)) for r in rows];adv=torch.cat([a[m] for x,act,m,a,ret in data]);mean=adv.mean();std=adv.std().clamp_min(1e-5);scale=max(.1,float(torch.cat([ret[m] for x,act,m,a,ret in data]).std()))
  for warm in range(4):
   co.zero_grad()
   for x,act,m,a,ret in data:(.5*((critic(x)[m]-ret[m])/scale).square().mean()/len(data)).backward()
   torch.nn.utils.clip_grad_norm_(critic.parameters(),1);co.step();critic_updates+=1
  frozen=Policy().eval();frozen.load_state_dict(behavior);old_policy=[]
  with torch.inference_mode():
   for x,act,m,a,ret in data:
    ol,os,_,_=sequence(frozen,x);old_policy.append((ol[m].clone(),torch.atanh(os[m].clamp(-1+1e-6,1-1e-6)).clone()))
  def post_step_check():
   values=[]
   with torch.inference_mode():
    for (x,act,m,a,ret),(ol,oz) in zip(data,old_policy):
     l,s,_,_=sequence(actor,x);nz=torch.atanh(s[m].clamp(-1+1e-6,1-1e-6));kl=kl_divergence(Categorical(logits=ol/CFG['mode_temperature']),Categorical(logits=l[m]/CFG['mode_temperature']))+(ol/CFG['mode_temperature']).softmax(-1)[:,1]*(nz-oz).square()/(2*CFG['steer_std']**2);values.append(float(kl.mean()))
   return dict(mean_KL=float(np.mean(values)),max_trajectory_KL=max(values),trajectory_KL=values)
  for epoch in range(CFG['epochs']):
   ao.zero_grad();co.zero_grad();metrics=[]
   for x,act,m,a,ret in data:
    l,s,_,_=sequence(actor,x);lp=density(l[m],s[m],act[m,0].long(),act[m,1]);ratio=(lp-act[m,2]).exp();ad=(a[m]-mean)/std;pl=-torch.minimum(ratio*ad,ratio.clamp(.85,1.15)*ad).mean()
    with torch.no_grad():rl,rs,_,_=sequence(ref,x)
    catkl=kl_divergence(Categorical(logits=rl[m]/CFG['mode_temperature']),Categorical(logits=l[m]/CFG['mode_temperature'])).mean()
    # Only reliable moving, forward approaches receive a steering reference penalty.
    reliable=m & (x[:,7]>.20) & (x[:,0]>0) & (torch.atan2(x[:,1],x[:,0]).abs()<.5)
    squared=(torch.atanh(s.clamp(-1+1e-6,1-1e-6))-torch.atanh(rs.clamp(-1+1e-6,1-1e-6))).square();anchor=squared[reliable].mean() if reliable.any() else s.sum()*0
    ((pl+CFG['mode_reference_KL']*catkl+CFG['steer_reference_squared_latent']*anchor)/len(data)).backward();vl=.5*((critic(x)[m]-ret[m])/scale).square().mean();(vl/len(data)).backward();metrics.append(dict(policy_loss=float(pl.detach()),mode_reference_KL=float(catkl.detach()),steer_anchor=float(anchor.detach()),ppo_KL=float(((ratio-1)-(lp-act[m,2])).mean().detach()),value_loss=float(vl.detach())))
   fg=float(actor.encoder[0].weight.grad[:,3:6].norm());ag=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1));cg=float(torch.nn.utils.clip_grad_norm_(critic.parameters(),1));assert np.isfinite(ag+cg) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());accepted,trials=bounded_actor_step(actor,ao,post_step_check,CFG['actor_lr'],CFG['post_step_mean_KL_limit'],CFG['post_step_trajectory_KL_limit'],CFG['max_backtracking_attempts'])
   trial_record=dict(batch=batch,epoch=epoch+1,accepted=accepted,trials=trials);(OUT/f'proposals-b{batch}-e{epoch+1}.json').write_text(json.dumps(trial_record,indent=2))
   if not accepted:
    print('NO_ACCEPTABLE_STEP',json.dumps(trial_record),flush=True);break
   co.step();critic_updates+=1;rec={k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]};rec.update(batch=batch,epoch=epoch+1,update=len(history)+1,forecast_gradient_norm=fg,gradient_norm=ag,accepted_lr=trials[-1]['lr'],post_step_mean_KL=trials[-1]['mean_KL'],post_step_max_trajectory_KL=trials[-1]['max_trajectory_KL'],optimizer_proposal_calls=len(trials));history.append(rec);d=save(actor,critic,ao,co,history,metas,hashes,decisions,all_decisions,batch,epoch+1,critic_updates);print('UPDATE',json.dumps(rec),flush=True)
  d=save(actor,critic,ao,co,history,metas,hashes,decisions,all_decisions,batch,epoch+1,critic_updates);d['batch_complete']=True;torch.save(d,OUT/'latest.staging.pt');(OUT/'latest.staging.pt').replace(OUT/'latest.pt');torch.save({'state_dict':actor.state_dict()},OUT/f'candidate-batch-{batch+1}.pt');(OUT/f'batch-{batch+1}-summary.json').write_text(json.dumps(dict(additional_PPO_decisions=decisions,all_new_learner_decisions=all_decisions,actor_updates=len(history),critic_updates=critic_updates,elapsed_this_stage=time.monotonic()-begin,finished_UTC=datetime.now(timezone.utc).isoformat(),candidate_SHA256=sha(OUT/f'candidate-batch-{batch+1}.pt'),promoted=False),indent=2))
 assert all(sha(ROOT/f)==h for f,h in preserved.items());print('STAGE_COMPLETE',len(history),decisions,all_decisions,'seconds',time.monotonic()-begin,flush=True)
if __name__=='__main__':main()
