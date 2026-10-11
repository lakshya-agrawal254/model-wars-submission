"""One forecast-enabled on-policy PPO batch; immutable V13 actor contract."""
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
from common import Policy,Critic,sequence,density
from runtime import decode
OUT=ROOT/'runs/forecast-final-24576-v1'
START=ROOT/'runs/outcome-flip-learning-v1/candidate.pt'
POOL=[START,ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',ROOT/'model/V13.pt',ROOT/'runs/retained/baseline/V13.pt']
TAGS=['E','O','F','E','B0','B1','B2','B3','B4','R','O','F']
CFG=dict(seed=11001011,decisions=24576,trajectories=12,length=2048,Hz=60,workers=4,epochs=4,actor_lr=0.00005,critic_lr=0.002,gamma=.999,gae_lambda=.999,steer_std=.15,clip=.15,reference_joint_KL_coefficient=.02,training_cutoff_UTC='2026-10-11T01:00:00+00:00',campaign_wall_limit_seconds=900,actor_start=str(START.relative_to(ROOT)),scenarios=TAGS,changes='Actual 2second physical forecast enabled, forecast weights unfrozen, Outcome128 initializer, varied legal own opponents, no geometric supervised labels or Chase CE. Existing reward coefficients unchanged.',goal_information='True team and goal coordinates restricted to rewards/evaluation; no reliable unique goal identification or direct opponent tracking from13D.',critic='Compatible preserved observation-only critic and Adam; no actor weights imported from rejected PPO.',selection='Reject if no competitive advantage over initializer or movement regressions; no promotion without Windows acceptance and user approval.')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(state,cs,index):
 torch.set_num_threads(1);seed=CFG['seed']+1009*index;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 p=Policy().eval();p.load_state_dict(state);c=Critic().eval();c.load_state_dict(cs);o=Policy().eval();op=POOL[index%4];o.load_state_dict(torch.load(op,weights_only=True)['state_dict'])
 tag=TAGS[index];stage='B' if tag.startswith('B') else tag;side=index%2;w=ForecastWorld(seed,stage,side,int(tag[1:]) if stage=='B' else None);h=oh=None;xs=[];hs=[];acts=[];vals=[];rewards=[];parts=[];changes=0;enemy_touches=0;goals=[];contacts={'touch':0,'goalward':0,'backward':0,'meaningless':0}
 try:
  with torch.inference_mode():
   for step in range(CFG['length']):
    x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh)
    m=Categorical(logits=l).sample().squeeze();z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),.15).sample().squeeze();lp=float(density(l.squeeze(0),s.squeeze(0),m,z));v=float(c(torch.tensor(x)))
    old,touch,goal=w.step(decode(int(m),float(torch.tanh(z))),decode(int(ol.argmax()),float(os)),ticks=2)
    component,diag=w.components(old,touch,goal,2,CFG['gamma']);changes+=int(np.linalg.norm(w.state.ball.linear_velocity-old.ball.linear_velocity)>300);enemy_touches+=int(bool(w.state.cars[w.enemy].ball_touches))
    if goal is not None:
     phi=w.potential(w.state);w.reset('E',side,w.state.tick_count);component['goal_reset_potential']=CFG['gamma']*(w.potential(w.state)-phi);goals.append(dict(step=step,goal_for=goal==side))
    xs.append(x);hs.append(h[0,0].numpy().copy());acts.append([int(m),float(z),lp]);vals.append(v);rewards.append(sum(component.values()));parts.append(list(component.values()))
    for k in contacts:contacts[k]+=diag[k]
   boot=float(c(torch.tensor(w.observe(w.our))))
   x=torch.tensor(np.array(xs));l,s,last,mem=sequence(p,x);a=torch.tensor(acts)
   torch.testing.assert_close(mem,torch.tensor(np.array(hs)),rtol=1e-5,atol=1e-6);torch.testing.assert_close(last,h,rtol=1e-5,atol=1e-6);torch.testing.assert_close(density(l,s,a[:,0].long(),a[:,1]),a[:,2],rtol=1e-4,atol=1e-4)
  return dict(x=np.array(xs),memory=np.array(hs),actions=np.array(acts,np.float32),values=np.array(vals,np.float32),rewards=np.array(rewards,np.float32),components=np.array(parts,np.float32),bootstrap=boot,meta=dict(index=index,seed=seed,side=side,scenario=tag,opponent=str(op.relative_to(ROOT)),goals=goals,contacts=contacts,abrupt_ball_velocity_changes=changes,opponent_touch_callbacks=enemy_touches,reward_component_order=list(component),reward_sums=np.sum(parts,axis=0).tolist(),modes=np.bincount(np.asarray(acts)[:,0].astype(int),minlength=4).tolist(),forecast_valid_fraction=float(x[:,9].mean()),forecast_difference_mean=float((x[:,3:6]-x[:,:3]).norm(dim=1).mean()),callback_delta_range=[float(x[1:,12].min()),float(x[1:,12].max())],nonterminal_bootstrap=boot,recurrent_and_behavior_replay='PASS'))
 finally:w.close()
def gae(r):
 a=np.zeros(len(r['rewards']),np.float32);last=0
 for i in range(len(a)-1,-1,-1):
  nxt=r['values'][i+1] if i+1<len(a) else r['bootstrap'];last=r['rewards'][i]+CFG['gamma']*nxt-r['values'][i]+CFG['gamma']*CFG['gae_lambda']*last;a[i]=last
 return torch.tensor(a),torch.tensor(a+r['values'])
def main():
 assert not OUT.exists();OUT.mkdir();begin=time.monotonic();torch.set_num_threads(2);torch.manual_seed(CFG['seed']);random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.use_deterministic_algorithms(True)
 preserved={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('runs/**/*.pt')};preserved['model/V13.pt']=sha(ROOT/'model/V13.pt');(OUT/'preservation-before.json').write_text(json.dumps(preserved,indent=2));(OUT/'protocol.json').write_text(json.dumps(CFG,indent=2))
 actor=Policy();actor.load_state_dict(torch.load(START,weights_only=True)['state_dict']);assert sum(p.numel() for p in actor.parameters())==26181
 ref=Policy().eval();ref.load_state_dict(actor.state_dict());ref.requires_grad_(False);critic=Critic();prior=torch.load(ROOT/'runs/anchored-final-32768/latest.pt',weights_only=True);critic.load_state_dict(prior['critic']);co=torch.optim.Adam(critic.parameters(),lr=CFG['critic_lr']);co.load_state_dict(prior['critic_optimizer'])
 for g in co.param_groups:g['lr']=CFG['critic_lr']
 ao=torch.optim.Adam(actor.parameters(),lr=CFG['actor_lr']);torch.save({'state_dict':actor.state_dict()},OUT/'initial.pt')
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  rows=list(pool.map(collect,[actor.state_dict()]*12,[critic.state_dict()]*12,range(12)))
 print('COLLECTED',json.dumps([r['meta'] for r in rows]),flush=True)
 arrays={k:np.stack([r[k] for r in rows]) for k in ['x','memory','actions','values','rewards','components']};arrays['bootstrap']=np.array([r['bootstrap'] for r in rows]);np.savez_compressed(OUT/'on_policy.npz',**arrays)
 data=[(torch.tensor(r['x']),torch.tensor(r['actions']),*gae(r)) for r in rows];aa=torch.cat([d[2] for d in data]);mean=aa.mean();std=aa.std().clamp_min(1e-5);scale=max(.1,float(torch.cat([d[3] for d in data]).std()));history=[]
 for warm in range(4):
  co.zero_grad()
  for x,act,a,ret in data:(.5*((critic(x)-ret)/scale).square().mean()/len(data)).backward()
  torch.nn.utils.clip_grad_norm_(critic.parameters(),1);co.step()
 for epoch in range(CFG['epochs']):
  if datetime.now(timezone.utc)>=datetime.fromisoformat(CFG['training_cutoff_UTC']) or time.monotonic()-begin>600:break
  ao.zero_grad();co.zero_grad();metrics=[]
  for x,act,a,ret in data:
   l,s,_,_=sequence(actor,x);lp=density(l,s,act[:,0].long(),act[:,1]);ratio=(lp-act[:,2]).exp();adv=(a-mean)/std;pl=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
   with torch.no_grad():rl,rs,_,_=sequence(ref,x)
   catkl=kl_divergence(Categorical(logits=rl),Categorical(logits=l));skl=(torch.atanh(s.clamp(-1+1e-6,1-1e-6))-torch.atanh(rs.clamp(-1+1e-6,1-1e-6))).square()/(2*.15**2);joint=(catkl+rl.softmax(-1)[:,1]*skl).mean()
   ((pl+.02*joint)/len(data)).backward();vl=.5*((critic(x)-ret)/scale).square().mean();(vl/len(data)).backward();metrics.append(dict(loss=float(pl.detach()),reference_joint_KL=float(joint.detach()),ppo_KL=float(((ratio-1)-(lp-act[:,2])).mean().detach()),value_loss=float(vl.detach())))
  forecast_grad=float(actor.encoder[0].weight.grad[:,3:6].norm());ag=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1));cg=float(torch.nn.utils.clip_grad_norm_(critic.parameters(),1));assert np.isfinite(ag+cg) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());ao.step();co.step()
  record={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]};record.update(update=epoch+1,gradient_norm=ag,forecast_gradient_norm=forecast_grad);history.append(record);print('UPDATE',record,flush=True)
  saved=dict(state_dict=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),configuration=CFG,additional_PPO_decisions=24576,actor_updates=len(history),critic_updates=4+len(history),history=history,rollouts=[r['meta'] for r in rows],torch_rng=torch.get_rng_state(),python_rng=random.getstate(),initial_sha256=sha(START),on_policy_sha256=sha(OUT/'on_policy.npz'),actor_optimizer_note='Fresh Adam for new PPO objective, original supervised optimizer preserved in original archive.',critic_source_sha256=sha(ROOT/'runs/anchored-final-32768/latest.pt'))
  torch.save(saved,OUT/f'update-{epoch+1}.pt')
  if record['ppo_KL']>.02 or record['reference_joint_KL']>.05:break
 assert history,'No actor update completed';torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':actor.state_dict()},OUT/'candidate.pt')
 with torch.no_grad():
  x=data[-1][0];l,s,_,_=sequence(actor,x);xx=x.clone();xx[:,3:6]=0;xx[:,9:12]=0;al,ass,_,_=sequence(actor,xx);rl,rs,_,_=sequence(ref,x)
  behavior=dict(forecast_ablation_steer_mean=float((s-ass).abs().mean()),forecast_ablation_steer_max=float((s-ass).abs().max()),forecast_ablation_mode_changes=int((l.argmax(-1)!=al.argmax(-1)).sum()),changes_vs_start=int((l.argmax(-1)!=rl.argmax(-1)).sum()),steer_change_vs_start_mean=float((s-rs).abs().mean()),forecast_column_norm=float(actor.encoder[0].weight[:,3:6].norm()))
 assert all(sha(ROOT/p)==v for p,v in preserved.items());report=dict(configuration=CFG,additional_decisions=24576,optimizer_updates=len(history),elapsed_seconds=time.monotonic()-begin,finished_UTC=datetime.now(timezone.utc).isoformat(),behavior=behavior,history=history,candidate_SHA256=sha(OUT/'candidate.pt'),preservation_pass=True,promoted=False);(OUT/'result.json').write_text(json.dumps(report,indent=2));(OUT/'rollouts.json').write_text(json.dumps([r['meta'] for r in rows],indent=2));print('COMPLETE',report,flush=True)
if __name__=='__main__':main()
