"""One fresh on-policy batch; detached critic features, no runtime alterations."""
import sys,json,hashlib,random,time,multiprocessing
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch
from torch.distributions import Categorical,Normal
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src'),str(ROOT/'training')]
from common import World,Policy,Critic,MemoryCritic,sequence,density,selective_terms
from runtime import decode
CFG=json.loads(Path(__file__).with_name('protocol.json').read_text());POOL=[ROOT/'runs/retained/baseline/V13.pt',ROOT/'runs/retained/selected/candidate.pt',ROOT/'runs/geometric-retained/latest.pt'];COLS=[3,4,5,9,10,11]
def value(c,x,h,memory):return c(x,h.detach()) if memory else c(x)
def collect(state,critic_state,use_memory,index):
 torch.set_num_threads(1);seed=CFG['training_seed']+index*1009;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed);p=Policy().eval();p.load_state_dict(state,strict=True);c=MemoryCritic().eval() if use_memory else Critic().eval();c.load_state_dict(critic_state,strict=True);o=Policy().eval();which=int(rng.integers(3));o.load_state_dict(torch.load(POOL[which],weights_only=True)['state_dict'],strict=True);tag=CFG['first_batch_scenarios'][index];stage='B' if tag.startswith('B') else tag;ki=int(tag[1:]) if stage=='B' else None;ticks=1 if index%2 else 2;side=(index//2)%2;w=World(seed,stage,side,ki);h=None;oh=None;xs=[];hs=[];acts=[];values=[];rewards=[];masks=[];parts=[];goals=[];contacts={'goalward':0,'backward':0,'meaningless':0,'touch':0};first=None
 try:
  with torch.inference_mode():
   for step in range(CFG['rollout_decisions']):
    x=w.observe(w.our);ox=w.observe(w.enemy);mask=w.approach_mask(x);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh);m=Categorical(logits=l).sample().squeeze();z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),CFG['steer_std']).sample().squeeze();lp=float(density(l.squeeze(0),s.squeeze(0),m,z));v=float(value(c,torch.tensor(x),h[0,0],use_memory));old,touch,goal=w.step(decode(int(m),float(torch.tanh(z))),decode(int(ol.argmax()),float(os)),ticks=ticks);gamma=CFG['gamma_at_60Hz']**(ticks/2);component,diag=w.components(old,touch,goal,ticks,gamma)
    if goal is not None:
     phi=w.potential(w.state);w.reset('E',side,w.state.tick_count);component['goal_reset_potential']=gamma*(w.potential(w.state)-phi);goals.append({'step':step,'for':goal==side})
    xs.append(x);hs.append(h[0,0].numpy().copy());acts.append([int(m),float(z),lp]);values.append(v);rewards.append(sum(component.values()));masks.append(mask);parts.append(list(component.values()))
    for k in contacts:contacts[k]+=diag[k]
    if touch and first is None:first=step*ticks/120
   nx=w.observe(w.our);nl,ns,nh=p(torch.tensor(nx[None]),h);boot=float(value(c,torch.tensor(nx),nh[0,0],use_memory))
   x=torch.tensor(np.array(xs));l,s,last,mem=sequence(p,x);torch.testing.assert_close(mem,torch.tensor(np.array(hs)),rtol=1e-5,atol=1e-6);torch.testing.assert_close(last,h,rtol=1e-5,atol=1e-6);a=torch.tensor(acts);torch.testing.assert_close(density(l,s,a[:,0].long(),a[:,1]),a[:,2],rtol=1e-4,atol=1e-4)
  return {'x':np.array(xs),'memory':np.array(hs),'actions':np.array(acts,np.float32),'values':np.array(values,np.float32),'rewards':np.array(rewards,np.float32),'mask':np.array(masks),'components':np.array(parts,np.float32),'bootstrap':boot,'duration':ticks/2,'meta':{'index':index,'seed':seed,'side':side,'scenario':tag,'Hz':120//ticks,'opponent':str(POOL[which].relative_to(ROOT)),'first_touch_seconds':first,'goals':goals,'contacts':contacts,'mode_counts':np.bincount(np.asarray(acts)[:,0].astype(int),minlength=4).tolist(),'anchor_fraction':float(np.mean(masks)),'reward_component_sums':np.sum(parts,axis=0).tolist(),'reward_component_order':['goal','potential','productive_velocity','nonproductive_contact','stationary','goal_reset_potential'],'bootstrap_value':boot,'alignment':'PASS fullsequence = callbacks, finite old behavior densities; nonterminal goal continuity'}}
 finally:w.close()
def gae(values,rewards,bootstrap,duration):
 g=CFG['gamma_at_60Hz']**duration;lam=CFG['gae_lambda_at_60Hz']**duration;adv=np.zeros(len(values),np.float32);last=0
 for i in range(len(values)-1,-1,-1):n=values[i+1] if i+1<len(values) else bootstrap;last=rewards[i]+g*n-values[i]+g*lam*last;adv[i]=last
 return torch.tensor(adv),torch.tensor(adv+values)
def main():
 out=ROOT/'runs/memory-selective-24576';assert not out.exists();out.mkdir();torch.set_num_threads(2);torch.manual_seed(CFG['training_seed']);np.random.seed(CFG['training_seed']);random.seed(CFG['training_seed']);torch.use_deterministic_algorithms(True);diag=ROOT/'runs/memory-selective-diagnostic';report=json.loads((diag/'summary.json').read_text());use_memory=report['accepted_memory'];actor=Policy();actor.load_state_dict(torch.load(POOL[2],weights_only=True)['state_dict'],strict=True);reference=Policy().eval();reference.load_state_dict(actor.state_dict());reference.requires_grad_(False);critic=MemoryCritic() if use_memory else Critic();initial=torch.load(diag/'memory.pt',weights_only=True) if use_memory else torch.load(ROOT/'runs/anchored-final-32768/latest.pt',weights_only=True);critic.load_state_dict(initial['critic'],strict=True);ao=torch.optim.Adam(actor.parameters(),lr=CFG['actor_lr']);co=torch.optim.Adam(critic.parameters(),lr=CFG['critic_lr']);co.load_state_dict(initial['optimizer'] if use_memory else initial['critic_optimizer']);assert all(p.requires_grad for p in actor.parameters()) and not any(p.requires_grad for p in reference.parameters());assert datetime.now(timezone.utc).timestamp()<datetime.fromisoformat(CFG['stop_training_IST']).timestamp()-120;start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  fs=[pool.submit(collect,actor.state_dict(),critic.state_dict(),use_memory,i) for i in range(12)];rows=[]
  for f in fs:rows.append(f.result());print('SELECTIVE ROLLOUT',rows[-1]['meta'],flush=True)
 arrays={k:np.stack([r[k] for r in rows]) for k in ['x','memory','actions','values','rewards','mask','components']};arrays.update(bootstrap=np.array([r['bootstrap'] for r in rows]),duration=np.array([r['duration'] for r in rows]));np.savez_compressed(out/'on_policy.npz',**arrays);data=[]
 for r in rows:
  a,ret=gae(r['values'],r['rewards'],r['bootstrap'],r['duration']);data.append((torch.tensor(r['x']),torch.tensor(r['memory']),torch.tensor(r['actions']),torch.tensor(r['mask']),a,ret))
 alladv=torch.cat([d[4] for d in data]);am=alladv.mean();asd=alladv.std().clamp_min(1e-5);scale=max(.1,float(torch.cat([d[5] for d in data]).std()));history=[]
 with torch.no_grad():
  for x,h,act,mask,a,ret in data:l,s,_,mem=sequence(actor,x);torch.testing.assert_close(mem,h,rtol=1e-5,atol=1e-6);torch.testing.assert_close(density(l,s,act[:,0].long(),act[:,1]),act[:,2],rtol=1e-4,atol=1e-4)
 # Critic uses stored, detached behavior-policy memory throughout this on-policy update.
 for warm in range(4):
  co.zero_grad()
  for x,h,act,mask,a,ret in data:(.5*((value(critic,x,h,use_memory)-ret)/scale).square().mean()/12).backward()
  torch.nn.utils.clip_grad_norm_(critic.parameters(),1);co.step()
 for epoch in range(CFG['PPO_epochs']):
  ao.zero_grad();co.zero_grad();metrics=[]
  for x,h,act,mask,a,ret in data:
   l,s,_,new_h=sequence(actor,x);lp=density(l,s,act[:,0].long(),act[:,1]);ratio=(lp-act[:,2]).exp();adv=(a-am)/asd;pl=-torch.minimum(ratio*adv,ratio.clamp(1-CFG['clip'],1+CFG['clip'])*adv).mean()
   with torch.no_grad():rl,rs,_,_=sequence(reference,x)
   ce,se,skl=selective_terms(l,s,rs,x,mask);((pl+.01*ce+.1*se+.02*skl)/12).backward();vl=.5*((value(critic,x,h,use_memory)-ret)/scale).square().mean();(vl/12).backward()
   catkl=torch.distributions.kl_divergence(Categorical(logits=rl),Categorical(logits=l)).mean();approx=((ratio-1)-(lp-act[:,2])).mean();metrics.append({'policy_loss':float(pl.detach()),'scoped_CE':float(ce.detach()),'scoped_steer_MSE':float(se.detach()),'scoped_steer_KL':float(skl.detach()),'all_state_mode_KL_monitor_only':float(catkl.detach()),'PPO_approx_KL':float(approx.detach()),'value_normalized_loss':float(vl.detach()),'actor_memory_shift_RMS':float((new_h-h).square().mean().sqrt())})
  actor.encoder[0].weight.grad[:,COLS]=0;ag=torch.nn.utils.clip_grad_norm_(actor.parameters(),1);cg=torch.nn.utils.clip_grad_norm_(critic.parameters(),1);assert torch.isfinite(ag) and torch.isfinite(cg) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters());ao.step();co.step();assert torch.count_nonzero(actor.encoder[0].weight[:,COLS]).item()==0;record={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]};record.update(epoch=epoch+1,actor_gradient_norm=float(ag),critic_gradient_norm=float(cg));history.append(record);print('SELECTIVE UPDATE',record,flush=True)
  if record['PPO_approx_KL']>.02 or record['all_state_mode_KL_monitor_only']>.2:break
 n=np.random.get_state();saved={'state_dict':actor.state_dict(),'critic':critic.state_dict(),'actor_optimizer':ao.state_dict(),'critic_optimizer':co.state_dict(),'critic_kind':'memory' if use_memory else 'observation','configuration':CFG,'additional_PPO_decisions':24576,'lineage_PPO_decisions':40960,'initial_supervised_presentations':337920,'rollout_index':12,'history':history,'rollouts':[r['meta'] for r in rows],'critic_diagnostic':report,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':(n[0],n[1].tolist(),*n[2:]),'opponent_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POOL},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('protocol.json'),ROOT/'training/anchored_final/world.py',ROOT/'training/geometric/world.py',ROOT/'training/sim_contract.py',ROOT/'src/policy.py']}}
 torch.save(saved,out/'latest.tmp.pt');(out/'latest.tmp.pt').replace(out/'latest.pt');torch.save({'state_dict':actor.state_dict()},out/'candidate.pt');(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');summary={'additional_PPO_decisions':24576,'lineage_PPO_decisions':40960,'elapsed_seconds':time.monotonic()-start,'critic_kind':saved['critic_kind'],'scenarios':[r['meta']['scenario'] for r in rows],'candidate_sha256':hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),'main_unchanged':CFG['main_preserved'],'submission_authorized':False};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
