"""Frozen actor real-state return regression, independent trajectory holdout."""
import sys,json,hashlib,time,random,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch
from torch.distributions import Categorical,Normal
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src'),str(ROOT/'training')]
from common import World,Policy,Critic,MemoryCritic,CapacityCritic,sequence,density,selective_terms
from runtime import decode
CFG=json.loads(Path(__file__).with_name('protocol.json').read_text());GEO=ROOT/'runs/geometric-retained/latest.pt';OLD=ROOT/'runs/anchored-final-32768/latest.pt';POOL=[ROOT/'runs/retained/baseline/V13.pt',ROOT/'runs/retained/selected/candidate.pt',GEO]

def collect(index):
 torch.set_num_threads(1);pair=index//2;side=index%2;seed=CFG['diagnostic_seed']+pair*1009;torch.manual_seed(seed+side);rng=np.random.default_rng(seed);stage=(['E','O','F','B','R','G']*2+['E','O','F','B','R','G','E','R'])[pair];p=Policy().eval();p.load_state_dict(torch.load(GEO,weights_only=True)['state_dict'],strict=True);old=Critic().eval();old.load_state_dict(torch.load(OLD,weights_only=True)['critic'],strict=True);o=Policy().eval();which=int(rng.integers(3));o.load_state_dict(torch.load(POOL[which],weights_only=True)['state_dict'],strict=True);w=World(seed,stage,side,pair%5 if stage=='B' else None);h=None;oh=None;xs=[];hs=[];rs=[];acts=[];masks=[];parts=[];goals=[];totals={};contacts={'goalward':0,'backward':0,'meaningless':0,'touch':0};recent=[];post=[]
 try:
  with torch.inference_mode():
   for step in range(4096):
    x=w.observe(w.our);ox=w.observe(w.enemy);mask=w.approach_mask(x);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh);m=Categorical(logits=l).sample().squeeze();z=Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),.15).sample().squeeze();lp=float(density(l.squeeze(0),s.squeeze(0),m,z));oldstate,touch,goal=w.step(decode(int(m),float(torch.tanh(z))),decode(int(ol.argmax()),float(os)));component,diag=w.components(oldstate,touch,goal,2,.999)
    if goal is not None:
     phi=w.potential(w.state);w.reset('E',side,w.state.tick_count);component['goal_reset_potential']=.999*(w.potential(w.state)-phi);goals.append({'step':step,'for':goal==side})
    xs.append(x);hs.append(h[0,0].numpy().copy());rs.append(sum(component.values()));acts.append([int(m),float(z),lp]);masks.append(mask);parts.append(list(component.values()))
    for k,v in component.items():totals[k]=totals.get(k,0)+v
    for k in contacts:contacts[k]+=diag[k]
    if diag['qualified_contact']:recent.append({'step':step,'direction':'goalward' if diag['goalward'] else 'backward' if diag['backward'] else 'meaningless','position':w.state.ball.position.copy()})
    if goal is not None:
     for item in recent:
      if step-item['step']<=300:post.append({'direction':item['direction'],'seconds_to_goal':(step-item['step'])/60,'goal_for':goal==side})
     recent=[]
    recent=[v for v in recent if step-v['step']<=300]
   nx=w.observe(w.our);nl,ns,nh=p(torch.tensor(nx[None]),h);boundary=float(old(torch.tensor(nx)));returns=np.zeros(4096,np.float32);v=boundary
   for j in range(4095,-1,-1):v=rs[j]+.999*v;returns[j]=v
   x=torch.tensor(np.array(xs));l,s,final,mem=sequence(p,x);torch.testing.assert_close(mem,torch.tensor(np.array(hs)),rtol=1e-5,atol=1e-6);torch.testing.assert_close(final,h,rtol=1e-5,atol=1e-6);a=torch.tensor(acts);torch.testing.assert_close(density(l,s,a[:,0].long(),a[:,1]),a[:,2],rtol=1e-4,atol=1e-4)
  return {'x':np.array(xs[:2048]),'memory':np.array(hs[:2048]),'returns':returns[:2048],'rewards':np.array(rs,np.float32),'components':np.array(parts,np.float32),'mask':np.array(masks[:2048]),'meta':{'index':index,'seed':seed,'pair':pair,'side':side,'stage':stage,'opponent':str(POOL[which].relative_to(ROOT)),'components_sum':totals,'contacts':contacts,'goals':goals,'postcontact_goals_within5sec_proxy':post,'boundary_common_oldcritic':boundary,'memory_and_behavior_alignment':'PASS'}}
 finally:w.close()

def main():
 out=ROOT/'runs/memory-selective-diagnostic';assert not out.exists();out.mkdir();start=time.monotonic();rows=[]
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  fs=[pool.submit(collect,i) for i in range(40)]
  for i,f in enumerate(fs):rows.append(f.result());print('CRITIC DATA',i+1,rows[-1]['meta']['stage'],rows[-1]['meta']['contacts'],flush=True)
 np.savez_compressed(out/'data.npz',**{k:np.stack([r[k] for r in rows]) for k in ['x','memory','returns','rewards','components','mask']});(out/'records.json').write_text(json.dumps([r['meta'] for r in rows],indent=2)+'\n');torch.set_num_threads(2)
 train_x=torch.tensor(np.stack([r['x'] for r in rows[:24]])).reshape(-1,13);train_h=torch.tensor(np.stack([r['memory'] for r in rows[:24]])).reshape(-1,64);train_y=torch.tensor(np.stack([r['returns'] for r in rows[:24]])).flatten();test_x=torch.tensor(np.stack([r['x'] for r in rows[24:]])).reshape(-1,13);test_h=torch.tensor(np.stack([r['memory'] for r in rows[24:]])).reshape(-1,64);test_y=torch.tensor(np.stack([r['returns'] for r in rows[24:]])).flatten();mean=float(train_y.mean());scale=max(.1,float(train_y.std()));results=[];errors={k:[] for k in ['observation','memory','capacity']}
 for seed in [0,1,2]:
  torch.manual_seed(9021+seed);base=Critic();models={'observation':base,'memory':MemoryCritic(),'capacity':CapacityCritic()}
  with torch.no_grad():
   models['memory'].net[0].weight[:,:13].copy_(base.net[0].weight);models['memory'].net[0].weight[:,13:].zero_();models['memory'].net[0].bias.copy_(base.net[0].bias)
   for layer in [2,4]:models['memory'].net[layer].load_state_dict(base.net[layer].state_dict())
  for name,model in models.items():
   torch.manual_seed(9177+seed);opt=torch.optim.Adam(model.parameters(),lr=.002)
   for it in range(400):
    ids=torch.randint(len(train_y),(2048,));pred=model(train_x[ids],train_h[ids]) if name=='memory' else model(train_x[ids]);loss=((pred-train_y[ids])/scale).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   with torch.no_grad():pred=model(test_x,test_h) if name=='memory' else model(test_x);e=(pred-test_y).numpy();mse=float(np.mean(e**2));ev=1-float(np.var(e))/float(test_y.var(unbiased=False));per=(e**2).reshape(16,2048).mean(1);errors[name].append(per);results.append({'seed':seed,'name':name,'parameters':sum(p.numel() for p in model.parameters()),'heldout_MSE':mse,'heldout_EV':ev});print('CRITIC FIT',results[-1],flush=True)
   if seed==0:torch.save({'critic':model.state_dict(),'optimizer':opt.state_dict(),'fit_steps':400,'normalization_scale':scale,'training_seed':seed,'features':name},out/(name+'.pt'))
 current=Critic();current.load_state_dict(torch.load(OLD,weights_only=True)['critic'],strict=True)
 with torch.no_grad():e=(current(test_x)-test_y).numpy();snapshot={'name':'preserved_previous_critic_no_refit','heldout_MSE':float(np.mean(e**2)),'heldout_EV':1-float(np.var(e))/float(test_y.var(unbiased=False))}
 boot=np.random.default_rng(9292);comparison={};accepted=True
 for control in ['observation','capacity']:
  mem=np.asarray(errors['memory']);ctrl=np.asarray(errors[control]);difference=(mem-ctrl).mean(0).reshape(8,2).mean(1);samples=difference[boot.integers(0,8,(10000,8))].mean(1);ci=np.quantile(samples,[.025,.975]).tolist();improvement=1-float(mem.mean()/ctrl.mean());wins=int(np.sum(mem.mean(1)<ctrl.mean(1)));comparison[control]={'relative_MSE_reduction':improvement,'fitting_seed_wins':wins,'paired_seed_bootstrap_MSE_difference_95':ci};accepted &= improvement>=.1 and wins>=2 and ci[1]<0
 # Quantify old all-state anchor influence outside the newly declared approach mask.
 actor=Policy();actor.load_state_dict(torch.load(GEO,weights_only=True)['state_dict']);x=torch.tensor(rows[0]['x']);mask=torch.tensor(rows[0]['mask']);l,s,_,_=sequence(actor,x);logits=l.detach().clone().requires_grad_();old_ce=torch.nn.functional.cross_entropy(logits,torch.ones(len(x),dtype=torch.long));old_gradient=torch.autograd.grad(old_ce,logits)[0];new_logits=l.detach().clone().requires_grad_();ce,se,kl=selective_terms(new_logits,s.detach(),s.detach(),x,mask);new_grad=torch.autograd.grad(.01*ce+.1*se+.02*kl,new_logits)[0]
 anchors={'approach_fraction':float(mask.float().mean()),'outside_approach_old_CE_gradient_norm':float(old_gradient[~mask].norm()),'outside_approach_new_anchor_mode_gradient_norm':float(new_grad[~mask].norm()),'old_KL_all_states':'At reference exact equality KL is0; once mode distribution differs categorical KL penalizes all states, including post-contact/recovery. Removed from candidate loss only, still monitored.'}
 result={'accepted_memory':bool(accepted),'comparisons':comparison,'fits':results,'current_snapshot':snapshot,'anchors':anchors,'elapsed_seconds':time.monotonic()-start,'real_frozen_policy_decisions':40*4096,'actor_updates':0,'fitted_target_samples':24*2048,'heldout_target_samples':16*2048,'target_variance':float(test_y.var(unbiased=False)),'target_caveat':'Discounted truncated Monte Carlo plus identical preserved critic boundary bootstrap; target proxy, not exact infinite-horizon value. Only firsthalf assessed; side-pair clustered bootstrap, eight independent holdout seed pairs. No match evidence or architecture changes inferred from regression.'};(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('DIAG COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
