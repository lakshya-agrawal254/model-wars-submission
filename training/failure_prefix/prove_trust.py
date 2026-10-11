"""Guard regression test on a copied actor; no saved playing weights are updated."""
from train import *
from trust_region import bounded_actor_step

def main():
 torch.set_num_threads(2);p=Policy().eval();p.load_state_dict(torch.load(START,weights_only=True)['state_dict']);opt=torch.optim.Adam(p.parameters(),lr=.00005);data=np.load(OUT/'batch-0.npz');x=torch.tensor(data['x_2'][:466]);a=torch.tensor(data['actions_2'][:466]);m=torch.tensor(data['ppo_mask_2'][:466]).bool()
 with torch.no_grad():bl,bs,_=p(x);old_l=bl[m].detach();old_z=torch.atanh(bs[m].clamp(-1+1e-6,1-1e-6)).detach()
 l,s,_=p(x);lp=density(l[m],s[m],a[m,0].long(),a[m,1]);loss=-(lp*torch.linspace(-1,1,len(lp))).mean();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1);before={k:v.clone() for k,v in p.state_dict().items()}
 def check():
  with torch.no_grad():
   l,s,_=p(x);z=torch.atanh(s[m].clamp(-1+1e-6,1-1e-6));kl=kl_divergence(Categorical(logits=old_l/CFG['mode_temperature']),Categorical(logits=l[m]/CFG['mode_temperature']))+(old_l/CFG['mode_temperature']).softmax(-1)[:,1]*(z-old_z).square()/(2*CFG['steer_std']**2);v=float(kl.mean());return dict(mean_KL=v,max_trajectory_KL=v)
 ok,bad=bounded_actor_step(p,opt,check,.1,mean_limit=1e-12,trajectory_limit=1e-12,max_attempts=1);assert not ok;assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items());assert not opt.state
 ok,good=bounded_actor_step(p,opt,check,.00005);assert ok;assert all(int(s['step'])==1 for s in opt.state.values());assert check()['mean_KL']<=.01
 # Failed proposals must also restore a NONEMPTY optimizer, including moments and step counts.
 import copy
 prior=copy.deepcopy(opt.state_dict());before={k:v.clone() for k,v in p.state_dict().items()};ok,more=bounded_actor_step(p,opt,lambda:dict(mean_KL=1.,max_trajectory_KL=1.),.1,max_attempts=3);assert not ok
 assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items());now=opt.state_dict();assert now['param_groups']==prior['param_groups']
 for key,s in prior['state'].items():
  for n,v in s.items():torch.testing.assert_close(now['state'][key][n],v,rtol=0,atol=0)
 report=dict(rejected_weight_restore_exact=True,rejected_empty_and_populated_Adam_restore_exact=True,accepted_step_count=1,discarded_proposal_calls=len(bad)+len(more)+len(good)-1,backtracking_trials=good,actual_post_update_KL_checked=True,scope='Copied-actor optimizer unit test using previously recorded real states; test actor discarded, no new training candidate or runtime changes.')
 (ROOT/'docs/evidence/failure-prefix-v1/trust-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
