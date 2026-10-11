"""Real-state recurrent prefix/suffix likelihood and gradient checks; no weight update."""
from train import *
def main():
 torch.set_num_threads(2);p=Policy().eval();p.load_state_dict(torch.load(START,weights_only=True)['state_dict'],strict=True);critic=torch.load(ROOT/'runs/near-greedy-final-v1/latest.pt',weights_only=True)['critic'];before={k:v.clone() for k,v in p.state_dict().items()};reports=[]
 for index in [0,2,3]:
  r=collect(p.state_dict(),critic,index,77,suffix_length=256,prefix_cap=300);x=torch.tensor(r['x'],requires_grad=True);a=torch.tensor(r['actions']);mask=torch.tensor(r['ppo_mask']).bool();p.zero_grad();l,s,h,mem=sequence(p,x);l.retain_grad();lp=density(l[mask],s[mask],a[mask,0].long(),a[mask,1]);ratio=(lp-a[mask,2]).exp();torch.testing.assert_close(ratio,torch.ones_like(ratio),rtol=1e-4,atol=1e-4)
  loss=-(lp*torch.linspace(-1,1,len(lp))).mean();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in p.parameters());pref=r['meta']['greedy_prefix'];assert torch.isfinite(x.grad).all();assert not pref or float(l.grad[:pref].abs().max())==0;prefix_gradient=float(x.grad[pref-1].norm()) if pref else None
  if pref:assert prefix_gradient>0,'Suffix gradients must reach chronological prefix history'
  reports.append(dict(index=index,prefix=pref,trigger=r['meta']['trigger'],actual_probability_replay=True,mask_excludes_prefix=True,last_prefix_input_gradient=prefix_gradient,all_actor_gradients_finite=True))
 assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items());assert sum(t.numel() for t in p.parameters())==26181
 w=ForecastWorld(11799001,'E',0)
 try:
  with torch.inference_mode():
   h=None;x0=w.observe(w.our);_,_,h=p(torch.tensor(x0[None]),h);w.observe(w.enemy);w.step(decode(1,0.),decode(1,0.),2);previous=w.previous[w.our];w.reset('E',0,w.state.tick_count);assert w.previous[w.our]==previous;x1=w.observe(w.our);assert abs(float(x1[12])-1)<1e-5;_,_,h1=p(torch.tensor(x1[None]),h);_,_,h2=p(torch.tensor(np.stack([x0,x1])));torch.testing.assert_close(h1,h2,rtol=1e-5,atol=1e-6)
 finally:w.close()
 out=ROOT/'docs/evidence/failure-prefix-v1';out.mkdir(parents=True,exist_ok=True);r=dict(actor_unchanged=True,parameters=26181,manual_goal_reset_clock_and_hidden_continuity=True,real_state_tests=reports,scope='No optimizer steps or native game execution; synthetic scalar advantages only for gradient-path test. Proof data never used for training.');(out/'proof.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()
