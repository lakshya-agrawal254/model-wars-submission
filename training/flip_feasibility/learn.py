"""One bounded proof-of-learning. Legal training-only sequences; exact original actor."""
from proof import *
import hashlib,random,copy
from torch.nn import functional as F
OUT=ROOT/'runs/flip-learning-v1'
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(p,seed):
 rows=[]
 for rate in [60,120]:
  for side in [0,1]:
   for kind,index in [('nearball',4)]+[('kickoff',i) for i in range(5)]:
    r,*_=trial(rate,side,kind,'geometric',seed=seed+index*1009,index=index,model=p);r.pop('trace');rows.append(r)
 return rows

def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1);torch.manual_seed(9860011);np.random.seed(9860011);random.seed(9860011);g=load();p=load();initial=digest(GEO);data={};records=[];selected=[]
 for rate in [60,120]:
  for side in [0,1]:
   for kind,index in [('nearball',4)]+[('kickoff',i) for i in range(5)]:
    r,x,m,s=trial(rate,side,kind,'release',seed=9840011+index*1009,index=index);key=str(len(records));data['x'+key]=x;data['m'+key]=m;data['s'+key]=s;records.append(r)
    if r['flip_events'] and r['first_contact'] is not None:selected.append(key)
 np.savez_compressed(OUT/'data.npz',**data);dump('demonstrations.json',records);print('successful demonstrations',len(selected),'of',len(records),flush=True)
 baseline=evaluate(g,9850011);dump('baseline-diagnostics.json',baseline)
 if not selected:dump('result.json',{'rejected':'No successful demonstrations; no learning'});return
 opt=torch.optim.Adam(p.parameters(),lr=.0001);start=torch.load(GEO,weights_only=True);opt.load_state_dict(start['supervised_optimizer'])
 for group in opt.param_groups:group['lr']=.0001
 seq=[]
 with torch.inference_mode():
  for key in selected:
   x=torch.tensor(data['x'+key]);l,s,h=g(x);seq.append((x,torch.tensor(data['m'+key]),l.detach().clone(),s.detach().clone()))
 history=[];began=time.monotonic()
 for update in range(8):
  opt.zero_grad();losses=[]
  for x,m,rl,rs in seq:
   l,s,h=p(x);event=(m==2)|(m==3);release=torch.zeros_like(event)
   for n in torch.where(m==3)[0]:release[max(0,int(n)-int(0.1/(float(x[n,12])/60))):int(n)]=True
   learn=event|release
   # Local legal sequence supervision, own geometric mode anchor everywhere else.
   ce=F.cross_entropy(l[learn],m[learn]);outside=~learn
   kl=F.kl_div(F.log_softmax(l[outside],-1),F.softmax(rl[outside],-1),reduction='batchmean');steer=(s-rs).square().mean();losses.append(ce+.5*kl+5*steer)
  loss=torch.stack(losses).mean();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),.5));assert np.isfinite(norm);opt.step();history.append({'update':update+1,'loss':float(loss.detach()),'gradient_norm':norm});print('UPDATE',history[-1],flush=True)
  # Preserve each checkpoint, complete compatible optimizer and RNG.
  saved=dict(state_dict=p.state_dict(),supervised_optimizer=opt.state_dict(),configuration=json.loads((ROOT/'training/flip_feasibility/protocol.json').read_text()),updates=update+1,initial_checkpoint_sha256=initial,data_sha256=digest(OUT/'data.npz'),torch_rng=torch.get_rng_state(),numpy_rng=str(np.random.get_state()),python_rng=str(random.getstate()),history=history,new_ppo_decisions=0)
  torch.save(saved,OUT/f'update-{update+1}.pt')
 torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');p.eval();candidate=evaluate(p,9850011);dump('candidate-diagnostics.json',candidate)
 def agg(rows):return dict(trials=len(rows),contacts=sum(r['first_contact'] is not None for r in rows),flips=sum(len(r['flip_events']) for r in rows),stationary_seconds=sum(r['stationary_seconds'] for r in rows),airborne_seconds=sum(r['airborne_seconds'] for r in rows),kickoffs={str(hz):dict(contacts=sum(r['first_contact'] is not None for r in rows if r['kind']=='kickoff' and r['rate']==hz),trials=10) for hz in [60,120]})
 a,b=agg(baseline),agg(candidate);reasons=[]
 if b['flips']==0:reasons.append('No deterministic learned dodge events')
 if b['contacts']<a['contacts']:reasons.append('Reduced contact reliability')
 if b['stationary_seconds']>a['stationary_seconds']+2:reasons.append('Stationary regression')
 dump('result.json',dict(baseline=a,candidate=b,reject_reasons=reasons,optimizer_updates=8,new_ppo_decisions=0,training_seconds=time.monotonic()-began,selected_demonstrations=len(selected),candidate_sha256=digest(OUT/'candidate.pt'),native_game='UNTESTED',promotion='NONE'));print('RESULT',a,b,reasons,flush=True)
 assert digest(GEO)==initial
if __name__=='__main__':main()
