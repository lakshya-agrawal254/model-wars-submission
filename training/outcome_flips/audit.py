"""No optimizer updates: actual prior loss/gradient/margin decomposition."""
import sys,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src')]
from policy import Policy
out=ROOT/'runs/outcome-flip-audit-v1';assert not out.exists();out.mkdir();torch.set_num_threads(1)
z=np.load(ROOT/'runs/flip-learning-v1/data.npz');rows=json.loads((ROOT/'runs/flip-learning-v1/demonstrations.json').read_text());g=Policy();g.load_state_dict(torch.load(ROOT/'runs/geometric-retained/latest.pt',weights_only=True)['state_dict']);cases=[]
for i,r in enumerate(rows):
 if r['flip_events'] and r['first_contact'] is not None:
  x=torch.tensor(z['x'+str(i)]);m=torch.tensor(z['m'+str(i)]);event=(m==2)|(m==3);release=torch.zeros_like(event)
  for n in torch.where(m==3)[0]:release[max(0,int(n)-int(.1/(float(x[n,12])/60))):int(n)]=True
  learn=event|release
  with torch.no_grad():rl,rs,_=g(x)
  cases.append((x,m,learn,rl,rs))
reports=[]
for name,path in [('geometric',ROOT/'runs/geometric-retained/latest.pt'),('prior8',ROOT/'runs/flip-learning-v1/latest.pt')]:
 p=Policy();p.load_state_dict(torch.load(path,weights_only=True)['state_dict']);terms={'event_CE':[],'weighted_outside_KL':[],'weighted_steering':[]};margins={2:[],3:[]};pr={2:[],3:[]};counts={i:0 for i in range(4)};totalrows=0
 for x,m,learn,rl,rs in cases:
  l,s,h=p(x);terms['event_CE'].append(F.cross_entropy(l[learn],m[learn]));terms['weighted_outside_KL'].append(.5*F.kl_div(F.log_softmax(l[~learn],-1),F.softmax(rl[~learn],-1),reduction='batchmean'));terms['weighted_steering'].append(5*(s-rs).square().mean());totalrows+=len(x)
  for k in range(4):counts[k]+=int((m[learn]==k).sum())
  for k in [2,3]:
   mask=m==k;margins[k].extend((l[mask].max(-1).values-l[mask,k]).detach().tolist());pr[k].extend(l[mask].softmax(-1)[:,k].detach().tolist())
 losses={k:torch.stack(v).mean() for k,v in terms.items()};grads={}
 for k,v in losses.items():
  grad=torch.autograd.grad(v,list(p.parameters()),retain_graph=True,allow_unused=True);grads[k]=torch.cat([torch.zeros_like(param).flatten() if a is None else a.flatten() for param,a in zip(p.parameters(),grad)]).detach()
 cos=lambda a,b:float(F.cosine_similarity(a[None],b[None])) if float(a.norm()*b.norm())>1e-10 else None
 reports.append(dict(model=name,losses={k:float(v.detach()) for k,v in losses.items()},gradient_norms={k:float(v.norm()) for k,v in grads.items()},CE_vs_KL_cosine=cos(grads['event_CE'],grads['weighted_outside_KL']),CE_vs_steering_cosine=cos(grads['event_CE'],grads['weighted_steering']),learn_mask_counts=counts,total_sequence_rows=totalrows,mean_Chase_minus_target_logit_gap={str(k):float(np.mean(v)) for k,v in margins.items()},mean_target_probability={str(k):float(np.mean(v)) for k,v in pr.items()}))
(out/'decomposition.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
