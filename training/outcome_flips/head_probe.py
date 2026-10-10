"""Training-only linear capacity probe / analytic fit of EXISTING mode rows.
No added actor features, parameters, runtime controller or encoder/GRU replacement.
"""
from world import *
import json,hashlib
OUT=ROOT/'runs/outcome-flip-head-probe-v1';LEARN=ROOT/'runs/outcome-flip-learning-v1'
def fit(x,y):
 counts=np.bincount(y,minlength=3);weights=1/np.maximum(counts[y],1);a=(x.T*weights)@x+1e-6*np.eye(x.shape[1]);b=(x.T*weights)@np.eye(3)[y];return np.linalg.solve(a,b)
def measure(x,y,r):
 pred=(x@r).argmax(1);return dict(per_class={str(k):dict(n=int((y==k).sum()),accuracy=float(np.mean(pred[y==k]==k))) for k in range(3)},overall=float(np.mean(pred==y)))
def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1);p=load(LEARN/'latest.pt');data=np.load(LEARN/'data.npz');records=json.loads((LEARN/'data-records.json').read_text());xs=[];ys=[];groups=[]
 with torch.inference_mode():
  for i,r in enumerate(records):
   x=torch.tensor(data['x'+str(i)]);m=data['m'+str(i)].copy();win=data['window'+str(i)];encoded=p.encoder(x);memory,_=p.gru(encoded.unsqueeze(0));features=memory.squeeze(0).numpy();selected=win if r['role']=='positive' else ~win
   # Allchronologicalrows processed before selecting loss features; no hiddenprefix reset.
   xx=np.concatenate([features,np.ones((len(features),1))],1)[selected];yy=np.maximum(m[selected]-1,0)
   scene=dict(r['record']['scene']);scene.pop('side');scene.pop('seed');group=int(hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest()[:8],16)%3
   xs.append(xx);ys.append(yy);groups.append(np.full(len(yy),group))
 x=np.concatenate(xs);y=np.concatenate(ys);group=np.concatenate(groups);train=group!=0;r=fit(x[train],y[train]);report=dict(method='Balanced least-squares diagnostic of existing64D GRU features; whole-scene grouped headholdout, NOT independent recurrent-policy validation',ridge=1e-6,rows=len(y),training=measure(x[train],y[train],r),heldout_head=measure(x[~train],y[~train],r))
 r=fit(x,y);report['full_fit']=measure(x,y,r);report['coefficient_norm']=float(np.linalg.norm(r));gate=all(report['heldout_head']['per_class'][str(k)]['accuracy']>=([.9,.7,.7][k]) for k in range(3));report['qualifies_for_real_physics_probe']=gate
 np.savez_compressed(OUT/'fit.npz',coefficients=r);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 if gate:
  old=p.head.weight.detach().clone();bias=p.head.bias.detach().clone();scale=4.
  with torch.no_grad():
   for row,column in [(2,1),(3,2)]:
    delta=torch.tensor(r[:,column]-r[:,0],dtype=torch.float32);p.head.weight[row]=old[1]+scale*delta[:64];p.head.bias[row]=bias[1]+scale*delta[64]
  assert torch.equal(p.head.weight[0],old[0]) and torch.equal(p.head.weight[1],old[1]) and torch.equal(p.head.weight[4],old[4])
  torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');print('Saved analytic mode-row probe; still requires autonomous physical and match proof',flush=True)
if __name__=='__main__':main()
