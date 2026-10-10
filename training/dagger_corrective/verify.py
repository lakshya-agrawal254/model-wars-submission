"""Restore complete supervised state; assert original recurrent controls and parity."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from campaign import ROOT,load,COLS,labels
from runtime import decode
out=Path(sys.argv[1]);torch.set_num_threads(1);s=torch.load(out/'latest.pt',weights_only=True);p=load(out/'latest.pt');q=load(out/'candidate.pt');opt=torch.optim.Adam(p.parameters());opt.load_state_dict(s['supervised_optimizer']);assert len(opt.state)==8 and sum(v.numel() for v in p.parameters())==26181;assert s['updates']==11 and s['torch_rng'].dtype==torch.uint8 and s['python_rng'];assert hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()==s['data_sha256']
x=torch.tensor(np.load(out/'data.npz')['x0']);stress=x.clone();stress[:,COLS]=torch.linspace(-100,100,len(x))[:,None]
with torch.inference_mode():
 y=p(x);z=q(x);zz=p(stress);h=None;ls=[];ss=[]
 for row in x:
  l,st,h=q(row[None],h);ls.append(l);ss.append(st)
 for a,b,c in zip(y,z,zz):assert torch.equal(a,b) and torch.equal(a,c)
 assert torch.allclose(y[0],torch.cat(ls),atol=2e-6) and torch.allclose(y[1],torch.cat(ss),atol=2e-6) and torch.allclose(y[2],h,atol=2e-6)
 for state in opt.state.values():
  for v in state.values():assert not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
assert [decode(i,.4) for i in range(4)]==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]]
r=dict(status='PASS',parameters=26181,features=13,optimizer_updates=s['updates'],actor_sha256=hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),inference_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in y)).hexdigest(),native_game='NOT TESTED',checks='strict restoration, all8 Adam tensors, RNG, data hash, forecast invariance, full-prefix/callback parity, exact4decoder');(out/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
