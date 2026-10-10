"""Complete supervised-state restore and original inference/recurrent compatibility."""
import sys,json,hashlib,torch,numpy as np
from pathlib import Path
from campaign import ROOT,load,GEO,COLS
from runtime import decode
out=Path(sys.argv[1]);torch.set_num_threads(1);saved=torch.load(out/'latest.pt',weights_only=True);p=load(out/'latest.pt');q=load(out/'candidate.pt');g=load(GEO);opt=torch.optim.Adam(p.head.parameters());opt.load_state_dict(saved['head_optimizer']);assert len(opt.state)==2;assert saved['updates']==16 and saved['collected_decisions']==122400 and saved['new_ppo_decisions']==0
assert sum(v.numel() for v in p.parameters())==26181 and len(p.state_dict())==8
for name,value in p.state_dict().items():
 assert torch.isfinite(value).all()
 if not name.startswith('head.'):assert torch.equal(value,g.state_dict()[name]),name
for state in opt.state.values():
 for v in state.values():assert not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
assert saved['torch_rng'].dtype==torch.uint8 and saved['numpy_rng'] and saved['python_rng'];assert hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()==saved['data_sha256']
for name,sha in saved['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
x=torch.tensor(np.load(out/'data.npz')['x0']);stress=x.clone();stress[:,COLS]=torch.linspace(-100,100,len(x))[:,None]
with torch.inference_mode():
 a=p(x);b=q(x);c=p(stress);d=g(x);h=None;ls=[];ss=[]
 for row in x:
  l,s,h=q(row[None],h);ls.append(l);ss.append(s)
 for u,v,w in zip(a,b,c):assert torch.equal(u,v) and torch.equal(u,w)
 assert torch.equal(a[2],d[2]);assert torch.allclose(a[0],torch.cat(ls),atol=2e-6) and torch.allclose(a[1],torch.cat(ss),atol=2e-6) and torch.allclose(a[2],h,atol=2e-6)
assert [decode(i,.4) for i in range(4)]==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]]
r=dict(status='PASS',parameters=26181,features=13,tensor_names=list(p.state_dict()),optimizer_updates=saved['updates'],encoder_GRU_frozen_bit_exact=True,hidden_contract_bit_exact_on_identical_states=True,actor_sha256=hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),inference_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in a)).hexdigest(),native_game='NOT TESTED');(out/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
