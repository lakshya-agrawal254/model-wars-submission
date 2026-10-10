"""Strict actor, optimizer, RNG, full-sequence and official contract verification."""
from proof import *
import hashlib
out=ROOT/'runs/flip-learning-v1';torch.set_num_threads(1);saved=torch.load(out/'latest.pt',weights_only=True);p=Policy();p.load_state_dict(saved['state_dict'],strict=True);q=Policy();q.load_state_dict(torch.load(out/'candidate.pt',weights_only=True)['state_dict']);g=load();opt=torch.optim.Adam(p.parameters());opt.load_state_dict(saved['supervised_optimizer']);assert len(opt.state)==8 and saved['updates']==8
assert len(p.state_dict())==8 and sum(v.numel() for v in p.parameters())==26181
assert saved['data_sha256']==hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()
for v in p.state_dict().values():assert torch.isfinite(v).all()
for st in opt.state.values():
 for v in st.values():assert not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
torch.set_rng_state(saved['torch_rng']);r=saved['numpy_rng'];np.random.set_state((r['bit_generator'],np.array(r['keys'],np.uint32),r['pos'],r['has_gauss'],r['cached_gaussian']));import random;random.setstate(saved['python_rng'])
x=torch.tensor(np.load(out/'data.npz')['x0']);h=None;ls=[];ss=[]
with torch.inference_mode():
 a=p(x);b=q(x)
 for row in x:
  l,s,h=q(row[None],h);ls.append(l);ss.append(s)
 for u,v in zip(a,b):assert torch.equal(u,v)
 assert torch.allclose(a[0],torch.cat(ls),atol=2e-6);assert torch.allclose(a[1],torch.cat(ss),atol=2e-6);assert torch.allclose(a[2],h,atol=2e-6)
assert hashlib.sha256(GEO.read_bytes()).hexdigest()==saved['initial_checkpoint_sha256']
assert [decode(i,.4) for i in range(4)]==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]]
report=dict(status='PASS',parameters=26181,observations=13,optimizer_parameters=8,updates=8,optimizer_step_counts=[float(st['step']) for st in opt.state.values()],full_sequence_vs_single_callback_parity=True,actor_weight_hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in saved['state_dict'].items()},geometric_unchanged=True,official_runtime='unchanged except package checkpoint identity and diagnostic-only logging',native_game='UNTESTED')
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
