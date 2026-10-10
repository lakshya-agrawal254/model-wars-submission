"""Complete strict restore, genuine13D data, sequence parity and unmodified actor."""
from world import *
import sys,json,hashlib,random
from pathlib import Path
path=Path(sys.argv[1]);out=path.parent;saved=torch.load(path,weights_only=True);p=load(path);torch.set_num_threads(1);parameters=list(p.encoder.parameters())+list(p.gru.parameters())+list(p.head.parameters());opt=torch.optim.Adam([{'params':parameters[:6]},{'params':parameters[6:]}]);opt.load_state_dict(saved['supervised_optimizer']);assert len(opt.state)==8;assert sum(a.numel() for a in p.parameters())==26181;assert len(p.state_dict())==8
for state in opt.state.values():
 for v in state.values():assert not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
for v in p.state_dict().values():assert torch.isfinite(v).all()
assert saved['data_sha256']==hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()
for name,h in saved['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
rng=saved['numpy_rng'];np.random.set_state((rng['bit_generator'],np.array(rng['keys'],np.uint32),rng['pos'],rng['has_gauss'],rng['cached_gaussian']));torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);random.Random().setstate(saved['sampler_rng'])
z=np.load(out/'data.npz');rows=json.loads((out/'data-records.json').read_text());g=load();x=torch.tensor(z['x0']);h=None;ls=[];ss=[]
assert all(z[k].shape[1]==13 and np.isfinite(z[k]).all() for k in z.files if k.startswith('x'))
for i,r in enumerate(rows):
 a=z['x'+str(i)];expected=60/r['record']['rate'];assert np.allclose(a[1:,12],expected,atol=1e-5)
assert torch.equal(p.encoder[0].weight[:,[3,4,5,9,10,11]],g.encoder[0].weight[:,[3,4,5,9,10,11]])
with torch.inference_mode():
 a=p(x)
 for row in x:
  l,s,h=p(row[None],h);ls.append(l);ss.append(s)
 assert torch.allclose(a[0],torch.cat(ls),atol=2e-6);assert torch.allclose(a[1],torch.cat(ss),atol=2e-6);assert torch.allclose(a[2],h,atol=2e-6)
 assert torch.equal(p(x)[2],p(x)[2])
 # No hiddenreset atgoal: chunking complete inputs withhcarry equalsfull chronological forward.
 split=len(x)//2;left=p(x[:split]);right=p(x[split:],left[2]);assert torch.allclose(right[2],a[2],atol=2e-6)
assert [decode(i,.4) for i in range(4)]==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]]
r=dict(status='PASS',checkpoint=str(path),parameters=26181,observations=13,Adam_parameter_states=8,Adam_step_counts=[float(s['step']) for s in opt.state.values()],updates=saved['updates'],total_curriculum_updates=saved.get('total_curriculum_updates',saved['updates']),authentic_callback_features=True,finite_13D_data=True,chronological_single_step_and_chunkcarry_parity=True,deterministic_repeat=True,no_new_actor_parameters=True,original_decoder=True,new_PPO_decisions=0,native_RocketLeague='UNTESTED');(out/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
