"""Strict restore and parity; weights/saveddata/source provenance checked."""
from world_ground import *
import hashlib,random
OUT=ROOT/'runs/ground-trajectory-learning-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 path=Path(sys.argv[1]);saved=torch.load(path,weights_only=True);p=load(path);opt=torch.optim.Adam(p.parameters());opt.load_state_dict(saved['supervised_optimizer']);assert len(opt.state)==8
 for name,state in opt.state.items():
  for v in state.values():assert not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
 assert sum(v.numel() for v in p.parameters())==26181 and len(p.state_dict())==8
 assert sha(OUT/'data.npz')==saved['data_sha256']
 for name,h in saved['source_sha256'].items():assert sha(OUT/'used-source'/name)==h,name
 assert sha(START)==saved['initial_checkpoint_sha256']
 rng=saved['numpy_rng'];np.random.set_state((rng['bit_generator'],np.asarray(rng['keys'],np.uint32),rng['pos'],rng['has_gauss'],rng['cached_gaussian']));torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);random.Random().setstate(saved['sampler_rng'])
 z=np.load(OUT/'data.npz');allx=[z[k] for k in z.files if k.startswith('x')];assert all(x.shape[1]==13 and np.isfinite(x).all() for x in allx);assert all(np.allclose(x[1:,12],1,atol=1e-5) for x in allx)
 x=torch.tensor(allx[0]);torch.set_num_threads(1)
 with torch.inference_mode():
  whole=p(x);h=None;l=[];s=[]
  for row in x:a,b,h=p(row[None],h);l.append(a);s.append(b)
  assert torch.allclose(whole[0],torch.cat(l),atol=2e-6);assert torch.allclose(whole[1],torch.cat(s),atol=2e-6);assert torch.allclose(whole[2],h,atol=2e-6)
  assert torch.equal(p(x)[2],whole[2]);split=len(x)//2;a=p(x[:split]);b=p(x[split:],a[2]);assert torch.allclose(b[2],whole[2],atol=2e-6)
  altered=x.clone();altered[:,3:6]=0;altered[:,9:12]=0;alt=p(altered);delta=(whole[1]-alt[1]).abs();report=dict(status='PASS',checkpoint=str(path),updates=saved['updates'],parameters=26181,observations=13,Adam_states=8,source_snapshot_checksums=True,data_checksum=True,finite=True,whole_single_step_chunkcarry_parity=True,deterministic=True,original_decoder=True,prediction_input_mean_abs_steer_change=float(delta.mean()),prediction_input_max_abs_steer_change=float(delta.max()),prediction_input_mode_change_fraction=float((whole[0].argmax(-1)!=alt[0].argmax(-1)).float().mean()),prediction_columns_norm=float(p.encoder[0].weight[:,[3,4,5,9]].norm()),native='UNTESTED',new_PPO_decisions=0)
 assert torch.equal(p.encoder[0].weight[:,[10,11]],load(START).encoder[0].weight[:,[10,11]])
 assert decode(0,0)==[0.,0.,0.,0.,0.,False,False,False] and decode(1,.3)==[1.,.3,0.,0.,0.,False,False,False] and decode(2,0)[5] and decode(3,0)[2]==-1
 (OUT/f'verification-{saved["updates"]}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
