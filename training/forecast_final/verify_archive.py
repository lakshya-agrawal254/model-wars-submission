"""Checksum-before-extraction and cold restore of private research backup."""
import argparse,hashlib,json,zipfile,sys
from pathlib import Path
import numpy as np,torch,random
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--archive',type=Path,required=True);a.add_argument('--sha256',required=True);a.add_argument('--destination',type=Path,required=True);args=a.parse_args();assert sha(args.archive)==args.sha256;assert not args.destination.exists()
 with zipfile.ZipFile(args.archive) as z:
  assert z.testzip() is None
  for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
  manifest=json.loads(z.read('backup-manifest.json'));assert set(z.namelist())==set(manifest['files'])|{'backup-manifest.json'}
  for name,expected in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
  z.extractall(args.destination)
 root=args.destination/'Project';sys.path[:0]=[str(root/'src'),str(root/'training')]
 from policy import Policy
 from ppo import Critic
 out=root/'runs/forecast-final-24576-v1';saved=torch.load(out/'latest.pt',weights_only=True);p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);plain=Policy().eval();plain.load_state_dict(torch.load(out/'candidate.pt',weights_only=True)['state_dict'],strict=True);assert sum(x.numel() for x in p.parameters())==26181
 ao=torch.optim.Adam(p.parameters());ao.load_state_dict(saved['actor_optimizer']);assert len(ao.state)==8 and all(int(v['step'])==4 for v in ao.state.values());c=Critic();c.load_state_dict(saved['critic']);co=torch.optim.Adam(c.parameters());co.load_state_dict(saved['critic_optimizer']);assert len(co.state)==6
 for opt in [ao,co]:
  for state in opt.state.values():
   for x in state.values():
    if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
 torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);n=json.loads((out/'numpy-resume-context.json').read_text());np.random.set_state((n['generator'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gaussian'],n['cached_gaussian']))
 assert sha(out/'on_policy.npz')==saved['on_policy_sha256'];d=np.load(out/'on_policy.npz');x=torch.tensor(d['x'][0]);torch.set_num_threads(2)
 with torch.inference_mode():
  l,s,h=p(x);ll,ss,hh=plain(x);torch.testing.assert_close(l,ll,rtol=0,atol=0);torch.testing.assert_close(s,ss,rtol=0,atol=0);ch=None;ls=[]
  for chunk in x.split(128):a,b,ch=p(chunk,ch);ls.append(a)
  torch.testing.assert_close(torch.cat(ls),l,rtol=1e-5,atol=1e-6)
 for name in ['initial.pt','update-1.pt','update-2.pt','update-3.pt','update-4.pt']:p.load_state_dict(torch.load(out/name,weights_only=True)['state_dict'],strict=True)
 for name,expected in json.loads((out/'export-source-manifest.json').read_text()).items():assert sha(root/name)==expected,name
 report=dict(archive_sha256=args.sha256,checksum_before_extraction=True,CRC=True,all_member_sha256_verified=len(manifest['files']),actor_and_all_four_checkpoints_loaded=True,actor_optimizer_states=8,actor_steps=4,critic_optimizer_states=6,torch_python_numpy_rng_restored=True,rollout_checksum=True,full_plain_inference_exact=True,recurrent_chunk_parity=True,parameters=26181,source_commit=manifest['source_commit'],native_acceptance=False,promoted=False)
 args.destination.with_suffix('.verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
