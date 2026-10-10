"""Cold restoration checks for an already downloaded private backup.

No network, no downloads, no learned updates; all policies strict-load stored tensors.
"""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
import torch

def restore(archive,destination):
 assert not destination.exists(),'Refusing overwrite'
 with zipfile.ZipFile(archive) as z:
  for n in z.namelist():
   p=Path(n);assert not p.is_absolute() and '..' not in p.parts
  z.extractall(destination)
 return destination

def verify(root):
 sys.path[:0]=[str(root/'src'),str(root/'training')]
 from policy import Policy
 from ppo import Critic
 manifest=json.loads((root/'runs/retained/retention-manifest.json').read_text())
 for name,digest in manifest['files'].items():
  p=root/'runs/retained'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 actors={}
 for role in ['selected','latest']:
  p=root/'runs/retained'/role/'latest.pt';s=torch.load(p,map_location='cpu',weights_only=True)
  a=Policy().eval();a.load_state_dict(s['state_dict'],strict=True);v=Critic();v.load_state_dict(s['critic'],strict=True)
  ao=torch.optim.Adam(a.parameters(),lr=3e-4);co=torch.optim.Adam(v.parameters(),lr=1e-3)
  ao.load_state_dict(s['actor_optimizer']);co.load_state_dict(s['critic_optimizer'])
  assert len(ao.state)==8 and len(co.state)==6
  assert all(torch.isfinite(x).all() for x in a.state_dict().values())
  assert all(torch.isfinite(x).all() for x in v.state_dict().values())
  for optimizer in [ao,co]:
   for state in optimizer.state.values():
    assert all(not isinstance(x,torch.Tensor) or torch.isfinite(x).all() for x in state.values())
  assert s['torch_rng'].dtype==torch.uint8 and s['python_rng'] and s['numpy_rng'] and s['history'] and s['configuration']
  actors[role]=a;print('RESTORED',role,s['decisions'],'actor/critic/both Adam/RNG/config/history')
 shipped=Policy().eval();shipped.load_state_dict(torch.load(root/'model/V13.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
 x=torch.linspace(-1,1,13*20).reshape(20,13)
 with torch.inference_mode():
  for a,b in zip(shipped(x),actors['selected'](x)):assert torch.equal(a,b),'Inference parity'
 b=Policy();b.load_state_dict(torch.load(root/'runs/retained/baseline/V13.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
 assert sum(p.numel() for p in shipped.parameters())==26181
 print('COMPLETE BACKUP RESTORATION AND EXACT INFERENCE PARITY PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();verify(restore(a.archive,a.destination))
