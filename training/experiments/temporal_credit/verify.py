"""Offline strict restoration of bounded temporal-credit experiment artifacts."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training')]
from policy import Policy
from ppo import Critic

def verify(folder):
 manifest=json.loads((folder/'manifest.json').read_text());torch.set_num_threads(1)
 for name,sha in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
 for name in manifest['policies']:
  a=Policy().eval();s=torch.load(folder/name,map_location='cpu',weights_only=True);a.load_state_dict(s['state_dict'],strict=True);assert sum(p.numel() for p in a.parameters())==26181
  x=torch.linspace(-1,1,52*13).reshape(52,13)
  with torch.inference_mode():out=a(x)
  assert all(torch.isfinite(y).all() for y in out)
  assert hashlib.sha256(b''.join(t.numpy().tobytes() for t in out)).hexdigest()==manifest['inference_sha256'][name]
 s=torch.load(folder/manifest['resumable'],map_location='cpu',weights_only=True);a=Policy().eval();a.load_state_dict(s['state_dict'],strict=True);v=Critic();v.load_state_dict(s['critic'],strict=True)
 ao=torch.optim.Adam(a.parameters(),lr=3e-4);co=torch.optim.Adam(v.parameters(),lr=1e-3);ao.load_state_dict(s['actor_optimizer']);co.load_state_dict(s['critic_optimizer']);assert len(ao.state)==8 and len(co.state)==6
 for obj in [a.state_dict(),v.state_dict()]:assert all(torch.isfinite(t).all() for t in obj.values())
 for o in [ao,co]:
  for state in o.state.values():assert all(not isinstance(t,torch.Tensor) or torch.isfinite(t).all() for t in state.values())
 assert s['decisions']==32768 and s['torch_rng'].dtype==torch.uint8 and s['python_rng'] and s['numpy_rng'] and s['experiment_history'] and s['configuration']
 for path,sha in s['configuration']['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
 print('PASS: hashes, all five 26181-param actors, complete latest critic/both Adam/RNG/config/history/source hashes, exact reload parity; no updates')
if __name__=='__main__':verify(Path(sys.argv[1]))
