"""Strict complete checkpoint restoration without collecting training rollouts."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training')]
from policy import Policy
from ppo import Critic

def verify(folder):
 torch.set_num_threads(1);saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
 p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);q=Policy().eval();q.load_state_dict(torch.load(folder/'candidate.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
 c=Critic();c.load_state_dict(saved['critic'],strict=True);ao=torch.optim.Adam(p.parameters());co=torch.optim.Adam(c.parameters());ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer']);assert len(ao.state)==8 and len(co.state)==6
 assert sum(v.numel() for v in p.parameters())==26181 and saved['additional_PPO_decisions']==32768 and saved['lineage_PPO_decisions']==49152 and saved['rollout_index']==8
 for model in [p,c]:assert all(torch.isfinite(v).all() for v in model.state_dict().values())
 for optimizer in [ao,co]:
  for row in optimizer.state.values():assert all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for v in row.values())
 for key in ['source_sha256','opponent_sha256']:
  for name,digest in saved[key].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 assert hashlib.sha256((ROOT/'runs/geometric-retained/data.npz').read_bytes()).hexdigest()==saved['anchor_data_sha256']
 assert saved['torch_rng'].dtype==torch.uint8 and saved['python_rng'] and saved['numpy_rng'] and len(saved['history'])==2
 data=np.load(ROOT/'runs/geometric-retained/data.npz');x=torch.tensor(data['observations'][0,:256]);stress=x.clone();stress[:,[3,4,5,9,10,11]]=torch.linspace(-50,50,256)[:,None];out=[]
 with torch.inference_mode():
  for a,b,d in zip(p(x),q(x),p(stress)):assert torch.equal(a,b) and torch.equal(a,d);out.append(a)
 return {'status':'PASS complete model/critic/both optimizers/RNG/source/pool/anchor/inference restoration; no training','additional_PPO_decisions':32768,'lineage_PPO_decisions':49152,'parameters':26181,'inference_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in out)).hexdigest(),'model_sha256':hashlib.sha256((folder/'candidate.pt').read_bytes()).hexdigest()}
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1])),indent=2))
