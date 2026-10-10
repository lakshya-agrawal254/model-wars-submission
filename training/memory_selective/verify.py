"""Full state restoration and actor probability/control parity, never training."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src'),str(ROOT/'training')]
from common import Policy,Critic,MemoryCritic,sequence
from runtime import decode

def verify(folder):
 torch.set_num_threads(1);saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);q=Policy().eval();q.load_state_dict(torch.load(folder/'candidate.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);c=MemoryCritic() if saved['critic_kind']=='memory' else Critic();c.load_state_dict(saved['critic'],strict=True);ao=torch.optim.Adam(p.parameters());co=torch.optim.Adam(c.parameters());ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer']);assert len(ao.state)==8 and len(co.state)==6
 assert saved['additional_PPO_decisions']==24576 and saved['lineage_PPO_decisions']==40960 and saved['rollout_index']==12;assert sum(v.numel() for v in p.parameters())==26181;assert saved['torch_rng'].dtype==torch.uint8 and saved['python_rng'] and saved['numpy_rng'];assert len(saved['rollouts'])==12
 for model in [p,c]:assert all(torch.isfinite(v).all() for v in model.state_dict().values())
 for opt in [ao,co]:
  for row in opt.state.values():assert all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for v in row.values())
 for key in ['source_sha256','opponent_sha256']:
  for name,digest in saved[key].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 x=torch.tensor(np.load(folder/'on_policy.npz')['x'][0,:256]);y=x.clone();y[:,[3,4,5,9,10,11]]=torch.linspace(-100,100,256)[:,None]
 with torch.inference_mode():
  out=p(x);shipped=q(x);stress=p(y)
  for a,b,d in zip(out,shipped,stress):assert torch.equal(a,b) and torch.equal(a,d)
 controls=[decode(m,.4) for m in range(4)];assert controls==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]];assert all(v[6]==0 and v[7]==0 and v[3]==0 and v[4]==0 for v in controls)
 return {'status':'PASS','parameters':26181,'features':13,'additional_PPO_decisions':24576,'lineage_PPO_decisions':40960,'critic_kind':saved['critic_kind'],'actor_sha256':hashlib.sha256((folder/'candidate.pt').read_bytes()).hexdigest(),'inference_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in out)).hexdigest(),'checks':'strict actor/critic, both finite Adam states, RNG/config, source/opponent hashes, exact serialized inference, forecast stress invariance and original controls','native_game':'NOT TESTED'}
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1])),indent=2))
