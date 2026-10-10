"""Offline integrity, labels, complete supervised state and inference restoration."""
import sys,hashlib,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from policy import Policy
from labels import targets

def verify(folder):
 torch.set_num_threads(1);manifest=json.loads((folder/'manifest.json').read_text())
 for n,sha in manifest['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==sha,n
 saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);optimizer=torch.optim.Adam(p.parameters(),lr=.001);optimizer.load_state_dict(saved['supervised_optimizer']);assert len(optimizer.state)==8
 assert saved['updates']==176 and saved['completed_epochs']==8 and saved['new_ppo_decisions']==0 and saved['real_simulation_decisions']==38400 and saved['supervised_sample_presentations']==337920
 assert all(torch.isfinite(v).all() for v in p.state_dict().values())
 for state in optimizer.state.values():assert all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for v in state.values())
 assert saved['torch_rng'].dtype==torch.uint8 and saved['python_rng'] and saved['numpy_rng'] and saved['configuration'] and saved['history']
 for n,sha in saved['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha,n
 assert hashlib.sha256((folder/'data.npz').read_bytes()).hexdigest()==saved['data_sha256'];data=np.load(folder/'data.npz');m,s=targets(data['observations']);assert np.array_equal(m,data['modes']) and np.array_equal(s,data['steering']);assert data['observations'].shape==(88,480,13)
 q=Policy().eval();q.load_state_dict(torch.load(ROOT/'model/V13.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);assert sum(v.numel() for v in q.parameters())==26181
 x=torch.tensor(data['observations'][0,:256]);y=x.clone();y[:,[3,4,5,9,10,11]]=torch.randn(256,6)*100
 with torch.inference_mode():
  out=p(x);shipped=q(x);stress=p(y)
  for a,b,c in zip(out,shipped,stress):assert torch.equal(a,b) and torch.equal(a,c)
 assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in out)).hexdigest()==manifest['inference_sha256']
 print('PASS: complete supervised optimizer/RNG/config/history, 26181-param strict load, original labels/data SHA/source SHA, inference parity and forecast-column invariance; no training')
if __name__=='__main__':verify(Path(sys.argv[1]))
