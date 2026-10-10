import sys,unittest,copy
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training'),str(ROOT/'src')]
from policy import Policy
from runtime import decode
from ppo import Critic,collect,log_probability,update
from sim_contract import World

class PPOTests(unittest.TestCase):
 def test_actor_exact_contract_and_controls(self):
  p=Policy();self.assertEqual(sum(v.numel() for v in p.parameters()),26181)
  for mode in range(4):
   c=decode(mode,.4);self.assertFalse(c[6] or c[7]);self.assertEqual(c[3:5],[0,0])
 def test_sim_observations_finite_missing_prediction_and_clocks(self):
  w=World(77,'B',1)
  try:
   for a in w.ids:
    x=w.observe(a);self.assertEqual(x.shape,(13,));np.testing.assert_array_equal(x[3:6],0);np.testing.assert_array_equal(x[9:12],0);self.assertEqual(x[12],0)
   w.step(decode(1,0),decode(0,0))
   for a in w.ids:self.assertAlmostEqual(w.observe(a)[12],1.)
  finally:w.close()
 def test_joint_behavior_probability_and_finite_gradient(self):
  torch.manual_seed(3);p=Policy();o=copy.deepcopy(p).eval();v=Critic();rows,s=collect(p,v,o,54,'A',0,16)
  x=torch.tensor(np.array([r[0] for r in rows]));m=torch.tensor([r[1] for r in rows]);z=torch.tensor([r[2] for r in rows]);old=torch.tensor([r[3] for r in rows]);l,t,h=p(x)
  lp=log_probability(l,t,m,z);np.testing.assert_allclose(lp.detach(),old,atol=1e-5,rtol=1e-5)
  (-lp.mean()).backward();self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
  self.assertFalse(any(a is b for a in p.parameters() for b in v.parameters()))
 def test_causal_full_sequence_equals_callback_inference(self):
  p=Policy().eval();x=torch.randn(32,13);h=None;parts=[]
  with torch.no_grad():
   full=p(x)
   for row in x:
    l,s,h=p(row[None,:],h);parts.append((l,s))
  np.testing.assert_allclose(torch.cat([a[0] for a in parts]),full[0],atol=1e-6,rtol=1e-5)
  np.testing.assert_allclose(torch.cat([a[1] for a in parts]),full[1],atol=1e-6,rtol=1e-5)
class CheckpointTests(unittest.TestCase):
 def test_selected_complete_checkpoint_strict_load_and_rng(self):
  path=ROOT/'runs/retained/selected/latest.pt'
  if not path.exists():self.skipTest('Complete resumable backup is separate from Windows inference ZIP')
  saved=torch.load(path,map_location='cpu',weights_only=True)
  p=Policy();p.load_state_dict(saved['state_dict'],strict=True)
  v=Critic();v.load_state_dict(saved['critic'],strict=True)
  ao=torch.optim.Adam(p.parameters(),lr=3e-4);co=torch.optim.Adam(v.parameters(),lr=1e-3)
  ao.load_state_dict(saved['actor_optimizer']);co.load_state_dict(saved['critic_optimizer'])
  self.assertTrue(ao.state and co.state)
  self.assertEqual(saved['decisions'],16384)
  torch.set_rng_state(saved['torch_rng']);a=torch.randn(12)
  torch.set_rng_state(saved['torch_rng']);b=torch.randn(12)
  self.assertTrue(torch.equal(a,b))
  shipped=Policy();shipped.load_state_dict(torch.load(ROOT/'runs/retained/selected/candidate.pt',weights_only=True)['state_dict'],strict=True)
  x=torch.randn(20,13)
  with torch.no_grad():
   expected=p(x);actual=shipped(x)
  for a,b in zip(expected,actual):self.assertTrue(torch.equal(a,b))

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
