import unittest,sys,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training/anchored_final'),str(ROOT/'src'),str(ROOT/'training')]
from train import logp,joint_kl,advantages,CFG
from world import World
from policy import Policy
class AnchoredTests(unittest.TestCase):
 def test_behavior_density_gradient_and_reference_KL(self):
  p=Policy();x=torch.randn(12,13);l,s,h=p(x);mode=torch.tensor([0,1,2,3]*3);z=torch.linspace(-1,1,12);lp=logp(l,s,mode,z);self.assertTrue(torch.isfinite(lp).all());(-lp.mean()).backward();self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
  self.assertTrue(torch.allclose(joint_kl(l.detach(),s.detach(),l.detach(),s.detach()),torch.zeros(12),atol=1e-7))
  with torch.no_grad():
   changed=l+.2*torch.randn_like(l);changed_s=(s+.1).clamp(-.99,.99);self.assertGreater(float(joint_kl(l,s,changed,changed_s).mean()),0)
 def test_actual_duration_discount_and_time_limit_bootstrap(self):
  row=(np.zeros(13,np.float32),1,0.,0.,2.,0.,.5,3.);a,r=advantages([row]);self.assertAlmostEqual(float(a[0]),CFG['gamma_at_60Hz']**.5*3-2,places=5)
  row=(np.zeros(13,np.float32),1,0.,0.,2.,0.,1.,3.);a,r=advantages([row]);self.assertAlmostEqual(float(a[0]),CFG['gamma_at_60Hz']*3-2,places=5)
 def test_privileged_rewards_do_not_change_actor_observation(self):
  for stage in ['O','F','E']:
   for side in [0,1]:
    w=World(551123,stage,side)
    try:
     x=w.observe(w.our);self.assertEqual(x.shape,(13,));self.assertTrue(np.isfinite(x).all());self.assertTrue(np.array_equal(x[[3,4,5,9,10,11]],np.zeros(6)))
     phi=w.potential(w.state);old,touch,goal=w.step([1,0,0,0,0,False,False,False],[0]*8,ticks=1);g=CFG['gamma_at_60Hz']**.5;reward,diag=w.competitive_reward(old,touch,None,1,g);self.assertAlmostEqual(reward,g*w.potential(w.state)-phi,places=5)
     y=w.observe(w.our);self.assertAlmostEqual(float(y[12]),.5,places=5)
    finally:w.close()
if __name__=='__main__':unittest.main(verbosity=2)
