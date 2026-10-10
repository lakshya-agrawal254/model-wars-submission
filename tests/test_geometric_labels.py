import unittest,sys,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training/geometric'),str(ROOT/'src'),str(ROOT/'training')]
from labels import targets
from world import World
from policy import Policy
from runtime import decode
from rlgym.rocket_league.state_mutators import KickoffMutator
class GeometricTests(unittest.TestCase):
 def test_labels_only_official_features_and_mirrored_direction(self):
  for angle in [-2.8,-.7,-.2,0,.2,.7,2.8]:
   x=np.zeros(13,np.float32);x[:2]=[math.cos(angle),math.sin(angle)];x[8]=1;m,s=targets(x)
   self.assertEqual(int(m),1);self.assertAlmostEqual(float(s),float(np.clip(2*angle,-1,1)),places=5)
   y=x.copy();y[1]*=-1;self.assertAlmostEqual(float(targets(y)[1]),-float(s),places=5)
   y=x.copy();y[[3,4,5,9,10,11]]=[100,-200,50,1,1,2];self.assertEqual(int(targets(y)[0]),int(m));self.assertEqual(float(targets(y)[1]),float(s))
  x[8]=0;self.assertEqual(int(targets(x)[0]),0);self.assertEqual(float(targets(x)[1]),0.)
  with self.assertRaises(AssertionError):targets(np.zeros(18))
 def test_real_steering_sign_and_original_controls(self):
  for sign in [-1,1]:
   w=World(4401,'G',0)
   try:
    state=w.state;c=state.cars[w.our];c.physics.position=np.array([0,0,17],np.float32);c.physics.euler_angles=np.zeros(3,np.float32);state.ball.position=np.array([1000,sign*700,93.15],np.float32);w.state=w.engine.set_state(state,{})
    x=w.observe(w.our);before=math.atan2(x[1],x[0]);m,s=targets(x);controls=decode(int(m),float(s));self.assertEqual(controls[0],1);self.assertEqual(controls[2:],[0.,0.,0.,False,False,False])
    for i in range(30):w.step(controls,decode(0,0.))
    y=w.observe(w.our);self.assertLess(abs(math.atan2(y[1],y[0])),abs(before))
   finally:w.close()
 def test_all_five_exact_kickoffs_both_sides(self):
  for side in [0,1]:
   for i in range(5):
    w=World(41000+i,'B',side,i)
    try:
     for c in w.state.cars.values():np.testing.assert_allclose(c.physics.position,(KickoffMutator.SPAWN_BLUE_POS if c.team_num==0 else KickoffMutator.SPAWN_ORANGE_POS)[i],atol=1e-3)
     x=w.observe(w.our);self.assertEqual(x.shape,(13,));self.assertTrue(np.isfinite(x).all());self.assertEqual(int(targets(x)[0]),1)
    finally:w.close()
 def test_gru_full_sequence_matches_callbacks_exact_architecture(self):
  torch.manual_seed(4110011);p=Policy().eval();x=torch.randn(32,13);h=None;out=[]
  with torch.inference_mode():
   l,s,end=p(x)
   for row in x:
    q,t,h=p(row[None,:],h);out.append((q,t))
  torch.testing.assert_close(l,torch.cat([a[0] for a in out]),rtol=1e-5,atol=1e-6);torch.testing.assert_close(s,torch.cat([a[1] for a in out]),rtol=1e-5,atol=1e-6);torch.testing.assert_close(end,h,rtol=1e-5,atol=1e-6);self.assertEqual(sum(q.numel() for q in p.parameters()),26181)
 def test_trained_candidate_reload_forecast_invariance_and_original_state(self):
  source=ROOT/'runs/geometric-supervised-v1/latest.pt'
  if not source.exists():self.skipTest('Complete supervised archive is separate from Windows inference ZIP')
  saved=torch.load(source,map_location='cpu',weights_only=True);p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);optimizer=torch.optim.Adam(p.parameters(),lr=.001);optimizer.load_state_dict(saved['supervised_optimizer']);self.assertTrue(optimizer.state)
  self.assertEqual(saved['updates'],176);self.assertEqual(saved['new_ppo_decisions'],0);self.assertEqual(saved['initial_ppo_lineage_decisions'],16384)
  candidate=Policy().eval();candidate.load_state_dict(torch.load(ROOT/'model/V13.pt',weights_only=True)['state_dict'],strict=True)
  x=torch.randn(40,13);x[:,[3,4,5,9,10,11]]=0;y=x.clone();y[:,[3,4,5,9,10,11]]=torch.randn(40,6)*100
  with torch.inference_mode():
   outputs=p(x);shipped=candidate(x);stress=p(y)
  for a,b,c in zip(outputs,shipped,stress):self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a,c));self.assertTrue(torch.isfinite(a).all())
if __name__=='__main__':unittest.main(verbosity=2)
