import sys,unittest,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training/experiments/temporal_credit'),str(ROOT/'training'),str(ROOT/'src')]
from train import tensors,penalty
from world import World
from policy import Policy
from rlgym.rocket_league.state_mutators import KickoffMutator
class TemporalTests(unittest.TestCase):
 def test_delayed_reward_and_time_limit_bootstrap(self):
  rows=[(np.zeros(13,np.float32),0,0.,0.,0.,float(i==599)) for i in range(600)];rows[-1]=(*rows[-1],0.)
  short=tensors(rows,.99,.95)[4][0];long=tensors(rows,.9995,.9995)[4][0]
  self.assertGreater(float(long),float(short)*1e10);self.assertAlmostEqual(float(long),(.9995*.9995)**599,places=5)
  rows=[(np.zeros(13,np.float32),0,0.,0.,2.,0.,3.)];self.assertAlmostEqual(float(tensors(rows,.99,.95)[4][0]),.99*3-2,places=5)
 def test_all_five_standard_kickoffs_both_sides(self):
  for side in [0,1]:
   for i in range(5):
    w=World(202+i,'B',side,kickoff_index=i)
    try:
     self.assertEqual(w.kickoff_index,i)
     for car in w.state.cars.values():
      expected=(KickoffMutator.SPAWN_BLUE_POS if car.team_num==0 else KickoffMutator.SPAWN_ORANGE_POS)[i]
      np.testing.assert_allclose(car.physics.position,expected,atol=1e-3)
    finally:w.close()
 def test_fresh_starts_reproduce_for_matched_policy_trials(self):
  for stage in ['A','E']:
   for side in [0,1]:
    snapshots=[]
    for repeat in range(2):
     w=World(1713011,stage,side)
     try:
      snapshots.append((w.observe(w.our).copy(),w.state.cars[w.our].physics.position.copy(),w.state.ball.position.copy()))
     finally:w.close()
    for a,b in zip(*snapshots):np.testing.assert_array_equal(a,b)
 def test_regularization_only_existing_prediction_columns(self):
  p=Policy();penalty(p).backward();g=p.encoder[0].weight.grad
  others=[i for i in range(13) if i not in [3,4,5,9,10,11]]
  self.assertEqual(float(g[:,others].abs().sum()),0.);self.assertGreater(float(g[:,[3,4,5,9,10,11]].abs().sum()),0.)
  self.assertEqual(sum(x.numel() for x in p.parameters()),26181)
if __name__=='__main__':unittest.main(verbosity=2)
