import sys,unittest
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training/memory_selective'),str(ROOT/'src'),str(ROOT/'training')]
from common import Policy,MemoryCritic,World,sequence,selective_terms
class MemorySelectiveTests(unittest.TestCase):
 def test_post_observation_memory_alignment_and_detached_value_gradient(self):
  p=Policy();x=torch.randn(32,13);l,s,h,mem=sequence(p,x);original=p(x)
  for a,b in zip((l,s,h),original):torch.testing.assert_close(a,b)
  h1=None;features=[]
  for row in x:
   logits,steer,h1=p(row[None],h1);features.append(h1[0,0])
  torch.testing.assert_close(torch.stack(features),mem,rtol=1e-5,atol=1e-6);self.assertFalse(mem.requires_grad)
  value=MemoryCritic();value(x,mem).sum().backward();self.assertTrue(all(v.grad is None for v in p.parameters()));self.assertTrue(all(v.grad is not None for v in value.parameters()));self.assertEqual(sum(v.numel() for v in p.parameters()),26181)
 def test_selective_anchor_has_no_mode_or_steering_gradient_outside_mask(self):
  x=torch.randn(12,13);mask=torch.tensor([True]*3+[False]*9);l=torch.randn(12,4,requires_grad=True);s=torch.zeros(12,requires_grad=True);ref_s=torch.zeros(12);ce,se,kl=selective_terms(l,s,ref_s,x,mask);(.01*ce+.1*se+.02*kl).backward();self.assertTrue(torch.equal(l.grad[~mask],torch.zeros_like(l.grad[~mask])));self.assertTrue(torch.equal(s.grad[~mask],torch.zeros_like(s.grad[~mask])))
  l=torch.randn(12,4,requires_grad=True);s=torch.zeros(12,requires_grad=True);sum(selective_terms(l,s,ref_s,x,torch.zeros(12,dtype=torch.bool))).backward();self.assertEqual(float(l.grad.norm()),0);self.assertEqual(float(s.grad.norm()),0)
 def test_components_equal_preserved_current_reward_and_recovery_not_grounded(self):
  import common
  for side in [0,1]:
   a=World(1951,'E',side);b=common.module.World(1951,'E',side)
   try:
    for tick in range(12):
     old,touch,goal=a.step([1,.1,0,0,0,False,False,False],[0]*8);old_b,touch_b,goal_b=b.step([1,.1,0,0,0,False,False,False],[0]*8);parts,diag=a.components(old,touch,goal,2,.999);reward,detail=b.competitive_reward(old_b,touch_b,goal_b,2,.999);self.assertAlmostEqual(sum(parts.values()),reward,places=6)
   finally:a.close();b.close()
  w=World(1951,'R',0)
  try:self.assertGreater(w.state.cars[w.our].physics.position[2],160);self.assertFalse(w.approach_mask(w.observe(w.our)))
  finally:w.close()
 def test_new_gae_time_limit_and_first_batch_coverage(self):
  import importlib.util,json
  spec=importlib.util.spec_from_file_location('selective_trainer_contract',ROOT/'training/memory_selective/train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  for duration in [.5,1.]:
   a,ret=mod.gae(np.array([2.],np.float32),np.array([0.],np.float32),3.,duration);self.assertAlmostEqual(float(a[0]),.999**duration*3-2,places=5)
  scenarios=mod.CFG['first_batch_scenarios'];self.assertEqual(len(scenarios),12);self.assertTrue(set(['E','O','F','R','G','B0','B1','B2','B3','B4']).issubset(scenarios))
if __name__=='__main__':unittest.main(verbosity=2)
