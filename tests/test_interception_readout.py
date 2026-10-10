import sys,runpy
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'training/interception_readout')]
from campaign import calibration,load,GEO,OUT
from targets import target
from forecast_world import ForecastWorld

def test_geometry_is_independent_and_physics_feasible():
 table,rows=calibration();assert all(r['yaw_delta']>0 for r in rows if r['steer']>0);base=[np.array([0.,0.,17]),np.array([1000.,0,0]),np.array([1.,0,0]),np.array([0.,1,0])]
 positive,tau=target(*base,np.array([1000.,500.,93]),np.zeros(3),1000,table);negative,_=target(*base,np.array([1000.,-500.,93]),np.zeros(3),1000,table);assert positive>0 and negative<0 and abs(positive+negative)<1e-6 and .12<=tau<=.8
 behind,_=target(*base,np.array([-1000.,500.,93]),np.zeros(3),1000,table);assert behind==1

def test_forecast_is_real_physics_and_does_not_change_actor_contract():
 w=ForecastWorld(9780011,'E',0)
 try:
  s=w.state;s.ball.position=np.array([0,0,1200],np.float32);s.ball.linear_velocity=np.array([300,200,0],np.float32)
  for i,c in enumerate(s.cars.values()):c.physics.position=np.array([-3000+i*6000,-3500,17],np.float32)
  w.state=w.engine.set_state(s,{});a=w.engine._arena;initial=a.ball.get_state().pos.as_numpy().copy();pred=a.get_ball_prediction(241,1);assert np.array_equal(initial,a.ball.get_state().pos.as_numpy());expected=pred[240].pos.as_numpy().copy();x=w.observe(w.our);assert len(x)==13 and x[9]==1 and x[10]==1 and x[11]==0 and x[12]==0;a.step(240);assert np.allclose(a.ball.get_state().pos.as_numpy(),expected,atol=.01)
 finally:w.close()

def test_encoder_GRU_and_architecture_unchanged():
 torch.set_num_threads(1);p=load(OUT/'candidate.pt');g=load(GEO);assert sum(v.numel() for v in p.parameters())==26181
 for k,v in p.state_dict().items():
  if not k.startswith('head.'):assert torch.equal(v,g.state_dict()[k])
 x=torch.tensor(np.load(OUT/'data.npz')['x0'][:101])
 with torch.inference_mode():
  y=p(x);assert torch.equal(y[2],g(x)[2]);h=None;ls=[]
  for row in x:
   l,s,h=p(row[None],h);ls.append(l)
  assert torch.allclose(y[0],torch.cat(ls),atol=2e-6) and torch.allclose(y[2],h,atol=2e-6)
 for n in ['src/policy.py','src/observation_contract.py','tools/validate_submission.py','.github/workflows/validate.yml']:assert (ROOT/n).read_bytes()==(ROOT.parent/'official'/n).read_bytes()

if __name__=='__main__':
 for name,f in list(globals().items()):
  if name.startswith('test_'):f();print('PASS',name)
