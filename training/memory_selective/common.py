"""Training-only memory features and independent, selectively scoped anchors."""
from pathlib import Path
import sys,importlib.util
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training')]
from policy import Policy
from ppo import Critic
spec=importlib.util.spec_from_file_location('preserved_anchored_world',ROOT/'training/anchored_final/world.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class MemoryCritic(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(77,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
 def forward(self,x,memory):return self.net(torch.cat((x,memory.detach()),-1)).squeeze(-1)
class CapacityCritic(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(13,128),nn.Tanh(),nn.Linear(128,56),nn.Tanh(),nn.Linear(56,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def sequence(actor,x,h=None):
 z,final=actor.gru(actor.encoder(x).unsqueeze(0),h);memory=z.squeeze(0);out=actor.head(memory);return out[:,:4],torch.tanh(out[:,4]),final,memory.detach()
def density(l,s,m,z,std=.15):
 from torch.distributions import Categorical,Normal
 return Categorical(logits=l).log_prob(m)+(m==1)*(Normal(torch.atanh(s.clamp(-1+1e-6,1-1e-6)),std).log_prob(z)-torch.log(1-torch.tanh(z).square()+1e-6))
def target(x):return torch.clamp(2*torch.atan2(x[...,1],x[...,0]),-1,1)
def selective_terms(l,s,ref_s,x,mask):
 zero=l.sum()*0+s.sum()*0
 if not mask.any():return zero,zero,zero
 ce=nn.functional.cross_entropy(l[mask],torch.ones(int(mask.sum()),dtype=torch.long));mse=(s[mask]-target(x)[mask]).square().mean();aligned=mask & ((ref_s-target(x)).abs()<.2)
 kl=(torch.atanh(s[aligned].clamp(-1+1e-6,1-1e-6))-torch.atanh(ref_s[aligned].clamp(-1+1e-6,1-1e-6))).square().mean()/(2*.15**2) if aligned.any() else zero
 return ce,mse,kl
class World(module.World):
 def reset(self,stage,team,tick):
  super().reset('E' if stage=='R' else stage,team,tick)
  if stage=='R':
   r=self.rng;c=self.state.cars[self.our];c.physics.position=np.array([r.uniform(-1200,1200),r.uniform(-2200,2200),r.uniform(180,420)],np.float32);c.physics.euler_angles=np.array([r.uniform(-.6,.6),r.uniform(-np.pi,np.pi),r.choice([-1,1])*r.uniform(1.7,2.8)],np.float32);c.wheels_with_contact=(False,False,False,False);c.physics.angular_velocity=np.array([r.uniform(-1,1),r.uniform(-1,1),r.uniform(-1,1)],np.float32);self.state.ball.position=np.array([r.uniform(-1500,1500),r.uniform(-2500,2500),93.15],np.float32);self.state=self.engine.set_state(self.state,{});self.kickoff=False
  self.either_touch_tick=tick-1000
 def approach_mask(self,x):
  car=self.state.cars[self.our];ground=bool(car.on_ground) and car.physics.position[2]<50;distance=float(np.linalg.norm(self.state.ball.position-car.physics.position));return bool(x[8]>.5 and ground and self.state.ball.position[2]<160 and x[0]>0 and abs(np.arctan2(x[1],x[0]))<.65 and 300<distance<2500 and self.state.tick_count-self.either_touch_tick>120)
 def components(self,old,touch,goal,ticks,gamma):
  team=self.state.cars[self.our].team_num;sign=1 if team==0 else -1;parts={'goal':10*(1 if goal==team else -1) if goal is not None else 0.,'potential':gamma*self.potential(self.state)-self.potential(old),'productive_velocity':0.,'nonproductive_contact':0.,'stationary':0.,'goal_reset_potential':0.};diag={'touch':int(touch),'goalward':0,'backward':0,'meaningless':0,'qualified_contact':0}
  if touch and self.state.tick_count-self.last_touch_tick>=30:
   dv=sign*float(self.state.ball.linear_velocity[1]-old.ball.linear_velocity[1]);parts['productive_velocity']=.08*float(np.clip(dv/2300,-1,1));parts['nonproductive_contact']=-.003 if abs(dv)<40 else 0.;self.last_touch_tick=self.state.tick_count;diag.update(goalward=int(dv>40),backward=int(dv < -40),meaningless=int(abs(dv)<40),qualified_contact=1)
  if any(c.ball_touches for c in self.state.cars.values()):self.either_touch_tick=self.state.tick_count
  speed=float(np.linalg.norm(self.state.cars[self.our].physics.linear_velocity));self.stationary_ticks=self.stationary_ticks+ticks if speed<100 else 0
  if self.stationary_ticks>120:parts['stationary']=-.001*ticks/2
  return parts,diag
