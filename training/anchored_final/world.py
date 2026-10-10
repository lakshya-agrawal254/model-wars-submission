"""Goal-aware rewards and starts ONLY in trainer; no privileged actor features."""
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'training/geometric'),str(ROOT/'training')]
# Explicit file import avoids collision with this module's name.
import importlib.util
spec=importlib.util.spec_from_file_location('geometric_start_states',ROOT/'training/geometric/world.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class World(module.World):
 def __init__(self,seed,stage='E',team=0,kickoff_index=None):
  self.last_touch_tick=-1000;self.stationary_ticks=0;super().__init__(seed,stage,team,kickoff_index)
 def reset(self,stage,team,tick):
  super().reset('E' if stage in ['O','F'] else stage,team,tick)
  if stage in ['O','F']:
   r=self.rng;sign=1 if team==0 else -1;c=self.state.cars[self.our]
   if stage=='O':
    self.state.ball.position=np.array([r.uniform(-1500,1500),sign*r.uniform(1200,3300),93.15],np.float32)
    yaw=sign*math.pi/2+r.uniform(-.7,.7);distance=r.uniform(400,1300);c.physics.position=self.state.ball.position-np.array([distance*math.cos(yaw),distance*math.sin(yaw),76.15],np.float32);c.physics.euler_angles=np.array([0,yaw+r.uniform(-.8,.8),0],np.float32)
    self.state.ball.linear_velocity=np.array([r.uniform(-300,300),sign*r.uniform(-150,600),0],np.float32)
   else:
    self.state.ball.position=np.array([r.uniform(-750,750),-sign*r.uniform(2100,3500),93.15],np.float32);self.state.ball.linear_velocity=np.array([r.uniform(-180,180),-sign*r.uniform(450,1100),0],np.float32)
    c.physics.position=np.array([r.uniform(-700,700),-sign*r.uniform(3800,4450),17],np.float32);c.physics.euler_angles=np.array([0,sign*math.pi/2+r.uniform(-.7,.7),0],np.float32)
   self.state=self.engine.set_state(self.state,{});self.kickoff=False
  self.last_touch_tick=tick-1000;self.stationary_ticks=0
 def potential(self,state):
  team=state.cars[self.our].team_num;sign=1 if team==0 else -1;ball=state.ball.position;ours=np.array([0,-sign*5120,93.15]);theirs=-ours;theirs[2]=93.15
  goal_progress=(np.linalg.norm(ball-ours)-np.linalg.norm(ball-theirs))/10240
  ball_distance=min(1,float(np.linalg.norm(ball-state.cars[self.our].physics.position)/6000))
  return .7*float(goal_progress)-.08*ball_distance
 def competitive_reward(self,old,touch,goal,ticks,gamma):
  team=self.state.cars[self.our].team_num;sign=1 if team==0 else -1;reward=gamma*self.potential(self.state)-self.potential(old);details={'goals_for':int(goal==team) if goal is not None else 0,'goals_against':int(goal!=team) if goal is not None else 0,'productive_contacts':0,'nonproductive_contacts':0}
  if goal is not None:reward+=10 if goal==team else -10
  if touch and self.state.tick_count-self.last_touch_tick>=30:
   delta=sign*float(self.state.ball.linear_velocity[1]-old.ball.linear_velocity[1]);effect=float(np.clip(delta/2300,-1,1));reward+=.08*effect
   if abs(delta)<40:reward-=.003;details['nonproductive_contacts']=1
   elif delta>40:details['productive_contacts']=1
   self.last_touch_tick=self.state.tick_count
  speed=np.linalg.norm(self.state.cars[self.our].physics.linear_velocity)
  self.stationary_ticks=self.stationary_ticks+ticks if speed<100 else 0
  if self.stationary_ticks>120:reward-=.001*ticks/2
  return float(reward),details
