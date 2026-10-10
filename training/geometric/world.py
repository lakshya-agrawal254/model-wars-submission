"""New real RocketSim starting-state distributions; runtime untouched."""
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training'))
from sim_contract import World as BaseWorld
from rlgym.rocket_league.state_mutators import KickoffMutator
class World(BaseWorld):
 def __init__(self,seed,stage='E',team=0,kickoff_index=None):
  self.kickoff_index=kickoff_index;super().__init__(seed,stage,team)
 def reset(self,stage,team,tick):
  super().reset('E' if stage in ['B','G','M'] else stage,team,tick)
  if stage in ['G','M']:
   r=self.rng;c=self.state.cars[self.our];yaw=r.uniform(-math.pi,math.pi);angle=r.uniform(-math.pi,math.pi);distance=r.uniform(350,2200)
   origin=np.array([r.uniform(-700,700),r.uniform(-1300,1300),17],np.float32);c.physics.position=origin;c.physics.euler_angles=np.array([0,yaw,0],np.float32)
   self.state.ball.position=origin+np.array([distance*math.cos(yaw+angle),distance*math.sin(yaw+angle),76.15],np.float32)
   if stage=='M':
    self.state.ball.linear_velocity=np.array([r.uniform(-450,450),r.uniform(-450,450),0],np.float32)
    speed=r.uniform(0,700);c.physics.linear_velocity=np.array([speed*math.cos(yaw),speed*math.sin(yaw),0],np.float32)
   self.state=self.engine.set_state(self.state,{});self.kickoff=False
  elif stage=='B':
   i=int(self.rng.integers(5)) if self.kickoff_index is None else self.kickoff_index
   assert 0<=i<5
   for c in self.state.cars.values():
    blue=c.team_num==0;c.physics.position=(KickoffMutator.SPAWN_BLUE_POS if blue else KickoffMutator.SPAWN_ORANGE_POS)[i].copy();yaw=(KickoffMutator.SPAWN_BLUE_YAW if blue else KickoffMutator.SPAWN_ORANGE_YAW)[i];c.physics.euler_angles=np.array([0,yaw,0],np.float32)
   self.state=self.engine.set_state(self.state,{});self.kickoff=True
