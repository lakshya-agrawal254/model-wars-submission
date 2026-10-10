"""Training/diagnostic-only all-five kickoff coverage, original runtime untouched."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'training'))
from sim_contract import World as OriginalWorld
from rlgym.rocket_league.state_mutators import KickoffMutator

class World(OriginalWorld):
 def __init__(self,seed,stage='A',team=0,kickoff_index=None):
  self.requested_kickoff_index=kickoff_index;self.kickoff_index=None
  super().__init__(seed,stage,team)
 def reset(self,stage,team,tick):
  # Never execute the legacy three-position B override.
  super().reset('E' if stage=='B' else stage,team,tick)
  if stage=='B':
   i=int(self.rng.integers(5)) if self.requested_kickoff_index is None else int(self.requested_kickoff_index)
   assert 0<=i<5;self.kickoff_index=i
   for car in self.state.cars.values():
    t=car.team_num
    car.physics.position=(KickoffMutator.SPAWN_BLUE_POS if t==0 else KickoffMutator.SPAWN_ORANGE_POS)[i].copy()
    yaw=(KickoffMutator.SPAWN_BLUE_YAW if t==0 else KickoffMutator.SPAWN_ORANGE_YAW)[i]
    car.physics.euler_angles=np.array([0,yaw,0],np.float32)
   self.state=self.engine.set_state(self.state,{});self.kickoff=True
