"""Training/evaluation-only real RocketSim bridge. No native forecasts fabricated."""
from pathlib import Path
import sys,random,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from observation_contract import build,basis
from runtime import decode
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator,KickoffMutator

class World:
 def __init__(self,seed,stage='E',team=0):
  self.training_stage=stage;self.seed=seed;self.rng=np.random.default_rng(seed);random.seed(seed);self.engine=RocketSimEngine(rlbot_delay=True)
  self.previous={};self.score=[0,0];self.reset(stage,team,0)
 def reset(self,stage,team,tick):
  s=self.engine.create_base_state();s.tick_count=tick;FixedTeamSizeMutator(1,1).apply(s,{});KickoffMutator().apply(s,{})
  self.ids=list(s.cars);self.our=next(a for a,c in s.cars.items() if c.team_num==team);self.enemy=next(a for a in self.ids if a!=self.our)
  c=s.cars[self.our];r=self.rng
  if stage=='A':
   pos=np.array([r.uniform(-1000,1000),r.uniform(-2000,2000),93.15],np.float32);s.ball.position=pos
   yaw=r.uniform(-math.pi,math.pi);distance=r.uniform(350,900)
   c.physics.position=pos-np.array([distance*math.cos(yaw),distance*math.sin(yaw),76.15],np.float32)
   c.physics.euler_angles=np.array([0,yaw+r.uniform(-.5,.5),0],np.float32)
  elif stage=='B':
   i=int(r.choice([0,1,4]))
   for car in s.cars.values():
    t=car.team_num;car.physics.position=(KickoffMutator.SPAWN_BLUE_POS if t==0 else KickoffMutator.SPAWN_ORANGE_POS)[i].copy()
    yaw=(KickoffMutator.SPAWN_BLUE_YAW if t==0 else KickoffMutator.SPAWN_ORANGE_YAW)[i];car.physics.euler_angles=np.array([0,yaw,0],np.float32)
  elif stage=='C':
   yaw=r.uniform(-math.pi,math.pi);s.ball.position=np.array([0,0,r.uniform(93.15,230)],np.float32)
   c.physics.position=np.array([-900*math.cos(yaw),-900*math.sin(yaw),17],np.float32);c.physics.euler_angles=np.array([0,yaw,0],np.float32)
   c.physics.linear_velocity=np.array([750*math.cos(yaw),750*math.sin(yaw),0],np.float32)
  elif stage=='D':
   sign=1 if team==0 else -1;s.ball.position=np.array([r.uniform(-500,500),-sign*2500,93.15],np.float32)
   s.ball.linear_velocity=np.array([r.uniform(-100,100),-sign*r.uniform(600,1100),0],np.float32)
   c.physics.position=np.array([r.uniform(-300,300),-sign*4300,17],np.float32);c.physics.euler_angles=np.array([0,sign*math.pi/2,0],np.float32)
  self.state=self.engine.set_state(s,{});self.kickoff=stage in ['B','E'];self.local_tick=0
 def observe(self,aid):
  p=self.state.cars[aid].physics;now=self.state.tick_count/120
  x=list(build(dict(position=p.position.tolist(),velocity=p.linear_velocity.tolist(),basis=basis(p.euler_angles.tolist()),ball_present=True,ball_position=self.state.ball.position.tolist()),dict(elapsed=now,previous_elapsed=self.previous.get(aid),previous_mode=0,previous_steer=0.,prediction_valid=False)))[:13]
  assert np.isfinite(x).all() and x[9]==0
  self.previous[aid]=now;return np.asarray(x,np.float32)
 def step(self,ours,theirs,ticks=2):
  old=self.state;self.state=self.engine.step({self.our:np.tile(np.array(ours,np.float32),(ticks,1)),self.enemy:np.tile(np.array(theirs,np.float32),(ticks,1))},{})
  self.local_tick+=ticks
  touch=bool(self.state.cars[self.our].ball_touches)
  if any(c.ball_touches for c in self.state.cars.values()) or np.linalg.norm(self.state.ball.position[:2])>1e-4:self.kickoff=False
  scoring=self.state.scoring_team
  if scoring is not None:self.score[int(scoring)]+=1
  return old,touch,scoring
 def close(self):self.engine.close()

 def reward(self,old,touch,scoring,mode,last_mode):
  aid=self.our;new=self.state;team=new.cars[aid].team_num
  d0=np.linalg.norm(old.ball.position-old.cars[aid].physics.position);d1=np.linalg.norm(new.ball.position-new.cars[aid].physics.position)
  # Privileged geometry is used ONLY for reward; actor always has 13 contract fields.
  sign=1 if team==0 else -1
  progress=float(np.clip((d0-d1)/200,-.25,.25))
  goal_progress=float(np.clip(sign*(new.ball.position[1]-old.ball.position[1])/1000,-.2,.2))
  goal=20*(1 if scoring==team else -1) if scoring is not None else 0
  unnecessary_jump=.012 if mode in [2,3] else 0
  idle_cost=0.
  if self.training_stage=='A':
   # Measured stationary-policy failure: train ground movement, no oracle action labels.
   idle_cost=.03 if np.linalg.norm(new.cars[aid].physics.linear_velocity[:2])<100 else 0.
   unnecessary_jump=.05 if mode in [2,3] else 0.
  return progress+.15*goal_progress+2*int(touch)+goal-unnecessary_jump-idle_cost-.002
