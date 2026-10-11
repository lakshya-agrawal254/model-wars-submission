"""Real-state observability counterexample; no training or runtime changes."""
from train import *
def main():
 a=ForecastWorld(11201011,'O',0);b=ForecastWorld(11201011,'O',0)
 try:
  state=b.state
  for physics in [state.ball]+[c.physics for c in state.cars.values()]:
   physics.position=physics.position*np.array([-1,-1,1],np.float32);physics.linear_velocity=physics.linear_velocity*np.array([-1,-1,1],np.float32);physics.angular_velocity=physics.angular_velocity*np.array([-1,-1,1],np.float32)
  for car in state.cars.values():
   e=car.physics.euler_angles.copy();e[1]+=np.pi;car.physics.euler_angles=e
  b.state=b.engine.set_state(state,{})
  x=a.observe(a.our);y=b.observe(b.our);error=float(np.max(np.abs(x-y)));assert error<1e-5
  target=np.array([0,5120,93.15]);pa=a.state.cars[a.our].physics;pb=b.state.cars[b.our].physics
  ga=[float(np.dot(target-pa.position,z)) for z in [pa.forward,pa.right,pa.up]];gb=[float(np.dot(target-pb.position,z)) for z in [pb.forward,pb.right,pb.up]];assert ga[0]*gb[0]<0
  result=dict(max_13D_difference=error,team_both=0,enemy_goal_world=target.tolist(),enemy_goal_local_A=ga,enemy_goal_local_B=gb,observation_A=x.tolist(),observation_B=y.tolist(),actual_forecasts=True,conclusion='Same legal13D current-observation vector but opposite forward/backward enemy-goal direction. Rotating an entire corresponding history in the symmetric arena preserves local observations too; no general reliable goal identity guarantee from these features.',extra_actor_inputs=False)
  (OUT/'goal-observability.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 finally:a.close();b.close()
if __name__=='__main__':main()
