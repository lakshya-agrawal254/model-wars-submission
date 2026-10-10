"""One read-only telemetry replay of the worst consumed screening seed; no tuning."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/flip_feasibility'));import screen
OUT=ROOT/'runs/ground-stall-contact-diagnostic-v1';CURRENT=None
class Telemetry(screen.Tracked):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);global CURRENT;CURRENT=self;self.trace=[];self.n=0
 def step(self,ours,theirs,ticks=2):
  r=super().step(ours,theirs,ticks);self.n+=1
  if self.n%12==0:
   c=self.state.cars[self.our].physics;b=self.state.ball;delta=b.position-c.position;raw=self.engine._cars[self.our].get_state();self.trace.append(dict(tick=self.state.tick_count,position=c.position.tolist(),velocity=c.linear_velocity.tolist(),speed=float(np.linalg.norm(c.linear_velocity)),upright=float(c.up[2]),on_ground=bool(raw.is_on_ground),forward=c.forward.tolist(),steer=float(ours[1]),throttle=float(ours[0]),controls=list(ours),ball_position=b.position.tolist(),ballgap=float(np.linalg.norm(delta)),bearing=float(np.arctan2(np.dot(delta,c.right),np.dot(delta,c.forward))),sidewall_gap=4096-abs(float(c.position[0])),backwall_gap=5120-abs(float(c.position[1])),opponent_position=self.state.cars[self.enemy].physics.position.tolist(),car_distance=float(np.linalg.norm(self.state.cars[self.enemy].physics.position-c.position)),car_contact_id=int(raw.car_contact_id),contact_cooldown=float(raw.car_contact_cooldown_timer),wheels=list(raw.wheels_with_contact),handbrake=float(raw.handbrake_val),is_demoed=bool(raw.is_demoed),is_auto_flipping=bool(raw.is_auto_flipping),world_contact=bool(raw.has_world_contact),engine_throttle=float(self.engine._cars[self.our].get_controls().throttle),engine_steer=float(self.engine._cars[self.our].get_controls().steer)))
  return r

def main():
 assert not OUT.exists();OUT.mkdir();screen.evaluation.match_impl.__wrapped__.__globals__['World']=Telemetry
 r=screen.evaluation.match(ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',ROOT/'runs/geometric-retained/latest.pt',10303029,1,60);trace=CURRENT.trace;stuck=[x for x in trace if x['speed']<100];report=dict(match=r,sampled_rows=len(trace),stationary_rows=len(stuck),stationary_median_position=np.median([x['position'] for x in stuck],axis=0).tolist(),stationary_median_speed=float(np.median([x['speed'] for x in stuck])),stationary_median_steer=float(np.median([x['steer'] for x in stuck])),stationary_median_bearing=float(np.median([x['bearing'] for x in stuck])),stationary_median_upright=float(np.median([x['upright'] for x in stuck])),stationary_on_ground_fraction=float(np.mean([x['on_ground'] for x in stuck])),stationary_wall_near_fraction=float(np.mean([min(x['sidewall_gap'],x['backwall_gap'])<150 for x in stuck])),stationary_car_contact_fraction=float(np.mean([x['car_distance']<200 for x in stuck])),stationary_median_car_distance=float(np.median([x['car_distance'] for x in stuck])),stationary_engine_throttle=float(np.median([x['engine_throttle'] for x in stuck])),scope='Replay of one consumedseed with read-onlyphysics telemetry; not new match evidence or extraoptimizerupdates. No control/callback/weights changes.')
 (OUT/'trace.json').write_text(json.dumps(trace,indent=2));(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
