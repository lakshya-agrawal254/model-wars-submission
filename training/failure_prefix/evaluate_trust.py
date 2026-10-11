"""Frozen fresh paired full-match evaluation; read-only contact telemetry."""
import sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/failure-prefix-trust-screen-v1';sys.path.insert(0,str(ROOT/'training/ground_trajectories'))
from final_compare import summarize
MODELS=dict(candidate=ROOT/'runs/failure-prefix-trust-v1/candidate-batch-2.pt',outcome=ROOT/'runs/outcome-flip-learning-v1/candidate.pt',ground=ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',geometric=ROOT/'model/V13.pt')
def job(args):
 sys.path.insert(0,str(ROOT/'training/flip_feasibility'));import screen
 if not hasattr(screen,'FORECAST_DIAGNOSTIC_ORIGINAL_TRACKED'):screen.FORECAST_DIAGNOSTIC_ORIGINAL_TRACKED=screen.Tracked
 class Diagnostic(screen.FORECAST_DIAGNOSTIC_ORIGINAL_TRACKED):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.near=False;self.near_touch=False;self.missed=0;self.collision_events=0;self.collision_start=None;self.collision_recoveries=[];self.last_cooldown=0.;self.abrupt_changes=0;self.slow_car_proximity=0.
  def reset(self,*a):
   super().reset(*a)
   if hasattr(self,'near'):self.near=False;self.near_touch=False;self.collision_start=None;self.last_cooldown=0.
  def step(self,ours,theirs,ticks=2):
   result=super().step(ours,theirs,ticks);old,touch,goal=result;c=self.state.cars[self.our].physics;dist=float(np.linalg.norm(c.position-self.state.ball.position));speed=float(np.linalg.norm(c.linear_velocity));raw=self.engine._cars[self.our].get_state();cool=float(raw.car_contact_cooldown_timer);now=self.state.tick_count/120
   if dist<450 and not self.near:self.near=True;self.near_touch=False
   if self.near:self.near_touch|=bool(touch)
   if self.near and dist>650:self.missed+=int(not self.near_touch);self.near=False
   if cool>self.last_cooldown+1e-5:
    self.collision_events+=1
    if self.collision_start is None:self.collision_start=now
   if self.collision_start is not None and speed>400 and bool(raw.is_on_ground):self.collision_recoveries.append(now-self.collision_start);self.collision_start=None
   self.last_cooldown=cool;self.abrupt_changes+=int(np.linalg.norm(self.state.ball.linear_velocity-old.ball.linear_velocity)>300);self.slow_car_proximity+=int(speed<100 and np.linalg.norm(c.position-self.state.cars[self.enemy].physics.position)<200)*ticks/120
   return result
 assert sum(c.__name__=='Diagnostic' for c in Diagnostic.__mro__)==1
 screen.Tracked=Diagnostic;r=screen.match(*args);w=screen.CURRENT;assert w.slow_car_proximity <= r['stationary_seconds']+1e-5
 r.update(missed_close_approaches_proxy=w.missed,actual_car_contact_events=w.collision_events,collision_to_ground_speed400_seconds=w.collision_recoveries,unrecovered_collision_at_end=w.collision_start is not None,abrupt_ball_velocity_changes=w.abrupt_changes,stationary_near_other_car_seconds=w.slow_car_proximity,extra_metric_limits='Missed approach=within450 then outside650UU without own touch, not a proven interception opportunity. Collision recovery=speed>400 and grounded, not tactical recovery. Contacts use engine cooldown increases; goal resets censor pending events.')
 return r
def main():
 assert not OUT.exists();OUT.mkdir();begin=time.monotonic();hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in MODELS.items()};configs=[('candidate',b,60,4) for b in ['outcome','ground','geometric']]+[('outcome',b,60,4) for b in ['ground','geometric']]
 protocol=dict(checkpoint_SHA256=hashes,configurations=configs,seeds=[12201011+1009*i for i in range(4)],training_seed_disjoint=True,Hz='60only;120not claimed',greedy=True,fullmatches='300active sec, floor proxy,120sec overtimecap censored,900wall-simulation cap',selection='Candidate must beat initializer convincingly, improve common-opponent results without movement regressions. This is screening, no native acceptance or promotion.',runtime_changed=False,confirmation_screen_gate={'direct_wins_at_least':6,'direct_matches':8,'common_opponent_wins_not_below_outcome':True,'common_goal_difference_not_below_outcome':True,'maximum_stationary_fraction_increase':.01,'maximum_circling_rate_ratio':1.25,'minimum_confirmation_matches_vs_initializer':100,'note':'Prespecified before evaluating this candidate. Gates only warrant independent confirmation; they do not authorize promotion. If reference circling is zero, no candidate circling is allowed.'})
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));records={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for a,b,hz,n in configs:
   key=f'{a}_vs_{b}_{hz}';records[key]=[]
   for i in range(n):
    for side in [0,1]:tasks[pool.submit(job,(MODELS[a],MODELS[b],12201011+1009*i,side,hz))]=key
  for f in as_completed(tasks):
   key=tasks[f];r=f.result();records[key].append(r);(OUT/'records.json').write_text(json.dumps(records,indent=2));print(key,len(records[key]),r['result'],r['end'],r['goals_for'],r['goals_against'],round(r['stationary_seconds'],2),flush=True)
 assert hashes=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in MODELS.items()}
 summary={}
 for k,rs in records.items():
  s=summarize(rs);s.update(missed_close_approaches_proxy=sum(r['missed_close_approaches_proxy'] for r in rs),car_contact_events=sum(r['actual_car_contact_events'] for r in rs),stationary_near_other_car_seconds=sum(r['stationary_near_other_car_seconds'] for r in rs),collision_recovery_seconds=[x for r in rs for x in r['collision_to_ground_speed400_seconds']],unrecovered_collisions_at_end=sum(r['unrecovered_collision_at_end'] for r in rs));summary[k]=s
 (OUT/'result.json').write_text(json.dumps(dict(summary=summary,elapsed_seconds=time.monotonic()-begin,promoted=False,native='UNTESTED'),indent=2));print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
