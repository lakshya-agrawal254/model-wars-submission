"""Frozen deterministic matched-side full matches; tracking only, no controls changed."""
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/interception_readout'))
import evaluate as evaluation
import numpy as np
CURRENT=None
class Tracked(evaluation.ForecastWorld):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);global CURRENT;CURRENT=self;self.actual_flips=[];self.fall_seconds=0.;self.recoveries=[];self.pending_flip=None
 def step(self,ours,theirs,ticks=2):
  raw=self.engine._cars[self.our].get_state();was_flip=bool(raw.is_flipping);was_ground=bool(raw.is_on_ground);before=self.state.tick_count
  result=super().step(ours,theirs,ticks);raw=self.engine._cars[self.our].get_state();car=self.state.cars[self.our]
  if raw.is_flipping and not was_flip:self.actual_flips.append(before/120);self.pending_flip=before
  if raw.is_on_ground and not was_ground and self.pending_flip is not None:self.recoveries.append((self.state.tick_count-self.pending_flip)/120);self.pending_flip=None
  self.fall_seconds+=int(car.physics.position[2]<60 and car.physics.up[2]<.5)*ticks/120
  return result
 def reset(self,*a):
  super().reset(*a)
  if hasattr(self,'pending_flip'):self.pending_flip=None

def match(path,op,seed,side,rate):
 evaluation.match_impl.__wrapped__.__globals__['World']=Tracked
 r=evaluation.match(path,op,seed,side,rate);r.update(actual_dodge_events=len(CURRENT.actual_flips),flip_times=CURRENT.actual_flips,landing_recovery_seconds=CURRENT.recoveries,fallen_low_height_seconds=CURRENT.fall_seconds);return r

def main():
 out=ROOT/'runs/flip-screen-corrected-v1';assert not out.exists();out.mkdir();records={};start=time.monotonic();candidate=ROOT/'runs/flip-learning-v1/candidate.pt'
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for rate in [60,120]:
   for name,p in [('candidate',candidate),('geometric',evaluation.GEO)]:
    for opp,op in [('original',evaluation.BASE),('A',evaluation.A),('geometric',evaluation.GEO)]:
     if name=='geometric' and opp=='geometric':continue
     key=f'{name}_vs_{opp}_{rate}';records[key]=[]
     for side in [0,1]:tasks[pool.submit(match,p,op,9873011,side,rate)]=key
  for f in as_completed(tasks):
   k=tasks[f];r=f.result();records[k].append(r);(out/'records.json').write_text(json.dumps(records,indent=2));print(k,r['result'],r['end'],r['goals_for'],r['goals_against'],'flips',r['actual_dodge_events'],flush=True)
 summary={}
 for k,rows in records.items():
  summary[k]={name:sum(test(r) for r in rows) for name,test in dict(W=lambda r:r['result']=='W' and 'cap' not in r['end'],L=lambda r:r['result']=='L' and 'cap' not in r['end'],D=lambda r:r['result']=='D' and 'cap' not in r['end'],unresolved=lambda r:'cap' in r['end'],GF=lambda r:r['goals_for'],GA=lambda r:r['goals_against'],flips=lambda r:r['actual_dodge_events'],stationary_seconds=lambda r:r['stationary_seconds'],circling_windows=lambda r:r['circling_5second_windows_proxy'],goalward_contacts=lambda r:r['productive_contacts'],backward_contacts=lambda r:r['backward_contacts'],falls_seconds=lambda r:r['fallen_low_height_seconds']).items()}
 summary['elapsed_seconds']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
