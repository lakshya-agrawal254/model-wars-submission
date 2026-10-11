"""Frozen full-match screen after movement guard stopped fitting. No further learning."""
from pathlib import Path
import sys,json,time
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/mirrored-own-8-screen-v1';C=ROOT/'runs/mirrored-own-8-v1/candidate.pt';O=ROOT/'runs/outcome-flip-learning-v1/candidate.pt';G=ROOT/'model/V13.pt';B=ROOT/'runs/retained/baseline/V13.pt'
def job(args):
 path,op,seed,side,hz,record=args;sys.path.insert(0,str(ROOT/'training/flip_feasibility'));import screen
 if record:
  class Recorded(screen.Tracked):
   def __init__(self,*a,**kw):super().__init__(*a,**kw);self.xs=[];self.oxs=[];self.controls=[];self.ocontrols=[];self.ticks=[]
   def observe(self,aid):
    x=super().observe(aid);(self.xs if aid==self.our else self.oxs).append(x.copy());return x
   def step(self,ours,theirs,ticks=2):
    self.controls.append(ours);self.ocontrols.append(theirs);self.ticks.append(self.state.tick_count);return super().step(ours,theirs,ticks)
  screen.Tracked=Recorded
 r=screen.match(path,op,seed,side,hz)
 if record:
  w=screen.CURRENT;assert len(w.xs)==len(w.oxs)==len(w.controls)==len(w.ticks)
  target=OUT/f'chronological-seed-{seed}-side-{side}.npz';assert not target.exists()
  np.savez_compressed(target,observations=np.asarray(w.xs,np.float32),opponent_observations=np.asarray(w.oxs,np.float32),transmitted_controls=np.asarray(w.controls,np.float32),opponent_transmitted_controls=np.asarray(w.ocontrols,np.float32),physics_ticks=np.asarray(w.ticks,np.int64))
  r['chronological_record']=target.name;r['recorded_actor_roles']=['candidate4','preservedOutcome128'];r['GRU_contract']='Each actor h=None only at match start; memory carries across goals; no timestep subsampling. Simulator does not reproduce native replay/countdown callbacks.'
 return r
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic();records={};configs=[('candidate_vs_outcome',C,O,4),('candidate_vs_original',C,B,4),('candidate_vs_geometric',C,G,2)]
 protocol=dict(frozen_candidate=str(C.relative_to(ROOT)),training_stopped_at=4,reason='Predeclared contact guard 26to24; no further updates',seed_base=10801011,pairs=[(k,n) for k,_,_,n in configs],Hz=60,confirmation=False,promotion=False,recording='First matched seed pair againstOutcome: complete chronological13D and actually transmitted controls for BOTH own legal policies. No external labels; no recorded-match learning inthis run.')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for key,path,op,n in configs:
   records[key]=[]
   for i in range(n):
    for side in [0,1]:tasks[pool.submit(job,(path,op,10801011+i*1009,side,60,key=='candidate_vs_outcome' and i==0))]=key
  for f in as_completed(tasks):
   key=tasks[f];r=f.result();records[key].append(r);(OUT/'records.json').write_text(json.dumps(records,indent=2));print(key,len(records[key]),r['result'],r['end'],r['goals_for'],r['goals_against'],flush=True)
 summary={}
 for key,rows in records.items():
  summary[key]=dict(W=sum(r['result']=='W' and 'cap' not in r['end'] for r in rows),L=sum(r['result']=='L' and 'cap' not in r['end'] for r in rows),D=sum(r['result']=='D' and 'cap' not in r['end'] for r in rows),U=sum('cap' in r['end'] for r in rows),GF=sum(r['goals_for'] for r in rows),GA=sum(r['goals_against'] for r in rows),stationary_seconds=sum(r['stationary_seconds'] for r in rows),circling=sum(r['circling_5second_windows_proxy'] for r in rows),observed_seconds=sum(sum(r['counts'])/60 for r in rows),actual_dodges=sum(r['actual_dodge_events'] for r in rows))
 result=dict(summary=summary,elapsed_seconds=time.monotonic()-started,promoted=False,native='UNTESTED',new_optimizer_updates=0);(OUT/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
