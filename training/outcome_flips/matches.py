"""Frozen greedy matches with verified actual forecast world and dodge tracking."""
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/flip_feasibility'))
import screen as prior
GEO=ROOT/'runs/geometric-retained/latest.pt';BASE=ROOT/'runs/retained/baseline/V13.pt';CURRENT=ROOT/'runs/outcome-flip-learning-v1/candidate.pt';OUT=ROOT/'runs/outcome-flip-matches-v1'
def main():
 assert not OUT.exists();OUT.mkdir();records={};begin=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for hz in [60,120]:
   for name,path in [('candidate128',CURRENT),('geometric',GEO)]:
    for opp,op in [('geometric',GEO),('original',BASE)]:
     if name=='geometric' and opp=='geometric':continue
     key=f'{name}_vs_{opp}_{hz}';records[key]=[]
     for i in range(2):
      for side in [0,1]:tasks[pool.submit(prior.match,path,op,9983011+i*1009,side,hz)]=key
  for f in as_completed(tasks):
   k=tasks[f];r=f.result();records[k].append(r);(OUT/'records.json').write_text(json.dumps(records,indent=2));print('MATCH',k,len(records[k]),r['result'],r['end'],r['goals_for'],r['goals_against'],r['actual_dodge_events'],flush=True)
 summary={}
 for k,rows in records.items():
  secs=sum(sum(r['counts'])/r['callback_Hz'] for r in rows);stat=sum(r['stationary_seconds'] for r in rows)
  summary[k]=dict(W=sum(r['result']=='W' and 'cap' not in r['end'] for r in rows),L=sum(r['result']=='L' and 'cap' not in r['end'] for r in rows),D=sum(r['result']=='D' and 'cap' not in r['end'] for r in rows),unresolved=sum('cap' in r['end'] for r in rows),GF=sum(r['goals_for'] for r in rows),GA=sum(r['goals_against'] for r in rows),actual_dodges=sum(r['actual_dodge_events'] for r in rows),stationary_seconds=stat,seconds_observed=secs,stationary_fraction=stat/secs,circling_windows=sum(r['circling_5second_windows_proxy'] for r in rows),goalward_contacts=sum(r['productive_contacts'] for r in rows),backward_contacts=sum(r['backward_contacts'] for r in rows),fallen_seconds=sum(r['fallen_low_height_seconds'] for r in rows))
 summary['elapsed_seconds']=time.monotonic()-begin;(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
