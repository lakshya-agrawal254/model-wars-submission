"""Frozen, fresh paired screening only. No optimizer or runtime changes."""
import sys, json, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/final-v13-comparison-v1'
MODELS={
 'ground64':ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt',
 'outcome128':ROOT/'runs/outcome-flip-learning-v1/candidate.pt',
 'geometric':ROOT/'model/V13.pt',
 'original':ROOT/'runs/retained/baseline/V13.pt'}
def job(args):
 sys.path.insert(0,str(ROOT/'training/flip_feasibility'))
 import screen
 return screen.match(*args)
def summarize(rows):
 def valid(r):return 'cap' not in r['end']
 n=len(rows);w=sum(valid(r) and r['result']=='W' for r in rows)
 z=1.96;p=w/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
 return dict(matches=n,W=w,L=sum(valid(r) and r['result']=='L' for r in rows),
 D=sum(valid(r) and r['result']=='D' for r in rows),U=sum(not valid(r) for r in rows),
 GF=sum(r['goals_for'] for r in rows),GA=sum(r['goals_against'] for r in rows),
 stationary_seconds=sum(r['stationary_seconds'] for r in rows),
 max_stationary_match_seconds=max(r['stationary_seconds'] for r in rows),
 observed_seconds=sum(sum(r['counts'])/r['callback_Hz'] for r in rows),
 circling_windows=sum(r['circling_5second_windows_proxy'] for r in rows),
 goalward_contacts=sum(r['productive_contacts'] for r in rows),
 backward_contacts=sum(r['backward_contacts'] for r in rows),
 fallen_seconds=sum(r['fallen_low_height_seconds'] for r in rows),
 actual_dodges=sum(r['actual_dodge_events'] for r in rows),
 win_fraction_wilson_iid_95=[mid-half,mid+half],
 uncertainty_note='Side pairs are correlated. Wilson treats matches as independent and is optimistic; U is not a draw.')
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic()
 hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in MODELS.items()}
 pairs=[('ground64','outcome128'),('ground64','geometric'),('outcome128','geometric')]
 configs=[(a,b,60,6) for a,b in pairs]+[(a,'original',60,4) for a in ['ground64','outcome128','geometric']]+[(a,b,120,2) for a,b in pairs]
 protocol=dict(checkpoints=hashes,configurations=configs,seeds=[10501011+i*1009 for i in range(6)],workers=4,training_updates=0,
 selection='Screening recommendation only; no promotion. Reject movement regression; full-match evidence precedes contact statistics. Native acceptance and user approval mandatory.',
 conditions='Frozen greedy actors, both sides, real 120Hz RocketSim physics, actual 2s simulator forecast, 300 active seconds plus capped overtime. Recurrent memory persists across goal resets.')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));records={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for a,b,hz,n in configs:
   key=f'{a}_vs_{b}_{hz}';records[key]=[]
   for i in range(n):
    for side in [0,1]:tasks[pool.submit(job,(MODELS[a],MODELS[b],10501011+i*1009,side,hz))]=key
  for f in as_completed(tasks):
   key=tasks[f];r=f.result();records[key].append(r)
   (OUT/'records.json').write_text(json.dumps(records,indent=2))
   print(key,len(records[key]),r['result'],r['end'],r['goals_for'],r['goals_against'],round(r['stationary_seconds'],2),flush=True)
 assert hashes=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in MODELS.items()}
 result=dict(protocol=protocol,summary={k:summarize(v) for k,v in records.items()},elapsed_seconds=time.monotonic()-started,weights_unchanged=True,native_acceptance='UNTESTED',promoted=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
