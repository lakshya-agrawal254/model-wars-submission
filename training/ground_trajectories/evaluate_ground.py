"""Independent fresh seeds, frozen deterministic actors; no search drivers."""
from world_ground import *
from train_ground import fresh
from concurrent.futures import ProcessPoolExecutor,as_completed
CANDIDATE=ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt'
def physical_job(args):s,path,hz=args;return run(s,path=path,hz=hz)[0]
def match_job(args):
 sys.path.insert(0,str(ROOT/'training/flip_feasibility'));import screen;return screen.match(*args)
def main():
 OUT=ROOT/'runs/ground-trajectory-evaluation-v1';assert not OUT.exists();OUT.mkdir();start=time.monotonic();records={};models={'candidate':CANDIDATE,'outcome128':START,'geometric':GEO}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,path in models.items():
   for hz in [60,120]:
    key=f'{name}_{hz}';records[key]=[]
    for s in fresh(10201011,change=-1):tasks[pool.submit(physical_job,(s,path,hz))]=key
  for f in as_completed(tasks):records[tasks[f]].append(f.result())
 (OUT/'physical-records.json').write_text(json.dumps(records,indent=2));physical={k:aggregate(v) for k,v in records.items()};(OUT/'physical-summary.json').write_text(json.dumps(physical,indent=2));print('INDEPENDENTPHYSICAL',physical,flush=True)
 matches={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  configurations=[('candidate','outcome128',60,4),('candidate','geometric',60,4),('outcome128','geometric',60,4),('candidate','original',60,4),('candidate','outcome128',120,2),('candidate','geometric',120,2),('outcome128','geometric',120,2)]
  models['original']=BASE
  for name,opp,hz,pairs in configurations:
   key=f'{name}_vs_{opp}_{hz}';matches[key]=[]
   for i in range(pairs):
    for side in [0,1]:tasks[pool.submit(match_job,(models[name],models[opp],10301011+i*1009,side,hz))]=key
  for f in as_completed(tasks):
   k=tasks[f];r=f.result();matches[k].append(r);(OUT/'match-records.json').write_text(json.dumps(matches,indent=2));print('MATCH',k,len(matches[k]),r['result'],r['end'],r['goals_for'],r['goals_against'],flush=True)
 summary={}
 for k,rows in matches.items():
  secs=sum(sum(r['counts'])/r['callback_Hz'] for r in rows);summary[k]=dict(W=sum(r['result']=='W' and 'cap' not in r['end'] for r in rows),L=sum(r['result']=='L' and 'cap' not in r['end'] for r in rows),D=sum(r['result']=='D' and 'cap' not in r['end'] for r in rows),unresolved=sum('cap' in r['end'] for r in rows),GF=sum(r['goals_for'] for r in rows),GA=sum(r['goals_against'] for r in rows),stationary_seconds=sum(r['stationary_seconds'] for r in rows),observed_seconds=secs,circling=sum(r['circling_5second_windows_proxy'] for r in rows),goalward_contacts=sum(r['productive_contacts'] for r in rows),backward_contacts=sum(r['backward_contacts'] for r in rows),falls_seconds=sum(r['fallen_low_height_seconds'] for r in rows),actual_dodges=sum(r['actual_dodge_events'] for r in rows))
 (OUT/'match-summary.json').write_text(json.dumps(summary,indent=2));result=dict(physical=physical,matches=summary,elapsed_seconds=time.monotonic()-start,promoted=False,actualWindows='UNTESTED');(OUT/'result.json').write_text(json.dumps(result,indent=2));print('FINALSCREEN',result,flush=True)
if __name__=='__main__':main()
