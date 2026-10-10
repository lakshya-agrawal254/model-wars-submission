"""Fresh side-paired complete matches; original evaluator, explicit callback rate."""
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'training/memory_selective')]
from campaign import GEO,BASE,A,rollout,batch,aggregate
# Preserve the prior full-match protocol; change only decision repeat from2 to1 at120Hz.
source=(ROOT/'training/memory_selective/evaluate.py').read_text();source=source[:source.index('\ndef run(')]
source=source.replace('def match(path,opponent,seed,team):','def match(path,opponent,seed,team,rate=60):\n step_ticks=120//rate')
source=source.replace('range(0,120*900,2)','range(0,120*900,step_ticks)').replace('decode(int(ol.argmax()),float(os)))','decode(int(ol.argmax()),float(os)),ticks=step_ticks)')
source=source.replace('*2/120','*step_ticks/120').replace('len(trajectory)==300','len(trajectory)==5*rate').replace('kickoff_wait+=2','kickoff_wait+=step_ticks').replace('active+=2','active+=step_ticks').replace('ot+=2','ot+=step_ticks').replace('Real RocketSim 60Hz decision callbacks','Real RocketSim variable-rate decision callbacks')
# Imported source uses __file__ to locate root, intentionally supplying original file identity.
ns={'__file__':str(ROOT/'training/memory_selective/evaluate.py'),'__name__':'diagnostic_match'};exec(compile(source,'preserved_match_protocol','exec'),ns);match_impl=ns['match']
def match(*args):return match_impl(*args)
def main():
 out=ROOT/'runs/dagger-corrective-screen';assert not out.exists();out.mkdir();path=ROOT/'runs/dagger-corrective-v1/candidate.pt';records={};start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for rate in [60,120]:
   for name,ours in [('candidate',path),('geometric',GEO)]:
    for opponent,op in [('original',BASE),('A',A),('geometric',GEO)]:
     if name=='geometric' and opponent=='geometric':continue
     key=f'{name}_vs_{opponent}_{rate}';records[key]=[]
     for side in [0,1]:tasks[pool.submit(match,ours,op,9543011,side,rate)]=key
  for f in as_completed(tasks):
   key=tasks[f];r=f.result();r['callback_Hz']=int(key.rsplit('_',1)[1]);records[key].append(r);(out/'records.json').write_text(json.dumps(records,indent=2));print(key,r['result'],r['goals_for'],r['goals_against'],r['end'],flush=True)
 sampled=batch([(p,'B',9550011+k*1009,side,k,rate,20,None,1,True) for p in [GEO,path] for rate in [60,120] for k in range(5) for side in [0,1]])
 (out/'sampled.json').write_text(json.dumps(sampled,indent=2));summary={k:dict(W=sum(r['result']=='W' and 'cap' not in r['end'] for r in rs),L=sum(r['result']=='L' and 'cap' not in r['end'] for r in rs),D=sum(r['result']=='D' and 'cap' not in r['end'] for r in rs),unresolved=sum('cap' in r['end'] for r in rs),GF=sum(r['goals_for'] for r in rs),GA=sum(r['goals_against'] for r in rs),stationary_seconds=sum(r['stationary_seconds'] for r in rs),circling_windows=sum(r['circling_5second_windows_proxy'] for r in rs),productive_contacts=sum(r['productive_contacts'] for r in rs),backward_contacts=sum(r['backward_contacts'] for r in rs)) for k,rs in records.items()};summary['elapsed_seconds']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SCREEN COMPLETE',summary,flush=True)
if __name__=='__main__':main()
