"""Fresh matched full-match screen. Both missing/native-schema physics-forecast domains explicit."""
import sys,json,time,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from campaign import GEO,BASE,A
from forecast_world import ForecastWorld
source=(ROOT/'training/memory_selective/evaluate.py').read_text();source=source[:source.index('\ndef run(')]
source=source.replace('def match(path,opponent,seed,team):','def match(path,opponent,seed,team,rate=60):\n step_ticks=120//rate')
source=source.replace('range(0,120*900,2)','range(0,120*900,step_ticks)').replace('decode(int(ol.argmax()),float(os)))','decode(int(ol.argmax()),float(os)),ticks=step_ticks)')
source=source.replace('*2/120','*step_ticks/120').replace('len(trajectory)==300','len(trajectory)==5*rate').replace('kickoff_wait+=2','kickoff_wait+=step_ticks').replace('active+=2','active+=step_ticks').replace('ot+=2','ot+=step_ticks')
ns={'__file__':str(ROOT/'training/memory_selective/evaluate.py'),'__name__':'preserved_fullmatch'};exec(compile(source,'preserved_fullmatch_protocol','exec'),ns);match_impl=ns['match'];match_impl.__globals__['World']=ForecastWorld

def match(path,opponent,seed,side,rate):
 r=match_impl(path,opponent,seed,side,rate);r['callback_Hz']=rate;r['observation_domain']='Exact13Dlayout with realRocketSim2secballforecast; NOT actualnativeRLBot prediction';r['domain']='Real RocketSim3D fullmatches, physicalballforecast enabled for BOTHagents; NOT actualRocketLeague';return r

def main():
 out=ROOT/'runs/interception-readout-screen';assert not out.exists();out.mkdir();start=time.monotonic();path=ROOT/'runs/interception-readout-v1/candidate.pt';records={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for rate in [60,120]:
   for name,ours in [('candidate',path),('geometric',GEO)]:
    for op_name,op in [('original',BASE),('geometric',GEO),('A',A)]:
     if name=='geometric' and op_name=='geometric':continue
     key=f'{name}_vs_{op_name}_{rate}';records[key]=[]
     for i in range(2):
      for side in [0,1]:tasks[pool.submit(match,ours,op,9723011+i*1009,side,rate)]=key
  for f in as_completed(tasks):
   key=tasks[f];r=f.result();records[key].append(r);(out/'records.json').write_text(json.dumps(records,indent=2));print('MATCH',key,len(records[key]),r['result'],r['goals_for'],r['goals_against'],r['end'],flush=True)
 summary={}
 for key,rows in records.items():
  summary[key]=dict(W=sum(r['result']=='W' and 'cap' not in r['end'] for r in rows),L=sum(r['result']=='L' and 'cap' not in r['end'] for r in rows),D=sum(r['result']=='D' and 'cap' not in r['end'] for r in rows),unresolved=sum('cap' in r['end'] for r in rows),GF=sum(r['goals_for'] for r in rows),GA=sum(r['goals_against'] for r in rows),stationary_seconds=sum(r['stationary_seconds'] for r in rows),circling_windows=sum(r['circling_5second_windows_proxy'] for r in rows),productive_contacts=sum(r['productive_contacts'] for r in rows),backward_contacts=sum(r['backward_contacts'] for r in rows),meaningless_contacts=sum(r['nonproductive_contacts'] for r in rows),postcontact_goal_events=sum(len(r['postcontact_goals_within5sec_proxy']) for r in rows))
 summary['elapsed_seconds']=time.monotonic()-start;summary['forecasts']='GenuineRocketSimballphysics,NOTnativeRLBot recording; originalforecastcolumns enabled instead ofzero. Olderzero-forecast scores NOT directly comparable.';(out/'summary.json').write_text(json.dumps(summary,indent=2));print('MATCH SCREEN COMPLETE',summary,flush=True)
if __name__=='__main__':main()
