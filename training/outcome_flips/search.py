"""Only retain outcome-improving complete manoeuvres; same genuine opponent."""
from world import *
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
OUT=ROOT/'runs/outcome-flip-search-v1'
def useful(r,b):
 if not r['flip_events'] or r['first_contact'] is None:return False
 if r['goals_for']<b['goals_for'] or r['goals_against']>b['goals_against']:return False
 if r['stationary_seconds']>b['stationary_seconds']+.5 or r['fallen_seconds']>b['fallen_seconds']+.5 or r['circling_windows']>b['circling_windows']:return False
 if len(r['recoveries'])<len(r['flip_events']) or any(a['stable_recovery_seconds']>2.5 for a in r['recoveries']):return False
 if b['first_touch_advantage'] is not None and b['first_touch_advantage']>0 and r['first_touch_advantage'] is not None and r['first_touch_advantage']<0:return False
 v=r['contacts'][0]['goalward_velocity'];bv=b['contacts'][0]['goalward_velocity'] if b['contacts'] else 0
 if v<300:return False
 return r['goals_for']>b['goals_for'] or v>bv+100 or (b['first_contact'] is None or r['first_contact']<b['first_contact']-.1) and v>=bv-50

def process(scene):
 b,*_=trajectory(scene);tt=b['first_contact'] if b['first_contact'] is not None else (3.5 if scene['kind']=='kickoff' else scene.get('gap',1500)/max(scene.get('speed',800)+400,1));onsets=[max(.15,min(2.,tt*.4)),max(.15,tt-.65),max(.15,tt-.3)];timings=[(.08,.06,.033333333333),(.1,.1,.05),(.15,.12,.05)];rows=[]
 for onset in onsets:
  for timing in timings:
   r,*_=trajectory(scene,[onset,*timing]);r['qualifies']=useful(r,b);rows.append(r)
 good=[r for r in rows if r['qualifies']]
 score=lambda r: 3000*(r['goals_for']-r['goals_against'])+(r['contacts'][0]['goalward_velocity'] if r['contacts'] else 0)-200*(r['first_contact'] or 7)-200*r['fallen_seconds']
 best=max(good,key=score) if good else None
 return dict(scene=scene,baseline=b,trials=rows,best=best)
def main():
 assert not OUT.exists();OUT.mkdir();start=time.monotonic();records=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks=[pool.submit(process,s) for s in scenes(9910011)]
  for f in as_completed(tasks):
   r=f.result();records.append(r);(OUT/'search.json').write_text(json.dumps(records,indent=2));print('SEARCH',len(records),r['scene']['kind'],r['scene']['side'],bool(r['best']),flush=True)
 result=dict(scenes=len(records),sequences=sum(len(r['trials']) for r in records),useful_scenes=sum(r['best'] is not None for r in records),useful_sequences=sum(t['qualifies'] for r in records for t in r['trials']),elapsed_seconds=time.monotonic()-start)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result,flush=True)
if __name__=='__main__':main()
