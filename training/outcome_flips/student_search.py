"""Fresh currentstudent prefixes, outcome-tested training-only intervention branches."""
from world import *
from search import useful
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
OUT=ROOT/'runs/outcome-flip-student-search-v1';CURRENT=ROOT/'runs/outcome-flip-learning-v1/latest.pt'
def process(scene):
 base,*_=trajectory(scene,path=CURRENT);tt=base['first_contact'] or (3.5 if scene['kind']=='kickoff' else scene['gap']/max(scene['speed']+400,1));candidates=[]
 for onset in [max(.15,min(2.,tt*.4)),max(.15,tt-.65),max(.15,tt-.3)]:
  for timing in [(.08,.06,.033333333333),(.1,.1,.05),(.15,.12,.05)]:
   r,*_=trajectory(scene,[onset,*timing],path=CURRENT)
   if useful(r,base):candidates.append(r)
 candidates.sort(key=lambda r:3000*(r['goals_for']-r['goals_against'])+r['contacts'][0]['goalward_velocity']-200*r['first_contact'],reverse=True)
 for r in candidates:
  no,*_=trajectory(scene,r['schedule'],path=CURRENT,disable_second_press=True);v=r['contacts'][0]['goalward_velocity'];nv=no['contacts'][0]['goalward_velocity'] if no['contacts'] else 0
  good=r['goals_for']>=no['goals_for'] and r['goals_against']<=no['goals_against'] and (v>nv+100 or r['goals_for']>no['goals_for'] or (no['first_contact'] is None or r['first_contact']<no['first_contact']-.1) and v>=nv-50)
  if good:return dict(scene=scene,selected=True,best=r,baseline=base,no_second_press=no)
 return dict(scene=scene,selected=False,baseline=base)
def main():
 assert not OUT.exists();OUT.mkdir();rows=scenes(9960011)
 # Include physically successful close contacts where the ground policy should be protected.
 for side in [0,1]:rows.append(dict(kind='close_ground',side=side,seed=9967011+side*1009,speed=1400,gap=240,ball_y=1000,ball_vx=60))
 results=[];begin=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(process,s) for s in rows]):
   r=f.result();results.append(r);(OUT/'records.json').write_text(json.dumps(results,indent=2));print('STUDENTSEARCH',len(results),r['scene']['kind'],r['selected'],flush=True)
 (OUT/'summary.json').write_text(json.dumps(dict(scenes=len(results),positive=sum(r['selected'] for r in results),elapsed_seconds=time.monotonic()-begin),indent=2))
if __name__=='__main__':main()
