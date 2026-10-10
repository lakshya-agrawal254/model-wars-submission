"""Untouched-seed/geometry final physical diagnostics. No sequence controller."""
from world import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
OUT=ROOT/'runs/outcome-flip-independent-v1'
def cases():
 rows=[]
 for side in [0,1]:
  for i in range(5):rows.append(dict(kind='kickoff',side=side,index=i,enemy_index=(i+1)%5,seed=9990011+side*10000+i*1009))
  for j,kw in enumerate([
   dict(kind='unseen_ground',speed=900,gap=1350,yaw_error=.09),
   dict(kind='unseen_moving',speed=1150,gap=1750,ball_vx=140,ball_vy=80,yaw_error=-.06),
   dict(kind='unseen_shot',speed=1000,gap=1150,ball_y=3000,ball_x=350,yaw_error=-.08,enemy_x=-850,enemy_speed=0),
   dict(kind='unseen_low_air',speed=950,gap=1450,height=210,ball_vz=130)]):rows.append(dict(side=side,seed=9992011+side*10000+j*1009,**kw))
 return rows

def worker(args):
 scene,path,hz=args;r,*_=trajectory(scene,path=path,rate=hz);return r

def main():
 assert not OUT.exists();OUT.mkdir();rows={};models={'geometric':GEO,'candidate128':ROOT/'runs/outcome-flip-learning-v1/candidate.pt'};begin=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,path in models.items():
   for hz in [60,120]:
    key=f'{name}_{hz}';rows[key]=[]
    for scene in cases():tasks[pool.submit(worker,(scene,path,hz))]=key
  for f in as_completed(tasks):key=tasks[f];rows[key].append(f.result())
 (OUT/'records.json').write_text(json.dumps(rows,indent=2));summary={}
 for key,r in rows.items():
  flips=sum(len(a['flip_events']) for a in r);useful=0
  for a in r:
   for e in a['flip_events']:
    useful+=any(0<=c['t']-e['t']<=.7 and c['goalward_velocity']>300 and c['goalward_delta_velocity']>100 for c in a['contacts'])
  kickoff=[a for a in r if a['scene']['kind']=='kickoff'];contacts=[a['first_contact'] for a in kickoff if a['first_contact'] is not None];summary[key]=dict(trials=len(r),contacts=sum(a['first_contact'] is not None for a in r),kickoff_contacts=len(contacts),kickoff_trials=len(kickoff),median_kickoff_contact=float(np.median(contacts)) if contacts else None,dodge_events=flips,unique_useful_flip_events=useful,useful_flip_fraction=useful/flips if flips else None,stationary_seconds=sum(a['stationary_seconds'] for a in r),fallen_seconds=sum(a['fallen_seconds'] for a in r),circling_windows=sum(a['circling_windows'] for a in r),goals_for=sum(a['goals_for'] for a in r),goals_against=sum(a['goals_against'] for a in r))
 summary['elapsed_seconds']=time.monotonic()-begin;(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
