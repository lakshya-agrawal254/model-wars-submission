from world_ground import *
from concurrent.futures import ProcessPoolExecutor,as_completed
OUT=ROOT/'runs/ground-trajectory-search-v1'
PARAMS=[(lead,gain,damp,lock) for lead,gain,damp,lock in [(0,1.5,0,True),(.3,1.5,0,True),(.6,1.5,.06,True),(.8,1.1,.08,True),(.4,2.,.04,True),(.7,2.,.08,True),(.25,1.,0,True),(.5,1.1,.12,True),(.6,1.5,0,False),(.3,2.,.12,False),(.8,1.5,-.03,True),(0,2.,.08,True)]]
def worker(args):scene,params=args;return run(scene,params=params)[0]
def useful(r,b):
 if r['GA']>b['GA'] or r['GF']<b['GF'] or r['stationary']>b['stationary']+.1 or r['circling']>b['circling'] or r['fallen']>b['fallen']+.2:return False
 if r['contact'] is None:return False
 if b['contact'] is None:return True
 bettertime=b['contact']-r['contact']>.25;direction=(r['contacts'][0]['goalward']>b['contacts'][0]['goalward']+150 and r['contacts'][0]['delta_goalward']>100) if r['contacts'] and b['contacts'] else False
 if r['GF']>b['GF']:return True
 if bettertime and (not b['contacts'] or r['contacts'][0]['goalward']>=b['contacts'][0]['goalward']-100):return True
 return direction

def score(r):return 40*r['GF']-60*r['GA']+12*(r['contact'] is not None)+(16-r['contact'] if r['contact'] is not None else 0)+sum(np.clip(c['delta_goalward']/1000,-2,2) for c in r['contacts'])-3*r['stationary']-4*r['circling']-r['postcontact_stationary']
def main():
 assert not OUT.exists();OUT.mkdir();begin=time.monotonic();cases=scenes(10001011);rows=[dict(scene=s,baseline=None,trials=[]) for s in cases]
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for i,s in enumerate(cases):
   tasks[pool.submit(worker,(s,None))]=(i,None)
   for p in PARAMS:tasks[pool.submit(worker,(s,p))]=(i,p)
  for f in as_completed(tasks):
   i,p=tasks[f];r=f.result()
   if p is None:rows[i]['baseline']=r
   else:rows[i]['trials'].append(r)
 for r in rows:
  good=[v for v in r['trials'] if useful(v,r['baseline'])];r['selected']=max(good,key=score) if good else None
 (OUT/'records.json').write_text(json.dumps(rows,indent=2));summary=dict(scenes=len(rows),physics_candidates=len(rows)*len(PARAMS),selected=sum(r['selected'] is not None for r in rows),by_kind={k:sum(r['selected'] is not None for r in rows if r['scene']['kind']==k) for k in sorted(set(r['scene']['kind'] for r in rows))},baseline=aggregate([r['baseline'] for r in rows]),selected_matched_control=aggregate([r['baseline'] for r in rows if r['selected']]),selected_teacher=aggregate([r['selected'] for r in rows if r['selected']]),elapsed_seconds=time.monotonic()-begin)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
