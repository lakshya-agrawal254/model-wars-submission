"""Same fresh matched-side diagnostics and full match conditions for every policy."""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'training'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from world import World
from evaluate import load,match
from runtime import decode

@torch.inference_mode()
def trial(path,kind,seed,side,index=None):
 torch.set_num_threads(1);p=load(path);op=load(ROOT/'runs/retained/baseline/V13.pt');w=World(seed,'A' if kind=='ground' else 'B',side,kickoff_index=index);h=None;oh=None;first=None;counts=np.zeros(4,int);flips=0;far_flips=0;steers=[];lat=[];seconds=30 if kind=='ground' else 90
 try:
  assert hasattr(w.state.cars[w.our],'is_flipping')
  for tick in range(seconds*60):
   x=w.observe(w.our);ox=w.observe(w.enemy);began=time.perf_counter();l,s,h=p(torch.tensor(x[None,:]),h);lat.append((time.perf_counter()-began)*1000);ol,os,oh=op(torch.tensor(ox[None,:]),oh);mode=int(l.argmax());counts[mode]+=1
   if mode==1:steers.append(abs(float(s)))
   flipping=w.state.cars[w.our].is_flipping;old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   if not flipping and w.state.cars[w.our].is_flipping:
    flips+=1;far_flips+=int(np.linalg.norm(old.cars[w.our].physics.position-old.ball.position)>800)
   if touch and first is None:first=tick/60
   if goal is not None:break
  return dict(kind=kind,seed=seed,team=side,kickoff_index=index,seconds_observed=(tick+1)/60,first_touch_seconds=first,contacted=first is not None,counts=counts.tolist(),flip_starts=flips,far_ball_flip_starts=far_flips,far_flip_definition='actual flip rising edge at >800UU from ball; proxy, not proven unnecessary',chase_steer_abs_mean=float(np.mean(steers)) if steers else None,latency_ms_mean=float(np.mean(lat)))
 finally:w.close()

def evaluate(models,output):
 assert not output.exists();output.mkdir(parents=True);began=time.monotonic();records={name:[] for name in models}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,path in models.items():
   for i in range(8):
    for side in [0,1]:tasks[pool.submit(trial,path,'ground',1710011+i*1009,side)]=name
   for index in range(5):
    for repeat in range(2):
     for side in [0,1]:tasks[pool.submit(trial,path,'kickoff',1712011+repeat*10007+index*1009,side,index)]=name
   for i in range(4):
    for side in [0,1]:tasks[pool.submit(match,path,1713011+i*1009,side)]=name
  done=0
  for future in as_completed(tasks):
   name=tasks[future];row=future.result();row.setdefault('kind','match');records[name].append(row);done+=1
   if row['kind']=='match' or done%20==0:print('DIAGNOSTIC',done,name,row['kind'],row.get('result'),row.get('first_touch_seconds'),flush=True)
   (output/(name+'.json')).write_text(json.dumps({'name':name,'model':str(models[name]),'records':records[name],'complete':len(records[name])==44},indent=2)+'\n')
 (output/'timing.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-began,'records':done,'workers':4},indent=2)+'\n')
 print('COMPLETE',done,'seconds',time.monotonic()-began,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only',nargs='*');a=p.parse_args();models={'untouched_baseline':ROOT/'runs/retained/baseline/V13.pt','frozen_A':ROOT/'model/V13.pt'}|{d.name:d/'candidate.pt' for d in a.runs.iterdir() if d.is_dir() and (d/'candidate.pt').exists()}
 if a.only:models={name:path for name,path in models.items() if name in a.only}
 evaluate(models,a.output)
