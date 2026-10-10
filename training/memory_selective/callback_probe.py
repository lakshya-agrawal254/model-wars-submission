"""Bounded 120Hz diagnostic, fresh seeds; no policy changes or fitting."""
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src')]
from common import Policy,World
from runtime import decode
@torch.inference_mode()
def trial(path,stage,seed,side,index,rate=120):
 torch.set_num_threads(1);p=Policy().eval();p.load_state_dict(torch.load(path,weights_only=True)['state_dict']);o=Policy().eval();o.load_state_dict(torch.load(ROOT/'runs/retained/baseline/V13.pt',weights_only=True)['state_dict']);w=World(seed,stage,side,index);h=None;oh=None;first=None;counts=np.zeros(4,int);cap=30 if stage=='A' else 90
 try:
  for tick in range(cap*rate):
   x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=o(torch.tensor(ox[None]),oh);m=int(l.argmax());counts[m]+=1;old,touch,goal=w.step(decode(m,float(s)),decode(int(ol.argmax()),float(os)),ticks=120//rate)
   if first is None and touch:first=tick/rate
   if goal is not None:break
  return dict(stage=stage,seed=seed,side=side,index=index,first_touch_seconds=first,counts=counts.tolist(),callback_Hz=rate,domain='Real RocketSim; predictions unavailable; NOT actual RLBot')
 finally:w.close()
def main(rate=120):
 out=ROOT/('runs/memory-selective-callback-probe' if rate==120 else 'runs/memory-selective-callback-probe-60Hz');assert not out.exists();out.mkdir();start=time.monotonic();records={'candidate':[],'geometric':[]}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,path in [('candidate',ROOT/'runs/memory-selective-24576/candidate.pt'),('geometric',ROOT/'runs/geometric-retained/latest.pt')]:
   for i in range(2):
    for side in [0,1]:tasks[pool.submit(trial,path,'A',8410011+i*1009,side,None,rate)]=name
   for i in range(5):
    for side in [0,1]:tasks[pool.submit(trial,path,'B',8411011+i*1009,side,i,rate)]=name
  for n,(future,name) in enumerate(tasks.items(),1):records[name].append(future.result());print(str(rate)+'HZ PROBE',n,name,records[name][-1]['first_touch_seconds'],flush=True)
 (out/'records.json').write_text(json.dumps(records,indent=2)+'\n');s={name:{stage:{'contacts':sum(r['first_touch_seconds'] is not None for r in rows if r['stage']==stage),'trials':sum(r['stage']==stage for r in rows),'median_first_contact_seconds_conditional':float(np.median([r['first_touch_seconds'] for r in rows if r['stage']==stage and r['first_touch_seconds'] is not None])) if any(r['stage']==stage and r['first_touch_seconds'] is not None for r in rows) else None} for stage in ['A','B']} for name,rows in records.items()};s['elapsed_seconds']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print('PROBE COMPLETE',s,flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--rate',type=int,choices=[60,120],default=120);args=parser.parse_args();main(args.rate)
