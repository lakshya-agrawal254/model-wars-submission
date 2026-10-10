"""Deterministic actual RocketSim trials/matches, never called an RLBot game test."""
import argparse,json,sys,time,math
from pathlib import Path
import numpy as np,torch
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from policy import Policy
from runtime import decode
from sim_contract import World

def load(path):
 p=Policy().eval();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True);return p
@torch.inference_mode()
def trial(path,stage,seed,team,seconds=15):
 torch.set_num_threads(1);p=load(path);o=load(ROOT/'runs/retained/baseline/V13.pt');w=World(seed,stage,team);h=None;oh=None;first=None;counts=np.zeros(4,int);steers=[];lat=[]
 try:
  for t in range(int(seconds*60)):
   x=w.observe(w.our);ox=w.observe(w.enemy);start=time.perf_counter();l,s,h=p(torch.tensor(x[None,:]),h);lat.append((time.perf_counter()-start)*1000);mode=int(l.argmax());counts[mode]+=1;steers.append(float(s));ol,os,oh=o(torch.tensor(ox[None,:]),oh)
   old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   if touch and first is None:first=t/60
   if goal is not None:break
  return dict(seed=seed,team=team,first_touch_seconds=first,contacted=first is not None,counts=counts.tolist(),steer_abs_mean=float(np.mean(np.abs(steers))),latency_ms_mean=float(np.mean(lat)))
 finally:w.close()
@torch.inference_mode()
def match(path,seed,team,seconds=300,opponent_path=None):
 torch.set_num_threads(1);p=load(path);o=load(opponent_path or ROOT/'runs/retained/baseline/V13.pt');w=World(seed,'E',team);h=None;oh=None;active=0;ot=0;phase='regulation';first=None;counts=np.zeros(4,int);start=time.monotonic();end='safety_cap';kickoff_wait=0
 try:
  for tick in range(0,120*900,2):
   x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None,:]),h);ol,os,oh=o(torch.tensor(ox[None,:]),oh);mode=int(l.argmax());counts[mode]+=1
   old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   if first is None and touch:first=tick/120
   if w.kickoff:kickoff_wait+=2
   elif phase=='regulation':active+=2
   else:ot+=2
   if goal is not None:
    if phase=='overtime' or (active>=seconds*120 and w.score[0]!=w.score[1]):end='decisive_goal';break
    w.reset('E',team,w.state.tick_count)
   if phase=='regulation' and active>=seconds*120:
    b=w.state.ball;prev=old.ball
    floor=(b.position[2]<=95 and b.linear_velocity[2]<=1) or (b.position[2]<=110 and prev.linear_velocity[2]<0<=b.linear_velocity[2])
    if floor:
     if w.score[0]!=w.score[1]:end='regulation_floor_proxy';break
     phase='overtime';w.reset('E',team,w.state.tick_count)
   if phase=='overtime' and ot>=120*120:end='censored_overtime_cap';break
  gf,ga=w.score[team],w.score[1-team]
  return dict(seed=seed,team=team,goals_for=gf,goals_against=ga,result='W' if gf>ga else 'L' if gf<ga else 'D',end=end,active_regulation_seconds=active/120,active_overtime_seconds=ot/120,first_touch_seconds=first,kickoff_wait_seconds=kickoff_wait/120,counts=counts.tolist(),elapsed_seconds=time.monotonic()-start,domain='RocketSim; both agents use explicitly absent prediction fields, not native RLBot')
 finally:w.close()

def main(path,output,kind,seed,pairs,opponent=None):
 assert not output.exists();rows=[];began=time.monotonic();fun=trial if kind!='matches' else match
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks=[pool.submit(fun,path,kind if kind!='matches' else seed+i*1009,seed+i*1009,side) if kind!='matches' else pool.submit(match,path,seed+i*1009,side,300,opponent) for i in range(pairs) for side in [0,1]]
  for f in as_completed(tasks):
   r=f.result();rows.append(r);print('EVAL',len(rows),r.get('result'),r.get('goals_for'),r.get('goals_against'),r.get('first_touch_seconds'),flush=True)
 rows.sort(key=lambda r:(r['seed'],r['team']));d=dict(model=str(path),kind=kind,seed=seed,paired_sides=True,opponent=str(opponent or ROOT/'runs/retained/baseline/V13.pt'),records=rows,elapsed_seconds=time.monotonic()-began)
 if kind=='matches':
  d.update(wins=sum(r['result']=='W' for r in rows),losses=sum(r['result']=='L' for r in rows),draws=sum(r['result']=='D' for r in rows),goals_for=sum(r['goals_for'] for r in rows),goals_against=sum(r['goals_against'] for r in rows),censored=sum(r['end'].startswith('censored') or r['end']=='safety_cap' for r in rows))
 else:d.update(contact_rate=sum(r['contacted'] for r in rows)/len(rows),median_first_touch_seconds=float(np.median([r['first_touch_seconds'] for r in rows if r['contacted']])) if any(r['contacted'] for r in rows) else None)
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(d,indent=2));print('COMPLETE',d.keys(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--opponent',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--kind',choices=['A','B','C','D','matches'],default='A');p.add_argument('--seed',type=int,required=True);p.add_argument('--pairs',type=int,default=8);a=p.parse_args();main(a.model,a.output,a.kind,a.seed,a.pairs,a.opponent)
