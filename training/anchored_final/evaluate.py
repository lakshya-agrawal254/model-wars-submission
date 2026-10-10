"""Identical full-match physics/clock, with observed attack/defence diagnostics."""
import argparse,json,sys,time,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'src'),str(ROOT/'training')]
from policy import Policy
from runtime import decode
from world import World
BASE=ROOT/'runs/retained/baseline/V13.pt';A=ROOT/'runs/retained/selected/candidate.pt';GEO=ROOT/'runs/geometric-retained/latest.pt'
def load(path):
 p=Policy().eval();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True);return p
@torch.inference_mode()
def match(path,opponent,seed,team):
 torch.set_num_threads(1);p=load(path);o=load(opponent);w=World(seed,'E',team);h=None;oh=None;active=0;ot=0;phase='regulation';first=None;counts=np.zeros(4,int);end='safety_cap';kickoff_wait=0;last_touch=None;conceded_own_last=0;last_qualified=-1000;productive=0;nonproductive=0;backward=0;clears=0;stationary=0;circles=0;lat=[];trajectory=[];began=time.monotonic()
 try:
  for tick in range(0,120*900,2):
   x=w.observe(w.our);ox=w.observe(w.enemy);stamp=time.perf_counter();l,s,h=p(torch.tensor(x[None,:]),h);lat.append((time.perf_counter()-stamp)*1000);ol,os,oh=o(torch.tensor(ox[None,:]),oh);mode=int(l.argmax());counts[mode]+=1;old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   their_touch=bool(w.state.cars[w.enemy].ball_touches)
   if touch or their_touch:last_touch=None if touch and their_touch else team if touch else 1-team
   if first is None and touch:first=tick/120
   if touch and w.state.tick_count-last_qualified>=30:
    sign=1 if team==0 else -1;delta=sign*float(w.state.ball.linear_velocity[1]-old.ball.linear_velocity[1]);productive+=int(delta>40);nonproductive+=int(abs(delta)<=40);backward+=int(delta < -40)
    clears+=int(sign*old.ball.position[1]<-2500 and sign*old.ball.linear_velocity[1]<0 and delta>40);last_qualified=w.state.tick_count
   car=w.state.cars[w.our].physics;stationary+=int(np.linalg.norm(car.linear_velocity)<100)*2/120;trajectory.append((car.position.copy(),float(np.linalg.norm(car.position-w.state.ball.position))))
   if len(trajectory)==300:
    length=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(trajectory,trajectory[1:]));displacement=np.linalg.norm(trajectory[-1][0]-trajectory[0][0]);circles+=int(length>500 and displacement<.1*length and trajectory[-1][1]>400 and trajectory[0][1]-trajectory[-1][1]<100);trajectory=[]
   if w.kickoff:kickoff_wait+=2
   elif phase=='regulation':active+=2
   else:ot+=2
   if goal is not None:
    if goal!=team and last_touch==team:conceded_own_last+=1
    if phase=='overtime' or (active>=300*120 and w.score[0]!=w.score[1]):end='decisive_goal';break
    w.reset('E',team,w.state.tick_count);last_touch=None;trajectory=[]
   if phase=='regulation' and active>=300*120:
    b=w.state.ball;prev=old.ball;floor=(b.position[2]<=95 and b.linear_velocity[2]<=1) or (b.position[2]<=110 and prev.linear_velocity[2]<0<=b.linear_velocity[2])
    if floor:
     if w.score[0]!=w.score[1]:end='regulation_floor_proxy';break
     phase='overtime';w.reset('E',team,w.state.tick_count);last_touch=None;trajectory=[]
   if phase=='overtime' and ot>=120*120:end='censored_overtime_cap';break
  gf,ga=w.score[team],w.score[1-team]
  return dict(kind='match',seed=seed,team=team,goals_for=gf,goals_against=ga,result='W' if gf>ga else 'L' if gf<ga else 'D',end=end,active_regulation_seconds=active/120,active_overtime_seconds=ot/120,first_touch_seconds=first,kickoff_wait_seconds=kickoff_wait/120,counts=counts.tolist(),elapsed_seconds=time.monotonic()-began,productive_contacts=productive,nonproductive_contacts=nonproductive,backward_contacts=backward,danger_clear_velocity_events_proxy=clears,goals_conceded_with_own_last_touch=conceded_own_last,stationary_seconds=stationary,circling_5second_windows_proxy=circles,inference_latency_ms_mean=float(np.mean(lat)),domain='Real RocketSim 60Hz decision callbacks, unavailable native predictions; NOT an RLBot game',diagnostic_limitations='Contact direction uses observed y-velocity changes; clears/circling/own-last-touch are proxies, not causal saves or proven own goals')
 finally:w.close()
@torch.inference_mode()
def trial(path,kind,seed,team,index=None):
 torch.set_num_threads(1);p=load(path);o=load(BASE);w=World(seed,'A' if kind=='ground' else 'B',team,index);h=None;oh=None;first=None;counts=np.zeros(4,int);seconds=30 if kind=='ground' else 90
 try:
  for tick in range(seconds*60):
   x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None,:]),h);ol,os,oh=o(torch.tensor(ox[None,:]),oh);mode=int(l.argmax());counts[mode]+=1;old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   if touch and first is None:first=tick/60
   if goal is not None:break
  return {'kind':kind,'seed':seed,'team':team,'index':index,'first_touch_seconds':first,'contacted':first is not None,'counts':counts.tolist(),'seconds_observed':(tick+1)/60}
 finally:w.close()

def run(candidate,output,confirmation=False):
 assert not output.exists();output.mkdir(parents=True);records={};start=time.monotonic();tag=711 if confirmation else 611;pairs=16 if confirmation else 4
 comparisons={'candidate_vs_baseline':(candidate,BASE,tag*10000+3011),'candidate_vs_A':(candidate,A,tag*10000+4011),'candidate_vs_geo':(candidate,GEO,tag*10000+5011)}
 if not confirmation:comparisons['geo_vs_baseline']=(GEO,BASE,6113011)
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,(path,op,seed) in comparisons.items():
   records[name]=[]
   for i in range(pairs):
    for side in [0,1]:tasks[pool.submit(match,path,op,seed+i*1009,side)]=(name,pairs*2)
  if not confirmation:
   for name,path in [('candidate',candidate),('geo',GEO),('A',A),('original',BASE)]:
    key=name+'_diagnostics';records[key]=[]
    for i in range(6):
     for side in [0,1]:tasks[pool.submit(trial,path,'ground',6110011+i*1009,side)]=(key,22)
    for index in range(5):
     for side in [0,1]:tasks[pool.submit(trial,path,'kickoff',6111011+index*1009,side,index)]=(key,22)
  for n,f in enumerate(as_completed(tasks),1):
   name,required=tasks[f];r=f.result();records[name].append(r);(output/(name+'.json')).write_text(json.dumps({'name':name,'candidate':str(candidate),'records':records[name],'complete':len(records[name])==required,'confirmation':confirmation},indent=2)+'\n')
   if r['kind']=='match' or n%20==0:print('ANCHOR EVAL',n,name,r.get('result'),r.get('goals_for'),r.get('goals_against'),r.get('first_touch_seconds'),flush=True)
 timing={'elapsed_seconds':time.monotonic()-start,'records':sum(len(rs) for rs in records.values()),'workers':4};(output/'timing.json').write_text(json.dumps(timing,indent=2)+'\n');print('COMPLETE',timing,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--confirmation',action='store_true');a=p.parse_args();run(a.candidate,a.output,a.confirmation)
