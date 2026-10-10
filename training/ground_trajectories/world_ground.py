"""Training/evaluation only: realphysics, exact13D, no exported controllers."""
from pathlib import Path
import sys,importlib.util,math,json,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('preserved_outcome_world',ROOT/'training/outcome_flips/world.py');ow=importlib.util.module_from_spec(spec);spec.loader.exec_module(ow)
load=ow.load;decode=ow.decode;GEO=ow.GEO;BASE=ow.BASE;START=ROOT/'runs/outcome-flip-learning-v1/latest.pt'

def scenes(seed):
 rows=[]
 for side in [0,1]:
  for i in range(5):rows.append(dict(kind='kickoff',side=side,index=i,enemy_index=(i+1)%5,seed=seed+side*10000+i*1009,prefix=2))
  for j,(kind,kw) in enumerate([
   ('curve',dict(speed=900,gap=1500,ball_vx=380,ball_vy=120,yaw_error=.45)),
   ('curve',dict(speed=1200,gap=1900,ball_vx=-420,ball_vy=100,yaw_error=-.35)),
   ('curve',dict(speed=600,gap=1000,ball_vx=250,ball_vy=-120,yaw_error=.8)),
   ('curve',dict(speed=1400,gap=2400,ball_vx=-300,yaw_error=-.7)),
   ('uturn',dict(speed=0,gap=1300,yaw_error=2.7)),
   ('uturn',dict(speed=800,gap=1200,yaw_error=-2.6,ball_vx=100)),
   ('beside',dict(speed=1100,gap=1000,yaw_error=1.65,ball_vy=100)),
   ('beside',dict(speed=500,gap=800,yaw_error=-1.8)),
   ('shot',dict(speed=900,gap=1400,ball_y=2700,ball_x=400,ball_vx=-100,yaw_error=.2,enemy_x=-1000,enemy_speed=0)),
   ('shot',dict(speed=700,gap=1200,ball_y=3100,ball_x=-450,ball_vx=120,yaw_error=-.25,enemy_x=1000,enemy_speed=0)),
   ('shot',dict(speed=1200,gap=1800,ball_y=2200,ball_x=250,yaw_error=.1,enemy_x=-1200,enemy_speed=0)),
   ('postmiss',dict(speed=1300,gap=650,yaw_error=1.1,ball_vx=-120))]):rows.append(dict(kind=kind,side=side,seed=seed+20000+side*15000+j*1009,prefix=2 if j%2==0 else 3,**kw))
 return rows

class SearchDriver:
 def __init__(self,params):self.lead,self.gain,self.damp,self.lock=params;self.previous=None;self.turn=None;self.smooth=0.
 def __call__(self,x,dt):
  if x[8]<.5:return 0,0.
  current=x[:2]*6000;distance=max(1.,np.linalg.norm(current));speed=max(0.,x[7]*2300)
  # Only the13 legalfeatures feed this training-only controller.
  lead=float(np.clip(distance/max(speed+500,500),.08,self.lead)) if self.lead>0 else 0.
  future=x[3:5]*6000 if x[9]>.5 else current
  aim=current+(future-current)*(lead/2);angle=math.atan2(aim[1],aim[0]);bearing=math.atan2(current[1],current[0])
  derivative=0. if self.previous is None else math.atan2(math.sin(angle-self.previous),math.cos(angle-self.previous))/dt;self.previous=angle
  if self.lock:
   if self.turn is None and abs(bearing)>1.3:self.turn=1. if bearing>0 else -1.
   if self.turn is not None and abs(bearing)<.65:self.turn=None
  if self.turn is not None:s=self.turn
  else:
   # Speed-dependent feasible turn: lead trajectory search selects gains/damping.
   scale=.7+min(speed/1600,1.)*.6;s=float(np.clip(self.gain*angle*scale+self.damp*derivative,-1,1))
  self.smooth=.75*self.smooth+.25*s
  return 1,self.smooth

@torch.inference_mode()
def run(scene,path=START,params=None,hz=60,record=False,seconds=16):
 torch.set_num_threads(1);p=load(path);op=load(BASE);w=ow.setup(scene);h=oh=None;driver=SearchDriver(params) if params is not None else None;xs=[];ms=[];ss=[];masks=[];counts=np.zeros(4,int);positions=[];stat=fall=poststat=0.;circle=0;first=None;enemyfirst=None;contact=[];trace=[];last_touch=-1000;last_contact=-1000;turnstart=None;recovery=[];currentturn=False;goal_for=goal_against=0;prefix=scene.get('prefix',2);direction=1 if scene['side']==0 else -1
 try:
  for n in range(hz*seconds):
   t=n/hz;x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=op(torch.tensor(ox[None]),oh);mode=int(l.argmax());steer=float(s);controlled=driver is not None and t>=prefix
   if controlled:mode,steer=driver(x,1/hz)
   if record:xs.append(x);ms.append(mode);ss.append(steer);masks.append(controlled)
   counts[mode]+=1;old,touch,goal=w.step(decode(mode,steer),decode(int(ol.argmax()),float(os)),ticks=120//hz);c=w.state.cars[w.our].physics;speed=float(np.linalg.norm(c.linear_velocity));gap=float(np.linalg.norm(w.state.ball.position-c.position));stat+=int(speed<100)/hz;fall+=int(c.position[2]<60 and c.up[2]<.5)/hz
   if w.state.cars[w.enemy].ball_touches and enemyfirst is None:enemyfirst=t
   bearing=abs(math.atan2(x[1],x[0]));behind=x[8]>.5 and bearing>1.3
   if behind and not currentturn:turnstart=t;currentturn=True
   if currentturn and bearing<.55 and speed>150:recovery.append(t-turnstart);currentturn=False
   if touch:
    if first is None:first=t
    if n-last_touch>=round(.25*hz):
     v=w.state.ball.linear_velocity;contact.append(dict(t=t,speed=float(np.linalg.norm(v)),goalward=direction*float(v[1]),delta_goalward=direction*float(v[1]-old.ball.linear_velocity[1])));last_touch=n
    last_contact=n
   if 0<n-last_contact<=hz*3:poststat+=int(speed<100)/hz
   positions.append((c.position.copy(),gap))
   if len(positions)>=hz*5:
    length=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(positions,positions[1:]));circle+=int(length>500 and np.linalg.norm(positions[-1][0]-positions[0][0])<.1*length and gap>400 and positions[0][1]-gap<100);positions=[]
   if record and n%hz==0:trace.append(dict(t=t,mode=mode,steer=steer,speed=speed,gap=gap,bearing=bearing))
   if goal is not None:
    goal_for+=int(goal==scene['side']);goal_against+=int(goal!=scene['side']);break
  stats=dict(scene=scene,hz=hz,params=params,contact=first,enemy_contact=enemyfirst,contacts=contact,stationary=stat,fallen=fall,circling=circle,postcontact_stationary=poststat,recovery=recovery,counts=counts.tolist(),GF=goal_for,GA=goal_against,final_gap=gap,observed_seconds=(n+1)/hz)
  arrays=dict(x=np.asarray(xs,np.float32),m=np.asarray(ms,np.int64),s=np.asarray(ss,np.float32),mask=np.asarray(masks,bool)) if record else None
  return stats,arrays,trace
 finally:w.close()

def aggregate(rows):
 values=[r['contact'] for r in rows if r['contact'] is not None];kick=[r for r in rows if r['scene']['kind']=='kickoff'];return dict(trials=len(rows),contacts=len(values),kickoff_contacts=sum(r['contact'] is not None for r in kick),kickoff_trials=len(kick),median_contact=float(np.median(values)) if values else None,stationary=sum(r['stationary'] for r in rows),circling=sum(r['circling'] for r in rows),fallen=sum(r['fallen'] for r in rows),postcontact_stationary=sum(r['postcontact_stationary'] for r in rows),GF=sum(r['GF'] for r in rows),GA=sum(r['GA'] for r in rows),goalward_contacts=sum(c['delta_goalward']>40 for r in rows for c in r['contacts']),backward_contacts=sum(c['delta_goalward']< -40 for r in rows for c in r['contacts']),median_turn_recovery=float(np.median([v for r in rows for v in r['recovery']])) if any(r['recovery'] for r in rows) else None)
