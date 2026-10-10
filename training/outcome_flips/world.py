"""Only original 13D inputs. Privileged state is used for training-only scoring."""
import sys,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training/interception_readout')]
from forecast_world import ForecastWorld
from policy import Policy
from runtime import decode
GEO=ROOT/'runs/geometric-retained/latest.pt';BASE=ROOT/'runs/retained/baseline/V13.pt'
def load(path=GEO):
 p=Policy().eval();p.load_state_dict(torch.load(path,weights_only=True)['state_dict'],strict=True);return p

def setup(scene):
 side=scene['side'];kind=scene['kind'];sign=1 if side==0 else -1;w=ForecastWorld(scene['seed'],'B' if kind=='kickoff' else 'A',side,scene.get('index'))
 if kind=='kickoff' and 'enemy_index' in scene:
  from rlgym.rocket_league.state_mutators import KickoffMutator
  e=w.state.cars[w.enemy];i=scene['enemy_index'];blue=e.team_num==0;e.physics.position=(KickoffMutator.SPAWN_BLUE_POS if blue else KickoffMutator.SPAWN_ORANGE_POS)[i].copy();e.physics.euler_angles=np.array([0,(KickoffMutator.SPAWN_BLUE_YAW if blue else KickoffMutator.SPAWN_ORANGE_YAW)[i],0],np.float32);w.state=w.engine.set_state(w.state,{})
 if kind!='kickoff':
  ball=np.array([scene.get('ball_x',0),sign*scene.get('ball_y',0),scene.get('height',93.15)],np.float32);gap=scene['gap'];speed=scene['speed'];yaw=sign*math.pi/2+scene.get('yaw_error',0)
  c=w.state.cars[w.our];c.physics.position=ball-np.array([0,sign*gap,ball[2]-17],np.float32);c.physics.euler_angles=np.array([0,yaw,0],np.float32);c.physics.linear_velocity=np.array([math.cos(yaw)*speed,math.sin(yaw)*speed,0],np.float32);c.physics.angular_velocity=np.zeros(3,np.float32);c.wheels_with_contact=(True,True,True,True)
  w.state.ball.position=ball;w.state.ball.linear_velocity=np.array([scene.get('ball_vx',0),sign*scene.get('ball_vy',0),scene.get('ball_vz',0)],np.float32)
  e=w.state.cars[w.enemy];e.physics.position=np.array([scene.get('enemy_x',1000),sign*min(scene.get('ball_y',0)+1700,4300),17],np.float32);e.physics.euler_angles=np.array([0,-sign*math.pi/2,0],np.float32);e.physics.linear_velocity=np.array([0,-sign*scene.get('enemy_speed',400),0],np.float32);e.physics.angular_velocity=np.zeros(3,np.float32);e.wheels_with_contact=(True,True,True,True);w.state=w.engine.set_state(w.state,{});w.kickoff=False
 return w

@torch.inference_mode()
def trajectory(scene,schedule=None,path=GEO,rate=60,record=False,disable_second_press=False,driver_rate=None):
 torch.set_num_threads(1);p=load(path);op=load(BASE);w=setup(scene);h=oh=None;xs=[];modes=[];steers=[];windows=[];counts=np.zeros(4,int);first=None;enemyfirst=None;events=[];contacts=[];recoveries=[];pending=None;stable=0;stationary=fallen=0.;positions=[];circles=0;trace=[];direction=1 if scene['side']==0 else -1;last_touch=-1000;last_ground=True;last_flip=False;last_mode=0;mode_transitions=[];driver_rate=rate if driver_rate is None else driver_rate;last_driver_t=None;last_l=last_s=last_ol=last_os=None
 assert rate%driver_rate==0
 try:
  for n in range(rate*7):
   t=n/rate;x=w.observe(w.our);ox=w.observe(w.enemy)
   if n%(rate//driver_rate)==0:
    # Own-reference teacher may run60Hz while collecting actual120Hz student observations.
    # Teacher delta is MEASURED at its callbacks; studentx[12] remains actualrate delta.
    dx=x.copy();dox=ox.copy();dt=1/60 if last_driver_t is None else t-last_driver_t;dx[12]=dt*60;dox[12]=dt*60;last_driver_t=t
    last_l,last_s,h=p(torch.tensor(dx[None]),h);last_ol,last_os,oh=op(torch.tensor(dox[None]),oh)
   mode=int(last_l.argmax());steer=float(last_s);window=False
   if schedule:
    onset,jump,release,dodge=schedule;window=onset-.15<=t<=onset+jump+release+dodge+.35
    dt=t-onset
    if 0<=dt<jump:mode=2
    elif jump<=dt<jump+release:mode=1
    elif jump+release<=dt<jump+release+dodge:mode=1 if disable_second_press else 3
   if record:xs.append(x);modes.append(mode);steers.append(steer);windows.append(window)
   if mode!=last_mode:mode_transitions.append(dict(t=t,mode=mode));last_mode=mode
   counts[mode]+=1;v_before=w.state.ball.linear_velocity.copy();before=w.state.cars[w.our];before_flipping=bool(w.engine._cars[w.our].get_state().is_flipping);old,touch,goal=w.step(decode(mode,steer),decode(int(last_ol.argmax()),float(last_os)),ticks=120//rate)
   raw=w.engine._cars[w.our].get_state();c=w.state.cars[w.our];speed=float(np.linalg.norm(c.physics.linear_velocity));upright=float(c.physics.up[2]);ground=bool(raw.is_on_ground);actualflip=bool(raw.is_flipping)
   stationary+=int(speed<100)/rate;fallen+=int(c.physics.position[2]<60 and upright<.5)/rate
   if actualflip and not before_flipping:events.append(dict(t=t,ballgap=float(np.linalg.norm(old.ball.position-before.physics.position)),mode=mode));pending=t;stable=0
   if pending is not None:
    stable=stable+1 if ground and upright>.85 and speed>150 else 0
    if stable>=max(1,round(.1*rate)):recoveries.append(dict(flip_t=pending,stable_recovery_seconds=t-pending));pending=None
   if touch:
    if first is None:first=t
    if n-last_touch>=round(.25*rate):
     v=w.state.ball.linear_velocity;dv=v-v_before;goalward=direction*float(v[1]);last_flip_t=events[-1]['t'] if events else None
     contacts.append(dict(t=t,ball_speed=float(np.linalg.norm(v)),goalward_velocity=goalward,goalward_delta_velocity=direction*float(dv[1]),lateral_velocity=float(v[0]),useful_flip_contact=last_flip_t is not None and 0<=t-last_flip_t<=.7 and goalward>300 and direction*float(dv[1])>100))
     last_touch=n
   if w.state.cars[w.enemy].ball_touches and enemyfirst is None:enemyfirst=t
   gap=float(np.linalg.norm(w.state.ball.position-c.physics.position));positions.append((c.physics.position.copy(),gap))
   if len(positions)>=5*rate:
    length=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(positions,positions[1:]));circles+=int(length>500 and np.linalg.norm(positions[-1][0]-positions[0][0])<.1*length and positions[-1][1]>400 and positions[0][1]-positions[-1][1]<100);positions=[]
   if record:trace.append(dict(t=t,mode=mode,jump=bool(decode(mode,steer)[5]),actual_flip=actualflip,on_ground=ground,upright=upright,speed=speed,ballgap=gap))
   if goal is not None:break
  stats=dict(scene=scene,rate=rate,schedule=schedule,first_contact=first,enemy_first_contact=enemyfirst,first_touch_advantage=None if first is None or enemyfirst is None else enemyfirst-first,flip_events=events,contacts=contacts,recoveries=recoveries,stationary_seconds=stationary,fallen_seconds=fallen,circling_windows=circles,counts=counts.tolist(),goals_for=w.score[scene['side']],goals_against=w.score[1-scene['side']],seconds_observed=(n+1)/rate,mode_transitions=mode_transitions)
  arrays=dict(x=np.asarray(xs,np.float32),m=np.asarray(modes,np.int64),s=np.asarray(steers,np.float32),window=np.asarray(windows,bool)) if record else None
  return stats,arrays,trace
 finally:w.close()

def scenes(seed):
 rows=[]
 for side in [0,1]:
  for i in range(5):rows.append(dict(kind='kickoff',side=side,index=i,seed=seed+side*10000+i*1009))
  for j,(kind,kw) in enumerate([
   ('moving',dict(speed=850,gap=1400,ball_vx=300,ball_vy=150)),
   ('moving',dict(speed=1100,gap=1800,ball_vx=-250,yaw_error=.1)),
   ('shooting',dict(speed=800,gap=900,ball_y=3300,ball_x=200,enemy_x=-900,enemy_speed=0)),
   ('shooting',dict(speed=1250,gap=1500,ball_y=2900,ball_x=-250,yaw_error=-.12,enemy_x=900,enemy_speed=0)),
   ('low_air',dict(speed=1000,gap=1200,height=220,ball_vz=250)),
   ('low_air',dict(speed=800,gap=1500,height=300,ball_vx=100,ball_vz=200)),
   ('misaligned',dict(speed=1200,gap=1800,yaw_error=.22,ball_vx=-180)),
   ('ground_preferred',dict(speed=550,gap=450,yaw_error=-.08)),
   ('ground_preferred',dict(speed=1250,gap=600,ball_vy=-200))]):rows.append(dict(kind=kind,side=side,seed=seed+20000+side*10000+j*1009,**kw))
 return rows
