"""Real-physics, training-only legal decoder sequence proof. No runtime controller."""
import sys,json,math,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training/interception_readout')]
from forecast_world import ForecastWorld
from runtime import decode
from policy import Policy
GEO=ROOT/'runs/geometric-retained/latest.pt'
def load():
 p=Policy().eval();p.load_state_dict(torch.load(GEO,weights_only=True)['state_dict']);return p

@torch.inference_mode()
def trial(rate,side,kind,sequence,seed=9830011,index=4,model=None):
 torch.set_num_threads(1);p=load() if model is None else model;w=ForecastWorld(seed,'B' if kind=='kickoff' else 'A',side,index if kind=='kickoff' else None);h=None;xs=[];modes=[];steers=[];trace=[];first=None;flips=[];lands=[];contact=[];previous_flip=False;previous_ground=True;start=time.monotonic()
 try:
  if kind!='kickoff':
   c=w.state.cars[w.our];sign=1 if side==0 else -1
   c.physics.position=np.array([0,-sign*1200,17],np.float32);c.physics.euler_angles=np.array([0,sign*math.pi/2,0],np.float32);c.physics.linear_velocity=np.array([0,sign*1000,0],np.float32);c.physics.angular_velocity=np.zeros(3,np.float32);c.wheels_with_contact=(True,True,True,True)
   w.state.ball.position=np.array([0,sign*800,93.15],np.float32);w.state=w.engine.set_state(w.state,{})
  # Settling/pursuit prefix precedes all interventions; nearball starts flip earlier than kickoff.
  onset=.3 if kind!='kickoff' else 1.5
  schedule={'release':[(.1,2),(.1,1),(.05,3)],'no_release':[(.2,2),(.05,3)],'late':[(.1,2),(1.6,1),(.05,3)],'ground_dodge':[(.25,3)]}
  for n in range(rate*8):
   x=w.observe(w.our);l,s,h=p(torch.tensor(x[None]),h);mode=int(l.argmax());steer=float(s);t=n/rate
   if sequence!='geometric':
    mode=1;steer=float(s) if kind=='kickoff' else 0.
    if sequence in schedule:
     dt=t-onset
     if dt>=0:
      for duration,target in schedule[sequence]:
       if dt<duration:mode=target;break
       dt-=duration
   xs.append(x);modes.append(mode);steers.append(steer)
   before=w.state.ball.linear_velocity.copy();old,touch,goal=w.step(decode(mode,steer),decode(0,0.),ticks=120//rate)
   raw=w.engine._cars[w.our].get_state();c=w.state.cars[w.our];flipping=bool(raw.is_flipping);ground=bool(raw.is_on_ground)
   if flipping and not previous_flip:flips.append(t)
   if ground and not previous_ground:lands.append(t)
   if touch:
    if first is None:first=t
    contact.append(dict(t=t,dv=(w.state.ball.linear_velocity-before).tolist(),ball_speed=float(np.linalg.norm(w.state.ball.linear_velocity))))
   trace.append(dict(t=t,mode=mode,z=float(c.physics.position[2]),speed=float(np.linalg.norm(c.physics.linear_velocity)),on_ground=ground,has_jumped=bool(raw.has_jumped),has_flipped=bool(raw.has_flipped),is_flipping=flipping,air_time_since_jump=float(raw.air_time_since_jump)))
   previous_flip=flipping;previous_ground=ground
   if goal is not None:break
  stationary=sum(r['speed']<100 for r in trace)/rate;falls=sum(r['z']>50 and not r['on_ground'] for r in trace)/rate
  recovery=[next((t-f for t in lands if t>f),None) for f in flips]
  return dict(rate=rate,side=side,kind=kind,sequence=sequence,first_contact=first,flip_events=flips,landing_recovery_seconds=recovery,airborne_seconds=falls,stationary_seconds=stationary,contacts=contact,goals=w.score,trace=trace),np.asarray(xs,dtype=np.float32),np.asarray(modes),np.asarray(steers,dtype=np.float32)
 finally:w.close()
def main():
 out=ROOT/'runs/flip-feasibility-v1';assert not out.exists();out.mkdir();records=[];arrays={}
 for rate in [60,120]:
  for side in [0,1]:
   for kind in ['nearball','kickoff']:
    for seq in ['geometric','chase','release','no_release','late','ground_dodge']:
     r,x,m,s=trial(rate,side,kind,seq);i=len(records);records.append(r);arrays.update({f'x{i}':x,f'm{i}':m,f's{i}':s});print(rate,side,kind,seq,'flips',len(r['flip_events']),'contact',r['first_contact'],'recovery',r['landing_recovery_seconds'],flush=True)
 (out/'physics-proof.json').write_text(json.dumps(records,indent=2));np.savez_compressed(out/'chronological-sequences.npz',**arrays)
if __name__=='__main__':main()
