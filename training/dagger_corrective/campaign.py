"""One bounded student-visited corrective fit. No PPO, external teacher or runtime override."""
import sys,json,time,hashlib,random
from pathlib import Path
import numpy as np,torch
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training/memory_selective')]
from policy import Policy
from common import World
from runtime import decode
GEO=ROOT/'runs/geometric-retained/latest.pt';BASE=ROOT/'runs/retained/baseline/V13.pt';A=ROOT/'runs/retained/selected/candidate.pt'
COLS=[3,4,5,9,10,11]
def load(path):
 p=Policy().eval();p.load_state_dict(torch.load(path,weights_only=True)['state_dict']);return p
# Targets depend ONLY on current legal observation and its chronological history.
def labels(x,reference_modes,reference_steering):
 angle=np.arctan2(x[:,1],x[:,0]);dt=np.maximum(x[:,12]/60,1/120);speed=x[:,7]*2300;distance=x[:,6]*6000
 da=np.zeros(len(x));da[1:]=np.arctan2(np.sin(np.diff(angle)),np.cos(np.diff(angle)))/dt[1:];da=np.clip(da,-2,2)
 # Bearing-rate extrapolation is not a privileged ball forecast; rotation and ball motion confound it.
 aim=np.arctan2(np.sin(angle+.15*da),np.cos(angle+.15*da));gain=1.25+ .75*np.clip(speed/1800,0,1)
 steer=np.clip(gain*aim,-1,1).astype('f');m=reference_modes.copy();st=reference_steering.copy()
 groundlike=(x[:,8]>.5)&(np.abs(x[:,2])<.04)&(distance>160)
 # Nominal speed-dependent radius heuristic, training label only. Cannot steer while Neutral.
 radius=180+.00022*speed**2
 coast=groundlike&(speed>1200)&(np.abs(angle)>1.25)&(distance<1.5*radius)
 m[groundlike]=1;m[coast]=0;st[groundlike]=steer[groundlike];m[x[:,8]<.5]=0
 return m,st,groundlike,coast
@torch.inference_mode()
def rollout(path,stage,seed,side,index,rate=60,seconds=20,dt_override=None,updates=1,sampled=False,collect=False):
 torch.set_num_threads(1);torch.manual_seed(seed);p=load(path);op=load(BASE);w=World(seed,stage,side,index);h=None;oh=None;rows=[];ms=[];ss=[];first=None;counts=np.zeros(4,int);stationary=0;touches=0;productive=0;backward=0;circles=0;positions=[];goals=[0,0];neutral_available=0;post_neutral=0;last_touch=-10000
 try:
  for n in range(seconds*rate):
   x=w.observe(w.our);ox=w.observe(w.enemy)
   if dt_override is not None:x[12]=dt_override
   for _ in range(updates):l,s,h=p(torch.tensor(x[None]),h)
   ol,os,oh=op(torch.tensor(ox[None]),oh);m=int(torch.distributions.Categorical(logits=l).sample()) if sampled else int(l.argmax());steer=float(s)
   if sampled:steer=float(torch.tanh(torch.atanh(s.clamp(-.999,.999))+torch.randn_like(s)*.15))
   counts[m]+=1;neutral_available+=int(m==0 and x[8]>.5);post_neutral+=int(m==0 and n-last_touch<rate*2)
   if collect:rows.append(x.copy());ms.append(m);ss.append(steer)
   old,touch,goal=w.step(decode(m,steer),decode(int(ol.argmax()),float(os)),ticks=120//rate)
   car=w.state.cars[w.our].physics;stationary+=int(np.linalg.norm(car.linear_velocity)<100)/rate;positions.append((car.position.copy(),float(np.linalg.norm(car.position-w.state.ball.position))))
   if len(positions)==rate*5:
    length=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(positions,positions[1:]));circles+=int(length>500 and np.linalg.norm(positions[-1][0]-positions[0][0])<.1*length and positions[-1][1]>400 and positions[0][1]-positions[-1][1]<100);positions=[]
   if touch:
    if first is None:first=n/rate
    touches+=1;last_touch=n;dv=(1 if side==0 else -1)*float(w.state.ball.linear_velocity[1]-old.ball.linear_velocity[1]);productive+=int(dv>40);backward+=int(dv< -40)
   if goal is not None:goals[int(goal!=side)]+=1;w.reset('E',side,w.state.tick_count) # hidden deliberately continues
  r=dict(stage=stage,seed=seed,side=side,index=index,rate=rate,dt_override=dt_override,updates_per_callback=updates,sampled=sampled,first_touch=first,counts=counts.tolist(),stationary_seconds=stationary,circling_windows=circles,touches=touches,productive_touch_ticks=productive,backward_touch_ticks=backward,goals_for=goals[0],goals_against=goals[1],neutral_ball_present=neutral_available,postcontact_neutral=post_neutral,seconds=seconds)
  return (r,np.array(rows),np.array(ms),np.array(ss)) if collect else r
 finally:w.close()
def batch(tasks):
 result=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(rollout,*t) for t in tasks]
  for f in fs:result.append(f.result())
 return result
def aggregate(rows):
 return dict(trials=len(rows),contacts=sum(r['first_touch'] is not None for r in rows),median_first_contact=float(np.median([r['first_touch'] for r in rows if r['first_touch'] is not None])) if any(r['first_touch'] is not None for r in rows) else None,stationary_seconds=sum(r['stationary_seconds'] for r in rows),circling_windows=sum(r['circling_windows'] for r in rows),goals_for=sum(r['goals_for'] for r in rows),goals_against=sum(r['goals_against'] for r in rows),neutral_ball_present=sum(r['neutral_ball_present'] for r in rows),postcontact_neutral=sum(r['postcontact_neutral'] for r in rows),productive_touch_ticks=sum(r['productive_touch_ticks'] for r in rows))
def diagnostics(path):
 tasks=[]
 for rate in [60,120]:
  for stage,index in [('G',None),('M',None)]+[('B',k) for k in range(5)]:
   for side in [0,1]:tasks.append((path,stage,9520011+(index or 0)*1009,side,index,rate,20))
 return batch(tasks)
def main():
 out=ROOT/'runs/dagger-corrective-v1';assert not out.exists();out.mkdir();torch.set_num_threads(2);torch.manual_seed(9510011);np.random.seed(9510011);random.seed(9510011);start=time.monotonic()
 # Factorial diagnostics: counterfactual inference ONLY, never exported as native behavior.
 tasks=[];names=[]
 for name,rate,dt,u in [('native60',60,None,1),('60_dt_half',60,.5,1),('60_double_gru',60,None,2),('60_both',60,.5,2),('native120',120,None,1),('120_dt_one',120,1.,1)]:
  for side in [0,1]:tasks.append((GEO,'B',9511011,side,0,rate,20,dt,u));names.append(name)
 factor=batch(tasks);(out/'timing-factorial.json').write_text(json.dumps([dict(configuration=n,**r) for n,r in zip(names,factor)],indent=2))
 before=diagnostics(GEO);(out/'before.json').write_text(json.dumps(before,indent=2));print('BEFORE',aggregate(before),flush=True)
 tasks=[]
 for rate in [60,120]:
  for j,(stage,index) in enumerate([('G',None),('M',None),('E',None),('O',None),('F',None),('R',None)]+[('B',k) for k in range(5)]):
   for side in [0,1]:tasks.append((GEO,stage,9530011+j*1009,side,index,rate,10,None,1,False,True))
 collected=batch(tasks);records=[];data=[];modes=[];steers=[];masks=[]
 for r,x,m,s in collected:
  tm,ts,mask,coast=labels(x,m,s);records.append(dict(**r,corrected_steer=int(np.sum(mask&(np.abs(ts-s)>.05))),coast_labels=int(coast.sum())));data.append(x);modes.append(tm);steers.append(ts);masks.append(mask)
 np.savez_compressed(out/'data.npz',**{f'x{i}':x for i,x in enumerate(data)},**{f'm{i}':x for i,x in enumerate(modes)},**{f's{i}':x for i,x in enumerate(steers)},**{f'mask{i}':x for i,x in enumerate(masks)})
 (out/'student-visited.json').write_text(json.dumps(records,indent=2));actor=load(GEO);saved=torch.load(GEO,weights_only=True);opt=torch.optim.Adam(actor.parameters(),lr=.00015);opt.load_state_dict(saved['supervised_optimizer'])
 for group in opt.param_groups:group['lr']=.00015
 reference=load(GEO);history=[];updates=0
 # One three-epoch campaign, 33 maximum optimizer updates. Each sequence starts at actual trial beginning.
 for epoch in range(3):
  order=torch.randperm(len(data));losses=[]
  for offset in range(0,len(order),4):
   ids=order[offset:offset+4];opt.zero_grad(set_to_none=True)
   for i in ids:
    i=int(i);x=torch.tensor(data[i]);l,s,h=actor(x);tm=torch.tensor(modes[i]);ts=torch.tensor(steers[i]);mask=torch.tensor(masks[i]);
    with torch.no_grad():rl,rs,rh=reference(x)
    ce=torch.nn.functional.cross_entropy(l[mask],tm[mask]) if mask.any() else l.sum()*0
    steer_mask=mask&(tm==1);se=(s[steer_mask]-ts[steer_mask]).square().mean() if steer_mask.any() else s.sum()*0
    # Small outside-target anchoring; never global Chase labels. Reference is own legal student.
    outside=~mask;preserve=torch.nn.functional.kl_div(l[outside].log_softmax(-1),rl[outside].softmax(-1),reduction='batchmean')+(s[outside]-rs[outside]).square().mean() if outside.any() else l.sum()*0
    loss=(.2*ce+2*se+.05*preserve)/len(ids);loss.backward();losses.append(float(loss.detach()))
   actor.encoder[0].weight.grad[:,COLS]=0;norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1);assert torch.isfinite(norm);opt.step();updates+=1
   with torch.no_grad():actor.encoder[0].weight[:,COLS]=0
  state=dict(state_dict=actor.state_dict(),supervised_optimizer=opt.state_dict(),updates=updates,completed_epochs=epoch+1,configuration=dict(lr=.00015,maximum_epochs=3,maximum_updates=33,new_ppo_decisions=0,source='own deterministic geometric student visited states',rate_mixture=[60,120]),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),data_sha256=hashlib.sha256((out/'data.npz').read_bytes()).hexdigest(),starting_checkpoint_sha256=hashlib.sha256(GEO.read_bytes()).hexdigest())
  torch.save(state,out/f'epoch-{epoch+1}.pt');torch.save({'state_dict':actor.state_dict()},out/'candidate.pt')
  check=diagnostics(out/'candidate.pt');(out/f'epoch-{epoch+1}-diagnostics.json').write_text(json.dumps(check,indent=2));old=aggregate(before);now=aggregate(check);worse=now['contacts']<old['contacts'] or now['stationary_seconds']>old['stationary_seconds']+10 or now['circling_windows']>old['circling_windows']+2
  history.append(dict(epoch=epoch+1,updates=updates,loss_mean=float(np.mean(losses)),diagnostics=now,early_stop=worse));print('EPOCH',history[-1],flush=True)
  if worse:break
 torch.save(state,out/'latest.pt');summary=dict(history=history,collected_real_decisions=sum(len(x) for x in data),new_ppo_decisions=0,optimizer_updates=updates,elapsed_seconds=time.monotonic()-start,baseline=aggregate(before),candidate=history[-1]['diagnostics'],rejected_mechanical=history[-1]['early_stop']);(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
