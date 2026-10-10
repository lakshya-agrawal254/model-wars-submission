"""One legal student-visited readout campaign; no runtime heuristic or architecture change."""
import sys,json,time,hashlib,random,copy,importlib.util
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training/memory_selective')]
from policy import Policy
from common import World
from runtime import decode
sys.path.insert(0,str(Path(__file__).parent));from targets import target
GEO=ROOT/'runs/geometric-retained/latest.pt';BASE=ROOT/'runs/retained/baseline/V13.pt';A=ROOT/'runs/retained/selected/candidate.pt';OUT=ROOT/'runs/interception-readout-v1';COLS=[3,4,5,9,10,11]
def load(path):
 p=Policy().eval();p.load_state_dict(torch.load(path,weights_only=True)['state_dict']);return p

def calibration():
 table=dict(speeds=[500,1000,1500,2000],steers=[0,.25,.5,.75,1],curvatures=[],method='Real RocketSim flatground 0.3sec constant legal Chase; abs yaw delta/pathlength; no boost/handbrake')
 rows=[]
 for speed in table['speeds']:
  values=[]
  for steering in table['steers']:
   w=World(9610011,'G',0)
   try:
    c=w.state.cars[w.our];c.physics.position=np.array([0,-1000,17],np.float32);c.physics.euler_angles=np.zeros(3,np.float32);c.physics.linear_velocity=np.array([speed,0,0],np.float32);c.physics.angular_velocity=np.zeros(3,np.float32);c.wheels_with_contact=(True,True,True,True);w.state.ball.position=np.array([0,3000,93.15],np.float32);w.state=w.engine.set_state(w.state,{});distance=0;previous=w.state.cars[w.our].physics.position.copy();angle0=np.arctan2(w.state.cars[w.our].physics.forward[1],w.state.cars[w.our].physics.forward[0])
    for _ in range(36):
     old,touch,goal=w.step(decode(1,steering),decode(0,0.),ticks=1);current=w.state.cars[w.our].physics.position.copy();distance+=np.linalg.norm(current-previous);previous=current
    f=w.state.cars[w.our].physics.forward;angle=np.arctan2(f[1],f[0])-angle0;angle=float(np.arctan2(np.sin(angle),np.cos(angle)));curvature=abs(angle)/max(distance,1);values.append(curvature);rows.append(dict(speed=speed,steer=steering,yaw_delta=angle,path_length=float(distance),curvature=curvature))
   finally:w.close()
  table['curvatures'].append(values)
 assert all(r['yaw_delta']>0 for r in rows if r['steer']>0),'Positive steering must turn toward positive official right coordinate'
 assert all(np.isfinite(table['curvatures']).ravel());return table,rows

@torch.inference_mode()
def collect(path,opponent,stage,seed,side,index,rate,seconds,table=None):
 torch.set_num_threads(1);p=load(path);op=load(opponent);w=World(seed,stage,side,index);h=None;oh=None;xs=[];targets=[];steering_masks=[];stall_masks=[];record=[];counts=np.zeros(4,int);first=None;last_qualified=-10000;contact_windows=[];last_contact=-10000;recovery_rows=[];stationary=0.;circles=0;positions=[];productive=0;backward=0;meaningless=0;goal_count=[0,0];last_ballgap=None
 try:
  for n in range(rate*seconds):
   x=w.observe(w.our);ox=w.observe(w.enemy);l,s,h=p(torch.tensor(x[None]),h);ol,os,oh=op(torch.tensor(ox[None]),oh);mode=int(l.argmax());st=float(s);counts[mode]+=1;car=w.state.cars[w.our];speed=float(np.linalg.norm(car.physics.linear_velocity));distance=float(x[6]*6000);ground=bool(car.on_ground) and car.physics.position[2]<45;ground_ball=w.state.ball.position[2]<180;near_plane=abs(x[2])<.035
   stall=bool(mode==0 and ground and ground_ball and near_plane and x[8]>.5 and speed<200 and distance>800)
   mask=bool(ground and ground_ball and near_plane and distance>220 and (mode==1 or stall));label=st;tau=0.
   if table is not None and mask:label,tau=target(car.physics.position,car.physics.linear_velocity,car.physics.forward,car.physics.right,w.state.ball.position,w.state.ball.linear_velocity,speed,table)
   if table is not None:xs.append(x.copy());targets.append(label);steering_masks.append(mask);stall_masks.append(stall)
   old,touch,goal=w.step(decode(mode,st),decode(int(ol.argmax()),float(os)),ticks=120//rate);car=w.state.cars[w.our].physics;speed_after=float(np.linalg.norm(car.linear_velocity));gap=float(np.linalg.norm(car.position-w.state.ball.position));stationary+=int(speed_after<100)/rate
   positions.append((car.position.copy(),gap))
   if len(positions)==rate*5:
    length=sum(np.linalg.norm(b[0]-a[0]) for a,b in zip(positions,positions[1:]));circles+=int(length>500 and np.linalg.norm(positions[-1][0]-positions[0][0])<.1*length and positions[-1][1]>400 and positions[0][1]-positions[-1][1]<100);positions=[]
   if touch:
    if first is None:first=n/rate
    if w.state.tick_count-last_qualified>=30:
     dv=(1 if side==0 else -1)*float(w.state.ball.linear_velocity[1]-old.ball.linear_velocity[1]);productive+=int(dv>40);backward+=int(dv< -40);meaningless+=int(abs(dv)<=40);last_qualified=w.state.tick_count;contact_windows.append(dict(at=n/rate,direction='goalward' if dv>40 else 'backward' if dv< -40 else 'meaningless'))
    last_contact=n
   if 0<n-last_contact<=rate*5:recovery_rows.append((speed_after<100,mode==0,last_ballgap is not None and gap<last_ballgap))
   if goal is not None:goal_count[int(goal!=side)]+=1;w.reset('E',side,w.state.tick_count);positions=[] # h and oh persist
   last_ballgap=gap
   if n%rate==0 or stall:record.append(dict(second=n/rate,mode=mode,speed=speed_after,ballgap=gap,steer=st,stationary=speed_after<100,stall_target=stall,ground=ground))
  summary=dict(stage=stage,seed=seed,side=side,index=index,rate=rate,seconds=seconds,opponent=str(opponent.relative_to(ROOT)),first_touch=first,counts=counts.tolist(),stationary_seconds=stationary,circling_windows=circles,productive_contacts=productive,backward_contacts=backward,meaningless_contacts=meaningless,goals_for=goal_count[0],goals_against=goal_count[1],stall_targets=sum(stall_masks),postcontact_observed_seconds=len(recovery_rows)/rate,postcontact_stationary_seconds=sum(v[0] for v in recovery_rows)/rate,postcontact_neutral_seconds=sum(v[1] for v in recovery_rows)/rate,postcontact_decreasing_gap_fraction=float(np.mean([v[2] for v in recovery_rows])) if recovery_rows else None)
  return summary,np.asarray(xs),np.asarray(targets,dtype=np.float32),np.asarray(steering_masks),np.asarray(stall_masks),record
 finally:w.close()
def batch(tasks):
 with ProcessPoolExecutor(max_workers=4) as pool:return [f.result() for f in [pool.submit(collect,*t) for t in tasks]]
def aggregate(records):
 return dict(trials=len(records),contacts=sum(r['first_touch'] is not None for r in records),stationary_seconds=sum(r['stationary_seconds'] for r in records),circling_windows=sum(r['circling_windows'] for r in records),productive_contacts=sum(r['productive_contacts'] for r in records),backward_contacts=sum(r['backward_contacts'] for r in records),goals_for=sum(r['goals_for'] for r in records),goals_against=sum(r['goals_against'] for r in records),postcontact_stationary_seconds=sum(r['postcontact_stationary_seconds'] for r in records),postcontact_neutral_seconds=sum(r['postcontact_neutral_seconds'] for r in records),conditional_first_contact_median=float(np.median([r['first_touch'] for r in records if r['first_touch'] is not None])) if any(r['first_touch'] is not None for r in records) else None)
def diagnostics(path,seed=9620011):
 tasks=[]
 for rate in [60,120]:
  for j,(stage,index) in enumerate([('G',None),('M',None),('E',None)]+[('B',k) for k in range(5)]):
   for side in [0,1]:tasks.append((path,BASE,stage,seed+j*1009,side,index,rate,30))
 return [r[0] for r in batch(tasks)]
def main():
 assert not OUT.exists();OUT.mkdir();cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());deadline=datetime.fromisoformat(cfg['training_stop_UTC']).timestamp();assert time.time()<deadline;torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(9630011);np.random.seed(9630011);random.seed(9630011);start=time.monotonic();table,cal=calibration();(OUT/'calibration.json').write_text(json.dumps(dict(table=table,measurements=cal),indent=2));print('CALIBRATION VERIFIED',flush=True)
 before=diagnostics(GEO);(OUT/'before.json').write_text(json.dumps(before,indent=2));print('BEFORE',aggregate(before),flush=True)
 tasks=[]
 for rate in [60,120]:
  for j,(stage,index) in enumerate([('G',None),('M',None),('E',None),('O',None),('F',None),('R',None)]+[('B',k) for k in range(5)]):
   for side in [0,1]:tasks.append((GEO,[BASE,A,GEO][j%3],stage,9631011+j*1009,side,index,rate,20,table))
  for j,op in enumerate([BASE,A,GEO]):
   for side in [0,1]:tasks.append((GEO,op,'E',9641011+j*1009,side,None,rate,40,table))
 collected=batch(tasks);xs=[r[1] for r in collected];ys=[r[2] for r in collected];masks=[r[3] for r in collected];stalls=[r[4] for r in collected];meta=[r[0] for r in collected];(OUT/'data-records.json').write_text(json.dumps(meta,indent=2));np.savez_compressed(OUT/'data.npz',**{f'x{i}':x for i,x in enumerate(xs)},**{f'y{i}':x for i,x in enumerate(ys)},**{f'mask{i}':x for i,x in enumerate(masks)},**{f'stall{i}':x for i,x in enumerate(stalls)})
 reference=load(GEO);actor=load(GEO)
 for v in actor.encoder.parameters():v.requires_grad_(False)
 for v in actor.gru.parameters():v.requires_grad_(False)
 assert sum(v.numel() for v in actor.parameters() if v.requires_grad)==325
 features=[];reference_logits=[];reference_steer=[]
 with torch.inference_mode():
  for x in xs:
   z,h=reference.gru(reference.encoder(torch.tensor(x)).unsqueeze(0));features.append(z.squeeze(0).clone());o=reference.head(z.squeeze(0));reference_logits.append(o[:,:4].clone());reference_steer.append(o[:,4].tanh().clone())
 # Whole-trajectory heldout split within independent generation; evaluation seeds remain separate.
 held_seeds=set(sorted(set(r['seed'] for r in meta))[3::7]);val_ids=[i for i,r in enumerate(meta) if r['seed'] in held_seeds];train_ids=[i for i,r in enumerate(meta) if r['seed'] not in held_seeds];assert not {meta[i]['seed'] for i in val_ids}&{meta[i]['seed'] for i in train_ids}
 def stack(ids):return (torch.cat([features[i] for i in ids]),torch.tensor(np.concatenate([ys[i] for i in ids])),torch.tensor(np.concatenate([masks[i] for i in ids])),torch.tensor(np.concatenate([stalls[i] for i in ids])),torch.cat([reference_logits[i] for i in ids]),torch.cat([reference_steer[i] for i in ids]))
 train=stack(train_ids);val=stack(val_ids);opt=torch.optim.Adam(actor.head.parameters(),lr=cfg['learning_rate']);saved=torch.load(GEO,weights_only=True);old=saved['supervised_optimizer'];old_ids=old['param_groups'][0]['params'];assert len(old_ids)==8
 # Head weights/bias are final2parameters in the unchanged official actor.
 for param,key in zip(actor.head.parameters(),old_ids[-2:]):opt.state[param]={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in old['state'][key].items()}
 history=[];updates=0;check=[];initial_val=float(((val[5][val[2]]-val[1][val[2]])**2).mean())
 for stage in range(2):
  for _ in range(16):
   assert time.time()<deadline,'05:15IST speculative training stop';ids=torch.randperm(len(train[0]))[:4096];h,y,mask,stall,rl,rs=[v[ids] for v in train];opt.zero_grad(set_to_none=True);out=actor.head(h);l=out[:,:4];s=out[:,4].tanh();se=(s[mask]-y[mask]).square().mean() if mask.any() else s.sum()*0;ce=torch.nn.functional.cross_entropy(l[stall],torch.ones(int(stall.sum()),dtype=torch.long)) if stall.any() else l.sum()*0;outside=~stall;kl=torch.nn.functional.kl_div(l[outside].log_softmax(-1),rl[outside].softmax(-1),reduction='batchmean');anchor=(s-rs).square().mean();loss=se+.1*ce+.3*kl+.1*anchor;loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1);assert torch.isfinite(norm);opt.step();updates+=1
   history.append(dict(update=updates,steer_mse=float(se.detach()),stall_chase_ce=float(ce.detach()),outside_mode_kl=float(kl.detach()),steer_anchor_mse=float(anchor.detach()),gradient_norm=float(norm)))
  for name,value in actor.state_dict().items():
   if not name.startswith('head.'):assert torch.equal(value,reference.state_dict()[name])
  with torch.inference_mode():
   output=actor.head(val[0]);val_mse=float((output[:,4].tanh()[val[2]]-val[1][val[2]]).square().mean())
  state=dict(state_dict=actor.state_dict(),head_optimizer=opt.state_dict(),updates=updates,initial_supervised_updates=saved['updates'],new_ppo_decisions=0,collected_decisions=sum(len(x) for x in xs),training_config=cfg,torch_rng=torch.get_rng_state(),numpy_rng=(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),python_rng=random.getstate(),data_sha256=hashlib.sha256((OUT/'data.npz').read_bytes()).hexdigest(),starting_checkpoint_sha256=hashlib.sha256(GEO.read_bytes()).hexdigest(),history=history,validation_target_mse=val_mse,initial_validation_target_mse=initial_val,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('targets.py'),Path(__file__).with_name('protocol.json'),ROOT/'src/policy.py',ROOT/'training/sim_contract.py',ROOT/'training/geometric/world.py',ROOT/'training/anchored_final/world.py',ROOT/'training/memory_selective/common.py']})
  torch.save(state,OUT/f'stage-{stage+1}.pt');torch.save({'state_dict':actor.state_dict()},OUT/'candidate.pt');rows=diagnostics(OUT/'candidate.pt');(OUT/f'stage-{stage+1}-diagnostics.json').write_text(json.dumps(rows,indent=2));base=aggregate(before);now=aggregate(rows);worse=now['contacts']<base['contacts'] or now['stationary_seconds']>base['stationary_seconds']+5 or now['circling_windows']>base['circling_windows']+2;check.append(dict(stage=stage+1,updates=updates,validation_target_mse=val_mse,initial_validation_target_mse=initial_val,diagnostics=now,early_stop=worse));print('STAGE',check[-1],flush=True)
  if worse:break
 torch.save(state,OUT/'latest.pt');summary=dict(configuration=cfg,new_ppo_decisions=0,collected_decisions=state['collected_decisions'],optimizer_updates=updates,stages=check,baseline=aggregate(before),candidate=check[-1]['diagnostics'],rejected_movement=check[-1]['early_stop'],initial_validation_target_mse=initial_val,final_validation_target_mse=val_mse,elapsed_seconds=time.monotonic()-start);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('CAMPAIGN COMPLETE',summary,flush=True)
if __name__=='__main__':main()
