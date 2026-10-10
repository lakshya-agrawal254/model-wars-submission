"""Fresh matched real RocketSim diagnostics, never native Rocket League acceptance."""
import sys,json,time,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training'),str(Path(__file__).parent)]
from policy import Policy
from runtime import decode
from world import World
from labels import targets
# Import existing declared full-match protocol without changing it.
import importlib.util
spec=importlib.util.spec_from_file_location('official_v13_match_evaluator',ROOT/'training/evaluate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def load(path):
 p=Policy().eval();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True);return p
@torch.inference_mode()
def trial(path,kind,seed,side,index=None):
 torch.set_num_threads(1);p=load(path);op=load(ROOT/'runs/retained/baseline/V13.pt');w=World(seed,{'ground':'A','wide_ground':'G','kickoff':'B'}[kind],side,index);h=None;oh=None;first=None;counts=np.zeros(4,int);flips=0;far_flips=0;lat=[];errors=[];seconds=90 if kind=='kickoff' else 30
 try:
  for tick in range(seconds*60):
   x=w.observe(w.our);ox=w.observe(w.enemy);began=time.perf_counter();l,s,h=p(torch.tensor(x[None,:]),h);lat.append((time.perf_counter()-began)*1000);ol,os,oh=op(torch.tensor(ox[None,:]),oh);mode=int(l.argmax());counts[mode]+=1
   if first is None:errors.append(abs(float(s)-float(targets(x)[1])))
   flipping=w.state.cars[w.our].is_flipping;old,touch,goal=w.step(decode(mode,float(s)),decode(int(ol.argmax()),float(os)))
   if not flipping and w.state.cars[w.our].is_flipping:flips+=1;far_flips+=int(np.linalg.norm(old.cars[w.our].physics.position-old.ball.position)>800)
   if touch and first is None:first=tick/60
   if goal is not None:break
  return dict(kind=kind,seed=seed,team=side,kickoff_index=index,seconds_observed=(tick+1)/60,first_touch_seconds=first,contacted=first is not None,counts=counts.tolist(),flip_starts=flips,far_ball_flip_starts=far_flips,far_flip_definition='actual flip starting >800UU from ball; proxy, not proven unnecessary',precontact_mean_steering_label_error=float(np.mean(errors)),latency_ms_mean=float(np.mean(lat)),domain='real RocketSim; prediction fields explicitly unavailable')
 finally:w.close()

def full_match(path,seed,side,opponent_path=None):
 torch.set_num_threads(1);return module.match(path,seed,side,300,opponent_path)

def run(output):
 assert not output.exists();output.mkdir(parents=True);models={'candidate':ROOT/'runs/geometric-supervised-v1/candidate.pt','retained_A':ROOT/'runs/retained/selected/candidate.pt','untouched_baseline':ROOT/'runs/retained/baseline/V13.pt'};records={name:[] for name in models};records['candidate_vs_retained_A']=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={}
  for name,path in models.items():
   for kind,seed in [('ground',1910011),('wide_ground',1911011)]:
    for i in range(12):
     for side in [0,1]:tasks[pool.submit(trial,path,kind,seed+i*1009,side)]=name
   for index in range(5):
    for repeat in range(2):
     for side in [0,1]:tasks[pool.submit(trial,path,'kickoff',1912011+repeat*10007+index*1009,side,index)]=name
   for i in range(4):
    for side in [0,1]:tasks[pool.submit(full_match,path,1913011+i*1009,side)]=name
  for i in range(4):
   for side in [0,1]:tasks[pool.submit(full_match,models['candidate'],1914011+i*1009,side,models['retained_A'])]='candidate_vs_retained_A'
  for count,future in enumerate(as_completed(tasks),1):
   name=tasks[future];r=future.result();r.setdefault('kind','match');records[name].append(r);required=8 if name=='candidate_vs_retained_A' else 76
   (output/(name+'.json')).write_text(json.dumps({'name':name,'model':str(models['candidate'] if name=='candidate_vs_retained_A' else models[name]),'records':records[name],'complete':len(records[name])==required,'opponent':'retained_A' if name=='candidate_vs_retained_A' else 'untouched official baseline'},indent=2)+'\n')
   if count%20==0 or r['kind']=='match':print('GEOMETRIC EVAL',count,name,r['kind'],r.get('result'),r.get('first_touch_seconds'),flush=True)
 timing={'elapsed_seconds':time.monotonic()-start,'records':sum(len(rs) for rs in records.values()),'workers':4};(output/'timing.json').write_text(json.dumps(timing,indent=2)+'\n');print('COMPLETE',timing,flush=True)
if __name__=='__main__':run(Path(sys.argv[1]))
