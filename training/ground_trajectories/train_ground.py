from world_ground import *
import hashlib,random,datetime
from concurrent.futures import ProcessPoolExecutor
from torch.nn import functional as F
OUT=ROOT/'runs/ground-trajectory-learning-v1';CFG=json.loads(Path(__file__).with_name('protocol.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh(seed,change=1):
 rows=scenes(seed)
 for r in rows:
  if r['kind']!='kickoff':r['gap']+=90*change;r['speed']+=60*change;r['yaw_error']=r.get('yaw_error',0)+.04*change;r['ball_vx']=r.get('ball_vx',0)+25*change
  else:r['enemy_index']=(r['index']+2)%5
 return rows

def trial(args):s,path,hz=args;return run(s,path,hz=hz)[0]
def diagnostics(path,seed=10101011,hz=60,change=1):
 with ProcessPoolExecutor(max_workers=4) as pool:return list(pool.map(trial,[(s,path,hz) for s in fresh(seed,change)]))
def collected(args):
 row,hz=args;scene=row['scene'];own,arr,_=run(scene,hz=hz,record=True);ret=[('own',own,arr)]
 if row['selected'] is not None:
  rec,x,_=run(scene,params=row['selected']['params'],hz=hz,record=True)
  from search_ground import useful
  if useful(rec,own):ret.append(('positive',rec,x))
 return ret

def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1);torch.manual_seed(10003011);np.random.seed(10003011);random.seed(10003011);rng=random.Random(10003011);begin=time.monotonic();rows=json.loads((ROOT/'runs/ground-trajectory-search-v1/records.json').read_text());records=[];data={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  groups=list(pool.map(collected,[(r,60) for r in rows]))
 for group in groups:
  for role,rec,a in group:
   i=len(records);records.append(dict(role=role,record=rec));data.update({f'{k}{i}':v for k,v in a.items()})
 np.savez_compressed(OUT/'data.npz',**data);(OUT/'data-records.json').write_text(json.dumps(records,indent=2));positive=[i for i,r in enumerate(records) if r['role']=='positive'];own=[i for i,r in enumerate(records) if r['role']=='own'];print('DATA',len(records),'positive',len(positive),'rows',sum(len(v) for k,v in data.items() if k.startswith('x')),flush=True)
 if len(positive)<4:(OUT/'result.json').write_text(json.dumps(dict(rejected='insufficient successful physicsdemonstrations',updates=0)));return
 baseline=diagnostics(START);(OUT/'development-control.json').write_text(json.dumps(baseline,indent=2));b=aggregate(baseline);p=load(START);ref=load(START);opt=torch.optim.Adam(p.parameters(),lr=CFG['learning_rate']);fixed=p.encoder[0].weight[:,[10,11]].detach().clone();seq=[]
 with torch.no_grad():
  for i,r in enumerate(records):
   x=torch.tensor(data['x'+str(i)]);l,s,_=ref(x);seq.append((x,torch.tensor(data['m'+str(i)]),torch.tensor(data['s'+str(i)]),torch.tensor(data['mask'+str(i)]),l.detach().clone(),s.detach().clone(),r['role']))
 history=[];seen=0;stop=None;source={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*') if f.is_file()};initial=sha(START);initialplain=sha(ROOT/'runs/outcome-flip-learning-v1/candidate.pt');deps=['training/outcome_flips/world.py','training/interception_readout/forecast_world.py','training/memory_selective/common.py','training/anchored_final/world.py','training/geometric/world.py','training/sim_contract.py','src/policy.py','src/runtime.py'];source.update({f:sha(ROOT/f) for f in deps});(OUT/'used-source').mkdir();
 for f in source:
  q=OUT/'used-source'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
 for update in range(1,CFG['max_updates']+1):
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(CFG['stop_UTC']):stop='Deadline';break
  indices=rng.sample(positive,min(4,len(positive)))+rng.sample(own,min(4,len(own)));opt.zero_grad();loss=[]
  for i in indices:
   x,m,target,mask,rl,rs,role=seq[i];xx=x.clone()
   # Training-only availability robustness; genuine forecasts kept otherwise.
   if rng.random()<.25:
    start=rng.randrange(len(xx));length=max(1,len(xx)//3);xx[start:start+length,3:6]=0;xx[start:start+length,9:12]=0
   l,s,h=p(xx);zero=l.sum()*0
   if role=='positive':
    steer=2*F.smooth_l1_loss(s[mask],target[mask]);mode=.2*F.cross_entropy(l[mask],m[mask]);outside=~mask;anchor=.5*(s[outside]-rs[outside]).square().mean() if outside.any() else zero
   else:steer=zero;mode=zero;anchor=.5*(s-rs).square().mean()
   kl=.2*F.kl_div(F.log_softmax(l,-1),F.softmax(rl,-1),reduction='batchmean');loss.append(steer+mode+anchor+kl);seen+=len(x)
  loss=torch.stack(loss).mean();loss.backward();p.encoder[0].weight.grad[:,[10,11]]=0;norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.));assert np.isfinite(norm);opt.step()
  with torch.no_grad():p.encoder[0].weight[:,[10,11]]=fixed
  history.append(dict(update=update,loss=float(loss.detach()),gradient_norm=norm,presentations=seen))
  if update in CFG['checks']:
   saved=dict(state_dict=p.state_dict(),supervised_optimizer=opt.state_dict(),updates=update,configuration=CFG,torch_rng=torch.get_rng_state(),numpy_rng=dict(bit_generator=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),pos=int(np.random.get_state()[2]),has_gauss=int(np.random.get_state()[3]),cached_gaussian=float(np.random.get_state()[4])),python_rng=random.getstate(),sampler_rng=rng.getstate(),history=history,data_sha256=sha(OUT/'data.npz'),source_sha256=source,initial_checkpoint_sha256=initial,initial_plain_sha256=initialplain,new_ppo_decisions=0,supervised_presentations=seen)
   torch.save(saved,OUT/f'update-{update}.pt');torch.save({'state_dict':p.state_dict()},OUT/f'actor-{update}.pt');now=diagnostics(OUT/f'actor-{update}.pt');(OUT/f'development-{update}.json').write_text(json.dumps(now,indent=2));a=aggregate(now);reasons=[]
   if a['contacts']<b['contacts']-1:reasons.append('contacts regression')
   if a['stationary']>b['stationary']+2:reasons.append('stationary regression')
   if a['circling']>b['circling']+1:reasons.append('circling regression')
   if a['fallen']>b['fallen']+2:reasons.append('fall regression')
   print('CHECK',update,a,reasons,flush=True)
   if reasons:stop=';'.join(reasons);break
 torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');assert sha(START)==initial
 report=dict(updates=saved['updates'],stop=stop or '96updatecap',baseline=b,candidate=aggregate(now),presentations=seen,collected_rows=sum(len(v) for k,v in data.items() if k.startswith('x')),positive_sequences=len(positive),new_PPO_decisions=0,elapsed_seconds=time.monotonic()-begin,candidate_SHA256=sha(OUT/'candidate.pt'),promoted=False,native='UNTESTED');(OUT/'result.json').write_text(json.dumps(report,indent=2));print('FINISHED',report,flush=True)
if __name__=='__main__':main()
