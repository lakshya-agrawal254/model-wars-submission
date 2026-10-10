"""Balanced whole-sequence curriculum; all forcing strictly confined to collection."""
from world import *
from search import useful
import json,time,hashlib,random,copy,datetime
from concurrent.futures import ProcessPoolExecutor
from torch.nn import functional as F
OUT=ROOT/'runs/outcome-flip-learning-v1';COLS=[3,4,5,9,10,11]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(name,v):(OUT/name).write_text(json.dumps(v,indent=2))
def collected(row,rate):
 scene=row['scene'];selected=row.get('selected',False);schedule=row['best']['schedule'] if selected else None
 base,bx,_=trajectory(scene,rate=rate,record=True,driver_rate=60)
 baseline_window=np.zeros(len(bx['x']),bool)
 if schedule:
  onset,jump,release,dodge=schedule;ts=np.arange(len(bx['x']))/rate;baseline_window=(ts>=onset-.2)&(ts<=onset+jump+release+dodge+.4)
 bx['window']=baseline_window
 results=[('own_driving',base,bx)]
 if selected:
  r,x,trace=trajectory(scene,schedule,rate=rate,record=True,driver_rate=60)
  # actual physics and outcome gate at this rate, never assume60Hz selection transfers.
  if useful(r,base):results.append(('positive',r,x))
 return results

def fresh_scenes():
 rows=[]
 for side in [0,1]:
  for i in range(5):rows.append(dict(kind='kickoff',side=side,index=i,enemy_index=(i+2)%5,seed=9950011+side*10000+i*1009))
  for j,(kind,kw) in enumerate([
   ('fresh_ground',dict(speed=700,gap=1100,yaw_error=.06)),
   ('fresh_ground',dict(speed=1200,gap=2100,yaw_error=-.08)),
   ('fresh_moving',dict(speed=950,gap=1600,ball_vx=-150,ball_vy=100)),
   ('fresh_shooting',dict(speed=950,gap=1050,ball_y=3200,ball_x=-100,yaw_error=.05,enemy_x=900,enemy_speed=0))]):rows.append(dict(kind=kind,side=side,seed=9952011+side*10000+j*1009,**kw))
 return rows

def diagnostics(path,rate):
 tasks=[(s,path,rate) for s in fresh_scenes()]
 with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(diagnostic_one,tasks))
 return rows

def diagnostic_one(t):
 scene,path,rate=t;r,*_=trajectory(scene,path=path,rate=rate);return r

def agg(rows):
 kicks=[r for r in rows if r['scene']['kind']=='kickoff'];ground=[r for r in rows if r['scene']['kind']!='kickoff'];return dict(trials=len(rows),contacts=sum(r['first_contact'] is not None for r in rows),ground_contacts=sum(r['first_contact'] is not None for r in ground),ground_trials=len(ground),kickoff_contacts=sum(r['first_contact'] is not None for r in kicks),kickoff_trials=len(kicks),flips=sum(len(r['flip_events']) for r in rows),useful_flip_contacts=sum(c['useful_flip_contact'] for r in rows for c in r['contacts']),trials_with_useful_flip_contact=sum(any(c['useful_flip_contact'] for c in r['contacts']) for r in rows),stationary_seconds=sum(r['stationary_seconds'] for r in rows),fallen_seconds=sum(r['fallen_seconds'] for r in rows),circling_windows=sum(r['circling_windows'] for r in rows),repeated_flip_trials=sum(len(r['flip_events'])>2 for r in rows),goals_for=sum(r['goals_for'] for r in rows),goals_against=sum(r['goals_against'] for r in rows),median_first_contact=float(np.median([r['first_contact'] for r in rows if r['first_contact'] is not None])) if any(r['first_contact'] is not None for r in rows) else None)

def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1);torch.manual_seed(9940011);np.random.seed(9940011);random.seed(9940011)
 attribution=json.loads((ROOT/'runs/outcome-flip-attribution-v1/records.json').read_text());negatives=json.loads((ROOT/'runs/outcome-flip-negatives-v1/search.json').read_text());attribution.extend(dict(scene=r['scene'],selected=False) for r in negatives if r['best'] is None)
 data={};metadata=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(collection_one,[(r,hz) for r in attribution for hz in [60,120]]))
 for group in results:
  for role,record,array in group:
   i=len(metadata);metadata.append(dict(role=role,record=record));data.update({f'{k}{i}':v for k,v in array.items()})
 np.savez_compressed(OUT/'data.npz',**data);save_json('data-records.json',metadata);print('DATA',len(metadata),'positive',sum(r['role']=='positive' for r in metadata),'rows',sum(len(v) for k,v in data.items() if k.startswith('x')),flush=True)
 baseline={str(hz):diagnostics(GEO,hz) for hz in [60,120]};save_json('baseline-diagnostics.json',baseline)
 g=load();p=load();opt=torch.optim.Adam([{'params':list(p.encoder.parameters())+list(p.gru.parameters()),'lr':.0005},{'params':list(p.head.parameters()),'lr':.003}]);forecast_weights=g.encoder[0].weight[:,COLS].detach().clone();seq=[]
 with torch.no_grad():
  for i,record in enumerate(metadata):
   x=torch.tensor(data['x'+str(i)]);m=torch.tensor(data['m'+str(i)]);mask=torch.tensor(data['window'+str(i)]);rl,rs,_=g(x);seq.append((x,m,mask,rl.detach().clone(),rs.detach().clone(),record['role']))
 positives=[i for i,s in enumerate(seq) if s[-1]=='positive'];driving=[i for i,s in enumerate(seq) if s[-1]=='own_driving'];rng=random.Random(9940011);history=[];decisions_seen=0;begin=time.monotonic();stop=None;protocol=json.loads((ROOT/'training/outcome_flips/protocol.json').read_text());source_sha={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'training/outcome_flips').glob('*') if f.is_file()};initial=sha(GEO)
 if not positives:save_json('result.json',dict(rejected='No positive qualifying demonstrations',optimizer_updates=0));return
 for update in range(1,129):
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,10,23,45,tzinfo=datetime.timezone.utc):stop='Deadline speculative stop';break
  indices=rng.sample(positives,min(6,len(positives)))+rng.sample(driving,min(6,len(driving)));opt.zero_grad();losses=[];correct=labelcount=0
  for i in indices:
   x,m,window,rl,rs,role=seq[i];l,s,h=p(x);outside=~window;zero=l.sum()*0
   if role=='positive':
    terms=[F.cross_entropy(l[window&(m==k)],m[window&(m==k)]) for k in [1,2,3] if (window&(m==k)).any()];ce=torch.stack(terms).mean();correct+=int((l[window].argmax(-1)==m[window]).sum());labelcount+=int(window.sum())
   else:ce=.5*F.cross_entropy(l[outside],m[outside]) if outside.any() else zero
   kl=.1*F.kl_div(F.log_softmax(l[outside],-1),F.softmax(rl[outside],-1),reduction='batchmean') if outside.any() else zero
   mse=2*(s-rs).square().mean();losses.append(ce+kl+mse);decisions_seen+=len(x)
  loss=torch.stack(losses).mean();loss.backward();p.encoder[0].weight.grad[:,COLS]=0;norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.0));assert np.isfinite(norm);opt.step()
  with torch.no_grad():p.encoder[0].weight[:,COLS]=forecast_weights
  history.append(dict(update=update,loss=float(loss.detach()),unclipped_gradient_norm=norm,window_accuracy=correct/max(1,labelcount),presentations=decisions_seen))
  if update%16==0:print('UPDATE',history[-1],flush=True)
  if update in protocol['supervised']['checks']:
   saved=dict(state_dict=p.state_dict(),supervised_optimizer=opt.state_dict(),updates=update,configuration=protocol,torch_rng=torch.get_rng_state(),numpy_rng=dict(bit_generator=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),pos=int(np.random.get_state()[2]),has_gauss=int(np.random.get_state()[3]),cached_gaussian=float(np.random.get_state()[4])),python_rng=random.getstate(),sampler_rng=rng.getstate(),history=history,data_sha256=sha(OUT/'data.npz'),source_sha256=source_sha,initial_checkpoint_sha256=initial,new_ppo_decisions=0,supervised_presentations=decisions_seen)
   torch.save(saved,OUT/f'update-{update}.pt');torch.save({'state_dict':p.state_dict()},OUT/f'actor-{update}.pt');candidate_rows=diagnostics(OUT/f'actor-{update}.pt',60);save_json(f'diagnostic-{update}.json',candidate_rows);a=agg(baseline['60']);b=agg(candidate_rows);reasons=[]
   if b['ground_contacts']<a['ground_contacts']-.2*a['ground_trials']:reasons.append('Independent ground contacts regression>20pct')
   if b['kickoff_contacts']<a['kickoff_contacts']-2:reasons.append('Kickoff contact regression>2/10')
   if b['stationary_seconds']>a['stationary_seconds']+3:reasons.append('Stationary regression>3s')
   if b['circling_windows']>a['circling_windows']+2:reasons.append('Circling regression')
   if b['repeated_flip_trials']:reasons.append('Repeated flipping')
   if b['fallen_seconds']>a['fallen_seconds']+3:reasons.append('Fall/recovery regression>3s')
   print('GATE',update,b,reasons,flush=True)
   if reasons:stop='; '.join(reasons);break
   if b['trials_with_useful_flip_contact']>=3 and b['flips']>=3:stop='Provisional learned useful flips; stop fitting for fullmatch screen';break
 last=history[-1]['update'];assert (OUT/f'update-{last}.pt').exists(),'Deadline stopped betweenpreservations; use preceding completecheckpoint';torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');more=diagnostics(OUT/'candidate.pt',120);save_json('candidate-120-diagnostics.json',more);result=dict(optimizer_updates=last,stop=stop or '128updatecap',baseline={k:agg(v) for k,v in baseline.items()},candidate60=agg(candidate_rows),candidate120=agg(more),supervised_presentations=decisions_seen,new_ppo_decisions=0,elapsed_seconds=time.monotonic()-begin,source_sha256=source_sha,candidate_sha256=sha(OUT/'candidate.pt'),native='UNTESTED',promoted=False);save_json('result.json',result);assert sha(GEO)==initial;print('RESULT',result,flush=True)
def collection_one(args):return collected(*args)
if __name__=='__main__':main()
