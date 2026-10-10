"""One conditional student-prefix phase-memory refinement, no PPO/runtime hooks."""
from world import *
sys.path.insert(0,str(Path(__file__).parent))
from train import diagnostics,agg
from search import useful
from student_search import CURRENT
import json,time,hashlib,random,datetime
from concurrent.futures import ProcessPoolExecutor
from torch.nn import functional as F
OUT=ROOT/'runs/outcome-flip-refinement-v1';COLS=[3,4,5,9,10,11]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2))
def collect(args):
 row,rate=args;scene=row['scene'];schedule=row['best']['schedule'] if row['selected'] else None;base,bx,_=trajectory(scene,path=CURRENT,rate=rate,record=True,driver_rate=60);bx['window'][:]=False
 if schedule:
  onset,jump,release,dodge=schedule;ts=np.arange(len(bx['x']))/rate;bx['window']=(ts>=onset-.2)&(ts<=onset+jump+release+dodge+.4)
 result=[('own_driving',base,bx)]
 if schedule:
  r,x,_=trajectory(scene,schedule,path=CURRENT,rate=rate,record=True,driver_rate=60)
  if useful(r,base):result.append(('positive',r,x))
 return result

def main():
 assert not OUT.exists();OUT.mkdir();torch.set_num_threads(1);torch.manual_seed(9968011);np.random.seed(9968011);random.seed(9968011);rng=random.Random(9968011);rows=json.loads((ROOT/'runs/outcome-flip-student-search-v1/records.json').read_text());data={};records=[]
 with ProcessPoolExecutor(max_workers=4) as pool:result=list(pool.map(collect,[(r,hz) for r in rows for hz in [60,120]]))
 for group in result:
  for role,record,array in group:
   i=len(records);records.append(dict(role=role,record=record));data.update({f'{k}{i}':v for k,v in array.items()})
 np.savez_compressed(OUT/'data.npz',**data);dump('data-records.json',records);p=load(CURRENT);ref=load(CURRENT);initial=torch.load(CURRENT,weights_only=True);opt=torch.optim.Adam([{'params':list(p.encoder.parameters())+list(p.gru.parameters()),'lr':.001},{'params':list(p.head.parameters()),'lr':.003}]);opt.load_state_dict(initial['supervised_optimizer']);opt.param_groups[0]['lr']=.001;opt.param_groups[1]['lr']=.003;channels=torch.argsort(ref.head.weight[4].abs())[:3].tolist();forecast_weights=ref.encoder[0].weight[:,COLS].detach().clone();sequences=[]
 with torch.no_grad():
  for i,r in enumerate(records):
   x=torch.tensor(data['x'+str(i)]);m=torch.tensor(data['m'+str(i)]);window=torch.tensor(data['window'+str(i)]);rl,rs,_=ref(x);phase=torch.zeros(len(x),dtype=torch.long);phase[m==2]=1;phase[m==3]=3
   jump=torch.where(m==2)[0];dodge=torch.where(m==3)[0]
   if len(jump) and len(dodge):phase[(torch.arange(len(x))>jump[-1])&(torch.arange(len(x))<dodge[0])&(m==1)]=2
   target=torch.full((len(x),3),-.5)
   for k in range(1,4):target[phase==k,k-1]=.5
   unsuitable=(x[:,0]<0)|(torch.atan2(x[:,1],x[:,0]).abs()>.45)|(x[:,7]*2300<600)|(x[:,6]*6000<250)
   sequences.append((x,m,window,rl.detach().clone(),rs.detach().clone(),phase,target,unsuitable,r['role']))
 positives=[i for i,v in enumerate(sequences) if v[-1]=='positive'];drive=[i for i,v in enumerate(sequences) if v[-1]=='own_driving'];baseline=json.loads((ROOT/'runs/outcome-flip-learning-v1/baseline-diagnostics.json').read_text());protocol=json.loads((ROOT/'training/outcome_flips/closed_loop_protocol.json').read_text());history=[];seen=0;begin=time.monotonic();stop=None;source_sha={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'training/outcome_flips').glob('*') if f.is_file()};print('REFINEMENT DATA',len(records),'positive',len(positives),'channels',channels,flush=True)
 for update in range(1,33):
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,10,23,45,tzinfo=datetime.timezone.utc):stop='Deadline stop';break
  batch=rng.sample(positives,min(6,len(positives)))+rng.sample(drive,min(6,len(drive)));opt.zero_grad();losses=[]
  for i in batch:
   x,m,window,rl,rs,phase,target,unsuitable,role=sequences[i];z,h=p.gru(p.encoder(x).unsqueeze(0));memory=z.squeeze(0);o=p.head(memory);l=o[:,:4];s=torch.tanh(o[:,4]);zero=l.sum()*0
   if role=='positive':
    ce=torch.stack([F.cross_entropy(l[window&(m==k)],m[window&(m==k)]) for k in [1,2,3] if (window&(m==k)).any()]).mean();phase_terms=[(memory[window&(phase==k)][:,channels]-target[window&(phase==k)]).square().mean() for k in [0,1,2,3] if (window&(phase==k)).any()];aux=torch.stack(phase_terms).mean()
    event=window&((m==2)|(m==3));chosen=l[event].gather(1,m[event,None]).squeeze();others=l[event].clone();others.scatter_(1,m[event,None],-1e6);margin=F.relu(.5+others.max(-1).values-chosen).mean();loss=ce+margin+2*aux
   else:
    mask=unsuitable&~window;loss=.1*F.cross_entropy(l[mask],m[mask]) if mask.any() else zero
   losses.append(loss+3*(s-rs).square().mean());seen+=len(x)
  loss=torch.stack(losses).mean();loss.backward();p.encoder[0].weight.grad[:,COLS]=0;norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.0));assert np.isfinite(norm);opt.step()
  with torch.no_grad():p.encoder[0].weight[:,COLS]=forecast_weights
  history.append(dict(update=update,total_curriculum_updates=128+update,loss=float(loss.detach()),gradient_norm=norm,presentations=seen))
  if update%8==0:print('REFINE',history[-1],flush=True)
  if update in [16,32]:
   saved=dict(state_dict=p.state_dict(),supervised_optimizer=opt.state_dict(),updates=update,total_curriculum_updates=128+update,configuration=protocol,phase_channels=channels,torch_rng=torch.get_rng_state(),numpy_rng=dict(bit_generator=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),pos=int(np.random.get_state()[2]),has_gauss=int(np.random.get_state()[3]),cached_gaussian=float(np.random.get_state()[4])),python_rng=random.getstate(),sampler_rng=rng.getstate(),history=history,initial_checkpoint_sha256=sha(CURRENT),geometric_initial_sha256=initial['initial_checkpoint_sha256'],data_sha256=sha(OUT/'data.npz'),source_sha256=source_sha,new_ppo_decisions=0,supervised_presentations=seen)
   torch.save(saved,OUT/f'update-{update}.pt');torch.save({'state_dict':p.state_dict()},OUT/f'actor-{update}.pt');candidate=diagnostics(OUT/f'actor-{update}.pt',60);dump(f'diagnostic-{update}.json',candidate);a=agg(baseline['60']);b=agg(candidate);reasons=[]
   if b['ground_contacts']<a['ground_contacts']-.2*a['ground_trials']:reasons.append('Groundcontact regression>20pct')
   if b['kickoff_contacts']<a['kickoff_contacts']-2:reasons.append('Kickoff regression>2/10')
   if b['stationary_seconds']>a['stationary_seconds']+3:reasons.append('Stationary regression')
   if b['fallen_seconds']>a['fallen_seconds']+3:reasons.append('Fall/recovery regression')
   if b['circling_windows']>a['circling_windows']+2:reasons.append('Circling regression')
   if b['repeated_flip_trials']:reasons.append('Repeated flips')
   print('REFINE GATE',update,b,reasons,flush=True)
   if reasons:stop=';'.join(reasons);break
   if b['trials_with_useful_flip_contact']>=3:stop='Useful learned flips; stopforindependentmatches';break
 assert 'saved' in locals();torch.save(saved,OUT/'latest.pt');torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');more=diagnostics(OUT/'candidate.pt',120);dump('candidate-120-diagnostics.json',more);report=dict(total_curriculum_updates=saved['total_curriculum_updates'],refinement_updates=saved['updates'],stop=stop or '32updatecap',candidate60=agg(candidate),candidate120=agg(more),phase_channels=channels,supervised_presentations=seen,new_ppo_decisions=0,elapsed_seconds=time.monotonic()-begin,candidate_sha256=sha(OUT/'candidate.pt'),native='UNTESTED',promoted=False);dump('result.json',report);print('REFINE RESULT',report,flush=True)
if __name__=='__main__':main()
