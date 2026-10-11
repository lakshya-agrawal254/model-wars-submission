"""One approved eight-update self-imitation symmetry test, no runtime controller/PPO."""
from world_ground import *
import random,hashlib,copy,datetime
from concurrent.futures import ProcessPoolExecutor
from torch.nn import functional as F
OUT=ROOT/'runs/mirrored-own-8-v1';INITIAL=ROOT/'runs/outcome-flip-learning-v1/candidate.pt';COLS=[3,4,5,9,10,11]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diag_job(args):scene,path,hz=args;return run(scene,path=path,hz=hz)[0]
def cases():
 out=scenes(10701011)
 for r in out:
  if r['kind']!='kickoff':r['gap']+=125;r['yaw_error']=r.get('yaw_error',0)-.045;r['ball_vx']=r.get('ball_vx',0)+35
  else:r['enemy_index']=(r['index']+3)%5
 return out
def diagnose(path,hz=60):
 with ProcessPoolExecutor(max_workers=4) as pool:return list(pool.map(diag_job,[(s,path,hz) for s in cases()]))
def main():
 assert not OUT.exists();OUT.mkdir();began=time.monotonic();torch.set_num_threads(1);torch.manual_seed(10721011);rng=random.Random(10721011)
 before=sha(INITIAL);metadata=json.loads((ROOT/'runs/ground-trajectory-learning-v1/data-records.json').read_text());data_path=ROOT/'runs/ground-trajectory-learning-v1/data.npz';data=np.load(data_path,allow_pickle=False);ids=[]
 for i,v in enumerate(metadata):
  r=v['record']
  if v['role']=='own' and r['hz']==60 and r['contact'] is not None and r['circling']==0 and r['stationary']<=.5 and r['GA']==0 and r['fallen']<=1 and any(c['delta_goalward']>100 and c['goalward']>100 for c in r['contacts']):ids.append(i)
 assert len(ids)>=8
 protocol=dict(initial_actor_SHA256=before,source_dataset=str(data_path.relative_to(ROOT)),source_dataset_SHA256=sha(data_path),selected_source_ids=ids,source='Only autonomous Outcome128 own trajectories, no searched controls or external teacher. Original and reflected full sequences are siblings; no timestep shuffle or hidden-state transplant.',objective='Chase-only mirrored steering self-imitation, original-side steering anchor, soft mode-distribution preservation on original/mirrored sequences. No unconditional Chase labels, no PPO/reward changes.',updates_cap=8,learning_rate=.0001,forecast_columns_preserved=COLS,hard_deadline_UTC='2026-10-11T00:07:00+00:00',budget='12min including screening/export; approval explicitly continues past prior training cutoff',promotion='No promotion/main replacement without independent competitive and native acceptance plus user approval',new_PPO_decisions=0)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));ref=load(INITIAL);p=load(INITIAL);seq=[];mismatch=0;total=0;maxlabel=0.
 with torch.no_grad():
  for i in ids:
   x=torch.tensor(data[f'x{i}']);recorded=torch.tensor(data[f'm{i}']);steer=torch.tensor(data[f's{i}']);l,s,_=ref(x);mismatch+=int((l.argmax(-1)!=recorded).sum());total+=len(x);maxlabel=max(maxlabel,float((s-steer).abs().max()));xx=x.clone();xx[:,[1,4]]*=-1;seq.append((x,xx,l.detach().clone(),s.detach().clone(),recorded))
 assert mismatch==0 and maxlabel<1e-5,(mismatch,maxlabel)
 print('LABELPARITY',len(ids),total,maxlabel,flush=True)
 baseline=diagnose(INITIAL);b=aggregate(baseline);(OUT/'development-control.json').write_text(json.dumps(baseline,indent=2));opt=torch.optim.Adam(p.parameters(),lr=.0001);fixed=p.encoder[0].weight[:,COLS].detach().clone();history=[];presentations=0;stop=None
 for update in range(1,9):
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,11,0,7,tzinfo=datetime.timezone.utc):stop='Hard deadline';break
  indices=rng.sample(range(len(seq)),4);opt.zero_grad();losses=[]
  for i in indices:
   x,xx,rl,rs,m=seq[i];l,s,_=p(x);ml,ms,_=p(xx);chase=m==1;zero=l.sum()*0
   symmetry=(ms[chase]+rs[chase]).square().mean() if chase.any() else zero
   anchor=(s[chase]-rs[chase]).square().mean() if chase.any() else zero
   modes=F.kl_div(F.log_softmax(l,-1),F.softmax(rl,-1),reduction='batchmean')+F.kl_div(F.log_softmax(ml,-1),F.softmax(rl,-1),reduction='batchmean')
   losses.append(symmetry+.5*anchor+.2*modes);presentations+=2*len(x)
  loss=torch.stack(losses).mean();loss.backward();p.encoder[0].weight.grad[:,COLS]=0;norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),.5));assert np.isfinite(norm);opt.step()
  with torch.no_grad():p.encoder[0].weight[:,COLS]=fixed
  history.append(dict(update=update,loss=float(loss.detach()),gradient_norm=norm,presentations=presentations))
  if update in [4,8]:
   full=dict(state_dict=p.state_dict(),supervised_optimizer=opt.state_dict(),updates=update,protocol=protocol,torch_rng=torch.get_rng_state(),sampler_rng=rng.getstate(),history=copy.deepcopy(history),supervised_presentations=presentations,new_PPO_decisions=0,source_sha256={Path(__file__).name:sha(__file__)})
   torch.save(full,OUT/f'update-{update}.pt');torch.save({'state_dict':p.state_dict()},OUT/f'actor-{update}.pt')
   rows=diagnose(OUT/f'actor-{update}.pt');a=aggregate(rows);(OUT/f'development-{update}.json').write_text(json.dumps(rows,indent=2));reasons=[]
   if a['contacts']<b['contacts']:reasons.append('Contact regression')
   if a['kickoff_contacts']<b['kickoff_contacts']:reasons.append('Kickoff regression')
   if a['stationary']>b['stationary']+.25:reasons.append('Stationary +.25s regression')
   if a['circling']>b['circling']:reasons.append('Circling regression')
   if a['fallen']>b['fallen']+.5:reasons.append('Falls +.5s regression')
   print('GATE',update,a,reasons,flush=True)
   if reasons:stop='; '.join(reasons);break
 assert history and history[-1]['update'] in [4,8]
 last=history[-1]['update'];torch.save(full,OUT/'latest.pt');torch.save({'state_dict':p.state_dict()},OUT/'candidate.pt');assert sha(INITIAL)==before
 report=dict(updates=last,stop=stop or '8updatecap',movement_pass=not stop,baseline=b,candidate=a,selected_sequences=len(ids),source_rows=total,supervised_presentations=presentations,new_PPO_decisions=0,label_mismatch=mismatch,label_max_error=maxlabel,elapsed_seconds=time.monotonic()-began,candidate_SHA256=sha(OUT/'candidate.pt'),native='UNTESTED',promoted=False)
 (OUT/'result.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True)
if __name__=='__main__':main()
