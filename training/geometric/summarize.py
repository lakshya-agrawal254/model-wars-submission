import json,math,sys
from pathlib import Path
import numpy as np

def wilson(k,n):
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [max(0,c-r),min(1,c+r)]
def summary(folder):
 result={}
 for p in folder.glob('*.json'):
  d=json.loads(p.read_text())
  if not isinstance(d.get('records'),list):continue
  rows=d['records'];entry={'complete':d['complete'],'records':len(rows)}
  for kind in ['ground','wide_ground','kickoff']:
   rs=[r for r in rows if r['kind']==kind]
   if not rs:continue
   times=[r['first_touch_seconds'] for r in rs if r['contacted']];counts=np.sum([r['counts'] for r in rs],axis=0);n=len(rs)
   entry[kind]={'contacts':len(times),'trials':n,'rate':len(times)/n,'wilson95':wilson(len(times),n),'median_touch_seconds_conditional':float(np.median(times)) if times else None,'mean_touch_or_cap_seconds':float(np.mean([r['first_touch_seconds'] if r['contacted'] else r['seconds_observed'] for r in rs])),'counts':counts.tolist(),'neutral_decision_fraction':int(counts[0])/int(counts.sum()),'jump_dodge_decisions':int(counts[2:].sum()),'actual_flip_starts':sum(r['flip_starts'] for r in rs),'far_ball_flip_proxy':sum(r['far_ball_flip_starts'] for r in rs),'latency_ms_mean':float(np.mean([r['latency_ms_mean'] for r in rs])),'precontact_steering_label_error':float(np.mean([r['precontact_mean_steering_label_error'] for r in rs]))}
   if kind=='kickoff':entry[kind]['contacts_by_position']={str(i):sum(r['contacted'] for r in rs if r['kickoff_index']==i) for i in range(5)}
  rs=[r for r in rows if r['kind']=='match'];completed=[r for r in rs if r['end'] not in ['safety_cap','censored_overtime_cap']]
  if rs:entry['matches']={'WLD':[sum(r['result']==v for r in completed) for v in ['W','L','D']],'unresolved':len(rs)-len(completed),'observed_GF_GA':[sum(r[k] for r in rs) for k in ['goals_for','goals_against']],'completed_GF_GA':[sum(r[k] for r in completed) for k in ['goals_for','goals_against']],'conservative_score':sum(1 if r['result']=='W' else .5 if r['result']=='D' else 0 for r in completed)/len(rs),'win_fraction_wilson95':wilson(sum(r['result']=='W' for r in completed),len(rs))}
  result[d['name']]=entry
 return result
if __name__=='__main__':
 folder=Path(sys.argv[1]);s=summary(folder);(folder/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
