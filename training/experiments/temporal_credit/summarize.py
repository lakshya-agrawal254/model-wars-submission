"""Caps are unresolved, not draws; conditional touch medians label survivor bias."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def interval(k,n):
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-r,c+r]
def summarize(folder):
 out={}
 for p in sorted(folder.glob('*.json')):
  d=json.loads(p.read_text())
  if 'records' not in d or not isinstance(d['records'],list):continue
  rows=d['records'];entry={'records':len(rows),'complete':d['complete']}
  for kind in ['ground','kickoff']:
   rs=[r for r in rows if r['kind']==kind];touches=[r['first_touch_seconds'] for r in rs if r['contacted']];n=len(rs)
   if not n:continue
   entry[kind]={'contacts':len(touches),'trials':n,'contact_rate':len(touches)/n,'wilson_95':interval(len(touches),n),'median_seconds_among_contacts':float(np.median(touches)) if touches else None,'restricted_mean_touch_or_cap_seconds':float(np.mean([r['first_touch_seconds'] if r['contacted'] else r['seconds_observed'] for r in rs])),'observed_flip_starts':sum(r['flip_starts'] for r in rs),'far_ball_flip_starts_proxy':sum(r['far_ball_flip_starts'] for r in rs),'decision_counts':np.sum([r['counts'] for r in rs],axis=0).tolist(),'mean_latency_ms':float(np.mean([r['latency_ms_mean'] for r in rs]))}
   if kind=='kickoff':entry[kind]['contacts_by_position']={str(i):sum(r['contacted'] for r in rs if r['kickoff_index']==i) for i in range(5)}
  rs=[r for r in rows if r['kind']=='match'];complete=[r for r in rs if r['end'] not in ['safety_cap','censored_overtime_cap']]
  entry['matches']={'completed_WLD':[sum(r['result']==v for r in complete) for v in ['W','L','D']],'unresolved':len(rs)-len(complete),'observed_goals_for':sum(r['goals_for'] for r in rs),'observed_goals_against':sum(r['goals_against'] for r in rs),'completed_goals_for':sum(r['goals_for'] for r in complete),'completed_goals_against':sum(r['goals_against'] for r in complete),'conservative_score':sum(1 if r['result']=='W' else .5 if r['result']=='D' else 0 for r in complete)/len(rs) if rs else None,'win_fraction_all_trials_wilson_95':interval(sum(r['result']=='W' for r in complete),len(rs)) if rs else None}
  out[d['name']]=entry
 return out
if __name__=='__main__':
 folder=ROOT/'runs/temporal-credit-evaluation-v1';out=summarize(folder);(folder/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
