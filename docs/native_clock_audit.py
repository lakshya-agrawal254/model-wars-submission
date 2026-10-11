"""Read actual raw logs: never infer error-phase from a neighboring sample."""
import sys,json
from pathlib import Path
result=[]
for file in sorted(Path(sys.argv[1]).glob('*.jsonl')):
 rows=[json.loads(line) for line in file.read_text().splitlines() if line.strip()];bursts=[];group=None
 for i,r in enumerate(rows):
  if r.get('event')!='callback_error':group=None;continue
  key=(r.get('T'),r.get('frame'),r.get('error'))
  if group is None or group['key']!=key:
   preceding=next((v for v in reversed(rows[:i]) if v.get('event')=='processed_packet'),None)
   group=dict(key=key,count=0,first_monotonic=r.get('monotonic'),last_monotonic=r.get('monotonic'),phases=[],last_accepted_T=[],preceding_successful_sample=preceding);bursts.append(group)
  group['count']+=1;group['last_monotonic']=r.get('monotonic');group['phases'].append(r.get('phase_code',r.get('phase')));group['last_accepted_T'].append(r.get('last_accepted_T'))
 for b in bursts:
  phases=[p for p in b['phases'] if p is not None];b['classification']='UNRESOLVED: error records lack phase; repeated T/frame establishes stale/frozen reported state, not its cause' if len(phases)!=b['count'] else 'Observed packet phase(s): '+repr(sorted(set(phases),key=str))+'; check video/core logs, an Active phase alone does not prove advancing physics';b['wall_seconds']=b['last_monotonic']-b['first_monotonic'] if b['last_monotonic'] is not None and b['first_monotonic'] is not None else None
 result.append(dict(file=file.name,error_bursts=bursts))
assert result,'No raw native JSONL provided; cannot classify paused versus active.'
print(json.dumps(result,indent=2))
