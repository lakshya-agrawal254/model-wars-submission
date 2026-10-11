"""Summarize actual native JSONL. No manufactured score, engine acknowledgements or causality."""
import sys,json,statistics
from pathlib import Path
rows=[]
for path in Path(sys.argv[1]).glob('*.jsonl'):
 for line in path.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
p=[r for r in rows if r.get('event')=='processed_packet'];errs=[r for r in rows if r.get('event')=='callback_error'];deltas=[r.get('callback_delta_seconds',r['observation13'][12]/60) for r in p];mode={m:sum(r['mode']==m for r in p) for m in ['Neutral','Chase','Jump','Front dodge']};stops=[r for r in p if sum(v*v for v in r['velocity'])<10000]
checks=[]
for r in p:
 raw=r.get('raw_head');checks.append(raw is None or ['Neutral','Chase','Jump','Front dodge'][max(range(4),key=lambda i:raw[0][i])]==r['mode'])
result=dict(native_records=len(p),logged_mode_counts_NOT_all_callbacks=mode,logged_stationary_records=len(stops),stationary_neutral=sum(r['mode']=='Neutral' for r in stops),stationary_chase=sum(r['mode']=='Chase' for r in stops),stationary_ball_absent=sum(r['observation13'][8]<.5 for r in stops),invalid_forecast_records=sum(r['observation13'][9]<.5 for r in p),median_callback_delta_seconds=statistics.median(deltas) if deltas else None,callback_errors=errs,logit_argmax_consistent=all(checks),teams=sorted(set(r['team'] for r in p)),checkpoints=[r.get('checkpoint') for r in rows if r.get('event')=='process_started'],limitations='Event/periodic samples are not a win rate or stationary-time estimate. Correlation is not causality. Preserve raw logs/video/full-match scores; game execution only verified by native evidence.')
print(json.dumps(result,indent=2))
