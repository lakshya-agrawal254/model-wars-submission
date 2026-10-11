"""Analyze real logs only; no simulator data accepted as native acceptance."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();files=sorted(a.logs.glob('*.jsonl'));assert files,'Run actual RLBot matches first; native logs are absent';rows=[]
for file in files:
 for line in file.read_text(encoding='utf-8').splitlines():rows.append((str(file),json.loads(line)))
starts=[r for f,r in rows if r.get('event')=='process_started'];packets=[(f,r) for f,r in rows if r.get('event')=='processed_packet'];errors=[r for f,r in rows if r.get('event')=='callback_error'];hashes=sorted(set(r['checkpoint'] for r in starts));dt=[r['observation13'][12] for f,r in packets if len(r.get('observation13',[]))==13 and r['observation13'][12]>0];rates=[]
for f in files:
 pr=[r for file,r in packets if file==str(f)]
 for old,new in zip(pr,pr[1:]):
  delta=new['T']-old['T'];n=new['callback']-old['callback']
  if delta>0 and n>0:rates.append(n/delta)
result={'actual_native_logs':len(files),'checkpoint_hashes':hashes,'callback_errors':errors,'prediction_receipts':sum(r.get('event')=='ball_prediction_received' for f,r in rows),'valid_prediction_logged_packets':sum(r['observation13'][9]==1 for f,r in packets),'controller_submission_logs':sum(r.get('event')=='controller_submitted' for f,r in rows),'teams_seen':sorted(set(r['team'] for f,r in packets)),'callback_dt_feature12_median':statistics.median(dt) if dt else None,'estimated_game_time_callback_Hz_median':statistics.median(rates) if rates else None,'logged_mode_counts':{mode:sum(r['mode']==mode for f,r in packets) for mode in ['Neutral','Chase','Jump','Dodge']},'limitations':'Periodic log sampling, countdown/replay clocks can distort rate estimate. Logs are not scores, engine acknowledgement or submission approval. No fabricated prediction data.'};print(json.dumps(result,indent=2))
if hashes!=[a.checkpoint] or errors or not packets or not result['prediction_receipts'] or not result['valid_prediction_logged_packets'] or not result['controller_submission_logs']:raise SystemExit('FAIL: native evidence incomplete or mixed checkpoint/error; do not promote')
if result['teams_seen']!=[0,1]:raise SystemExit('INCOMPLETE: test actual native matches on both sides')
if dt and statistics.median(dt)<.75:print('WARNING: 120Hz-like callback input. Paired RocketSim kickoff probe failed0/10 at120Hz for both geometric and selective candidates. Check native kickoff contacts; do not change the runtime contract or force a rate.')
print('Native log checks passed; inspect actual scores, kickoff performance and video before any approval.')
