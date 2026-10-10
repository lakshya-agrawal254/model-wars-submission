"""Read actual local RLBot JSONL logs. Never fabricate match/prediction evidence."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,default=Path('logs'));p.add_argument('--checkpoint');a=p.parse_args();files=sorted(a.logs.glob('*.jsonl'));assert files,'No native logs: launch a real RLBot match first'
rows=[]
for file in files:
 for i,line in enumerate(file.read_text(encoding='utf-8').splitlines(),1):
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:raise SystemExit(f'Invalid JSONL {file}:{i}')
starts=[r for r in rows if r.get('event')=='process_started'];errors=[r for r in rows if r.get('event')=='callback_error'];packets=[r for r in rows if r.get('event')=='processed_packet'];pred=[r for r in rows if r.get('event')=='ball_prediction_received'];controls=[r for r in rows if r.get('event')=='controller_submitted'];valid=[r for r in packets if r.get('observation13',[0]*13)[9]==1];teams=sorted(set(r['team'] for r in packets));result={'files':len(files),'checkpoint_hashes':sorted(set(r.get('checkpoint','') for r in starts)),'callback_errors':errors,'logged_processed_callbacks':len(packets),'native_prediction_receipts':len(pred),'native_valid_prediction_callbacks':len(valid),'controller_submission_logs':len(controls),'teams_seen':teams,'limitation':'Logs provide SDK evidence, not engine acknowledgement, match scores or organizer acceptance.'};print(json.dumps(result,indent=2))
if a.checkpoint and result['checkpoint_hashes']!=[a.checkpoint]:raise SystemExit('Wrong or mixed checkpoint logs: use a clean extraction')
if errors or not packets or not pred or not controls or not valid:raise SystemExit('Native evidence incomplete or callback error: DO NOT promote')
if teams!=[0,1]:raise SystemExit('Play an actual match on each side before acceptance')
print('LOG EVIDENCE CHECK PASSED; manually verify full-match scores, GUI operation and controls before acceptance')
