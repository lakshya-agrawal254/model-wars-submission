"""Private research backup, no publication, submission or prior backup deletion."""
import argparse,json,hashlib,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid(p):return p.is_file() and p.suffix not in ['.pyc','.zip'] and not any(x in p.parts for x in ['.git','__pycache__','.venv','logs'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--kit',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();assert not args.output.exists();items={}
 for sub in ['training','src','runs/forecast-final-24576-v1','runs/forecast-final-screen-v1','runs/forecast-final-screen-v2']:
  for p in (ROOT/sub).rglob('*'):
   if valid(p):items['Project/'+str(p.relative_to(ROOT))]=p
 for sub in ['model/V13.pt','runs/outcome-flip-learning-v1/candidate.pt','runs/outcome-flip-learning-v1/latest.pt','runs/ground-trajectory-learning-v1/screened-actor.pt','runs/retained/baseline/V13.pt','runs/anchored-final-32768/latest.pt','tools/validate_submission.py','tools/kit.py','.github/workflows/validate.yml','requirements.txt','docs/FINAL_FORECAST_PPO_REVIEW.md','docs/FINAL_V13_WINDOWS_REVIEW.md','docs/FINAL_V13_COMPARISON.md']:
  items['Project/'+sub]=ROOT/sub
 for p in args.kit.rglob('*'):
  if valid(p):items['Forecast-PPO-Research-Only/'+str(p.relative_to(args.kit))]=p
 manifest=dict(kind='private-complete-forecast-PPO-research-not-selected',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),main_unchanged='c151310b5c36996aeb1fe192be5eea7cc328e846',new_decisions=24576,actor_updates=4,selected_playing_model='Outcome128 unchanged in existing final-Windows package',research_actor='Project/runs/forecast-final-24576-v1/candidate.pt',research_resume='Project/runs/forecast-final-24576-v1/latest.pt',actor_optimizer_reset='Fresh Adam forPPO; originalOutcome supervised optimizer included unchanged',critic_source='Earlier compatible critic/Adam only; rejected actor never initialization. Source fullstate included solely for reproducibility.',native_acceptance=False,promoted=False,submitted=False,previous_backups_deleted=False,prior_windows_package_drive_id='1nTPJq8eq_L2NNsq-1bG8TCXv6KBs4LPW',prior_complete_backup_drive_ids=['1_KiGVTJOHh6XzSIlT0dag10dT1Lh136C','1PCS_JE9oav8oRvPTiyZp1HTzEXll1I85'],files={n:sha(p) for n,p in sorted(items.items())})
 with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,p in sorted(items.items()):z.write(p,n)
  z.writestr('backup-manifest.json',json.dumps(manifest,indent=2))
 result=dict(path=str(args.output),bytes=args.output.stat().st_size,SHA256=sha(args.output),members=len(items)+1,source_commit=manifest['source_commit']);args.output.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
