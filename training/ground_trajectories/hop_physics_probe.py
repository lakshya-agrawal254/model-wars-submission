"""Counterfactual physical feasibility only; no learned model or exported override."""
from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/ground_trajectories'));import diagnose_contacts as diag
OUT=ROOT/'runs/ground-hop-physics-v1';DURATION=.1
class Hop(diag.Telemetry):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.stuck=0;self.hop_remaining=0;self.hop_applied=0;self.trigger=None;self.ever_jumping=False
 def step(self,ours,theirs,ticks=2):
  c=self.state.cars[self.our].physics;raw=self.engine._cars[self.our].get_state();close=np.linalg.norm(self.state.cars[self.enemy].physics.position-c.position)<200;slow=np.linalg.norm(c.linear_velocity)<100
  self.stuck=self.stuck+1 if close and slow and raw.is_on_ground else 0
  if self.trigger is None and self.n>=120*60 and self.stuck>=120:
   self.trigger=dict(tick=self.state.tick_count,position=c.position.tolist(),car_distance=float(np.linalg.norm(self.state.cars[self.enemy].physics.position-c.position)));self.hop_remaining=round(DURATION*60)
  if self.hop_remaining:
   # Only training-only physicalprobe; originalfourmode Jump, no secondpress/Dodge.
   ours=[0.,0.,0.,0.,0.,True,False,False];self.hop_remaining-=1;self.hop_applied+=1
  r=super().step(ours,theirs,ticks);self.ever_jumping|=bool(self.engine._cars[self.our].get_state().is_jumping);return r

def main():
 global DURATION
 assert not OUT.exists();OUT.mkdir();path=ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt';initial=hashlib.sha256(path.read_bytes()).hexdigest();records=[]
 for DURATION in [.1,.15]:
  diag.screen.evaluation.match_impl.__wrapped__.__globals__['World']=Hop;r=diag.screen.evaluation.match(path,ROOT/'runs/geometric-retained/latest.pt',10303029,1,60);w=diag.CURRENT;records.append(dict(duration=DURATION,trigger=w.trigger,applied_Jump_callbacks=w.hop_applied,physical_is_jumping_observed=w.ever_jumping,actual_dodges=len(w.actual_flips),match=r,actor_mode_counts=r['counts'],applied_input_override='TRAINING-ONLY shortJump then originaldeterministicactor; not a learnedcandidate or runtimefix'))
 baseline=json.loads((ROOT/'runs/ground-stall-contact-diagnostic-v1/report.json').read_text())['match'];report=dict(baseline=baseline,counterfactuals=records,candidate_sha_unchanged=initial,scope='One consumedseed; legitimatephysics feasibility, not independent evaluation/promotion or neural learning. Zero optimizerupdates, no FrontDodge sequence, no newcontrols.')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==initial;(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
