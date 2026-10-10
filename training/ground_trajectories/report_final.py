"""Read-only final comparison report and paired uncertainty; never train/seal."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/final-v13-comparison-v1'
def main():
 result=json.loads((OUT/'result.json').read_text());records=json.loads((OUT/'records.json').read_text())
 before=json.loads((OUT/'preservation-before.json').read_text())
 assert all((ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
 preservation=dict(checkpoints=len(before),all_unchanged=True,main='c151310b5c36996aeb1fe192be5eea7cc328e846',training_updates=0,submitted=False)
 (OUT/'preservation-after.json').write_text(json.dumps(preservation,indent=2))
 pairs={};rng=np.random.default_rng(110102026)
 for key,rows in records.items():
  by={}
  for r in rows:by.setdefault(r['seed'],[]).append(r)
  assert all(len(v)==2 and {r['team'] for r in v}=={0,1} for v in by.values())
  margins=np.array([sum((r['result']=='W')-(r['result']=='L') for r in v if 'cap' not in r['end'])/2 for v in by.values()])
  samples=rng.choice(margins,size=(10000,len(margins)),replace=True).mean(axis=1)
  pairs[key]=dict(seed_groups=len(by),mean_win_minus_loss=margins.mean(),bootstrap_seed_group_95=np.quantile(samples,[.025,.975]).tolist(),positive_pairs=int((margins>0).sum()),negative_pairs=int((margins<0).sum()),tied_pairs=int((margins==0).sum()),note='Exploratory small-sample cluster bootstrap; censored matches contribute neither W nor L. Not a promotion test.')
 (OUT/'paired-uncertainty.json').write_text(json.dumps(pairs,indent=2))
 lines=['# Final V13 comparison — private review, no submission','',
 'Recommendation: test Outcome-128 first on Windows, with Geometric as the native-tested fallback. Outcome has the strongest overall fresh simulator screening profile; its narrow head-to-head edge over Geometric is not statistically convincing. Do not replace main or claim native acceptance. Ground-64 is rejected for competitive selection because long stationary behavior repeated independently.',
 '', '## New frozen match evidence','',
 '72 fresh real RocketSim matches, four evaluation workers, zero training/optimizer updates. Six seed groups paired by team for each 60 Hz head-to-head; four paired groups against the original V13; two paired groups per 120 Hz head-to-head. Seeds 10501011 + 1009*i were declared before play. Both agents receive the same real two-second arena ball forecast and deterministic argmax controls. Weights are byte-identical before/after.',
 '', '| Configuration | W/L/D/U | Goals for:against | Stationary seconds / observed seconds | Circling windows |', '|---|---:|---:|---:|---:|']
 for k,s in result['summary'].items():lines.append(f"| {k} | {s['W']}/{s['L']}/{s['D']}/{s['U']} | {s['GF']}:{s['GA']} | {s['stationary_seconds']:.2f} / {s['observed_seconds']:.2f} | {s['circling_windows']} |")
 lines += ['', 'U means unresolved at an overtime/safety cap, not an observed match draw. Scores at a cap are still reported but do not establish a match winner. Stationary means own speed below 100 UU/s and includes match-start waiting. Circling, contact direction, saves and post-contact outcomes are proxies. Different match durations require reading stationary totals with observed time.', '',
 f"Evaluation wall time: {result['elapsed_seconds']:.2f} seconds. Independent small-screen uncertainty is broad; matched sides are correlated. Full per-seed results, IID Wilson ranges and exploratory seed-group bootstrap intervals are included. No candidate was promoted, and no 100-match confirmation was triggered by a convincing movement-safe improvement.", '',
 '## Previously established movement and forecast evidence', '',
 '| 60 Hz short-trial model | Contacts | Stationary seconds | Circling windows |', '|---|---:|---:|---:|', '| Geometric | 26/38 | 28.85 | 6 |', '| Outcome-128 | 31/38 | 1.35 | 5 |', '| Ground-64 | 38/38 | 1.50 | 1 |', '',
 'Those 16-second trials did not predict Ground-64\'s long-match failures. Earlier Ground screening was 4W/3L/1U against Outcome and 4W/2L/2U against Geometric; earlier Outcome\'s 4/4 preliminary wins were only two seed groups. Keep these records separate from the fresh comparison rather than selecting whichever run looks best.', '',
 'Ground-64\'s original 162-second stationary episode was predominantly car-to-car pinning: 99.75% of stationary samples were within 200 UU of the opponent, median distance 148 UU, median speed 45 UU/s, median steering 0.97 and throttle 1. The network selected Chase throughout. Only about 13% of those samples were near a wall. A training-only jump pulse disrupted that one consumed-seed pin, but the trigger occurred on a wall; this was a scripted physical feasibility probe, not a learned recovery, independent match improvement or permitted runtime intervention.', '',
 'The new Ground-versus-Outcome seed 10506056 produced about 299 stationary seconds on each side while selecting Chase on every callback. These new summary records confirm recurrence, but do not contain collision geometry to prove the exact cause of each new stall. No extra recovery training was attempted after the deadline cutoff.', '',
 'Geometric and Outcome retain zero forecast encoder columns. Ground learns forecast position and validity columns while retaining zero horizon/first-offset columns. On a real prefix, forecast invalidation changed mean steering by 0.02793, maximum 0.13451, without a mode change. In 22 separate controlled trials, actual forecasts yielded 22 contacts and 10 goals versus 21 contacts and 7 goals with in-memory forecast-column ablation; falls and recovery worsened slightly. This is partial interception evidence, not established competitive/native improvement.', '',
 'Native Geometric logs contain seven Active-phase sampled Neutral outputs with a visible distant ball; this is distinct from Ground\'s all-Chase stalls. Native callbacks were approximately 60 Hz. The 135 repeated-clock errors lack the phase at the error timestamp, so paused versus active simulation remains unresolved. Neither 120 Hz failures nor those errors establish the cause of all poor native gameplay.', '',
 'The actor has no direct opponent or goal-position input. Own speed and ball history may support collision inference, but useful recovery targets must be validated on multiple independent closed-loop trajectories. No runtime manoeuvre or extra input is added. No flip experiment was repeated; actual dodge counts are preserved in the raw fresh records.', '',
 '## Checkpoints, compliance and preservation', '',
 '| Candidate | Actor SHA256 | Exact actor / official validator |', '|---|---|---|']
 for n in ['geometric','outcome128','ground64']:lines.append(f"| {n} | `{result['protocol']['checkpoints'][n]}` | 13D / 26181 / original GRU + decoder / PASS |")
 lines += ['', 'Each already sealed kit passes the untouched official validator from its own directory. Original policy, observation contract, validator and workflow files remain unchanged. No external pretrained model, teacher output, incompatible checkpoint, additional control or runtime motion script is used. This establishes the supplied numerical/architecture checks; event-specific rules beyond the supplied materials are not invented.', '',
 'Outcome-128 actor: `runs/outcome-flip-learning-v1/candidate.pt`; complete resumable state: `latest.pt` in that directory. Geometric: `model/V13.pt` and `runs/geometric-retained/latest.pt`. Ground-64: `screened-actor.pt` and complete `update-64.pt` under `runs/ground-trajectory-learning-v1/`. All 60 inventoried local checkpoints are unchanged; prior verified private Drive backups retain optimizer/RNG/data/source state. This comparison introduces no new learner checkpoint.', '',
 'Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`. New evaluation evidence is on a separate private branch. Existing main Actions passed; new branch Actions has not run because API workflow dispatch was Forbidden and the unchanged push trigger only targets main. Do not merge/push main to work around this.', '',
 '## Windows and submission blockers', '',
 'Use FINAL_V13_WINDOWS_REVIEW.md for exact local tests and approval-dependent submission steps. Outcome/Ground actual Windows gameplay is UNTESTED. Geometric has qualitative user native evidence, not a formal win rate or complete acceptance report. Original V13 is included only as a benchmark and fails the no-baseline-weights requirement as a submission.', '',
 'First test Outcome versus Geometric and original V13 on both native teams. Preserve scores, video and raw logs, inspect prediction/callback errors and compare stationary/circling recovery. If Outcome does not improve actual gameplay, retain Geometric as the candidate fallback; keep the currently validated main untouched until approval.', '',
 'Deadline: 11 October 2026 07:00 IST (01:30 UTC). Stop speculative training by 05:15 IST; zero training was started in this task. Required remaining steps: Windows acceptance evidence, explicit user selection/approval, chosen-kit resealing and integrity checks after documentation, official GitHub Actions validation, and confirmed judge identity/read-access/handoff. User deferred judge identity. No merge, main replacement, organizer sharing/access or tournament submission has occurred. Deadline is not authorization.' ]
 (ROOT/'docs/FINAL_V13_COMPARISON.md').write_text('\n'.join(lines)+'\n')
 with (OUT/'summary.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['configuration','matches','W','L','D','U','GF','GA','stationary_seconds','observed_seconds','circling_windows','max_stationary_match_seconds','actual_dodges'])
  writer.writeheader()
  for k,s in result['summary'].items():writer.writerow({'configuration':k,**{n:s[n] for n in writer.fieldnames if n!='configuration'}})
 print(json.dumps(dict(preservation=preservation,paired_uncertainty=pairs),indent=2))
if __name__=='__main__':main()
