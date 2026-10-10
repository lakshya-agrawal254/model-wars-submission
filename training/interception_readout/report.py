"""Compact factual report; failed supervised gate prevents any RL extension."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'runs/interception-readout-v1';screen=ROOT/'runs/interception-readout-screen';audit=ROOT/'runs/interception-readout-audit'
s=json.loads((out/'summary.json').read_text());m=json.loads((screen/'summary.json').read_text());evidence=ROOT/'docs/evidence/interception-readout';evidence.mkdir(parents=True,exist_ok=True)
for folder in [out,screen,audit]:
 for p in folder.glob('*.json'):
  if 'physical' in p.name or p.name.startswith('failed-side'):continue
  shutil.copy2(p,evidence/(folder.name+'-'+p.name))
for name in ['interception-validator.log','interception-preflight.log','interception-tests.log']:
 shutil.copy2(ROOT.parent/name,evidence/name)
rows=[]
for key,v in m.items():
 if isinstance(v,dict):rows.append(f"|{key}|{v['W']}/{v['L']}/{v['D']}/{v['unresolved']}|{v['GF']}:{v['GA']}|{v['stationary_seconds']:.2f}|{v['circling_windows']}|{v['productive_contacts']}/{v['backward_contacts']}|")
report='''# Final targeted campaign: no replacement qualified

Selected playing reference remains preserved Geometric V13, SHA25655fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc, full state runs/geometric-retained/latest.pt. Main c151310b5c36996aeb1fe192be5eea7cc328e846 unchanged. Rejected new candidate SHA25672f8338c383f54137181328667b8752602901eda6652a260664111fbd8ab767f. No model was promoted, merged, submitted or shared with organizers.

## Why the prior corrective fit stopped more

Audited its actual saved data:3560/39600 targets were Neutral, all at speeds1200–1410. In the exact moving-ball failed development case on both sides, the actor selects Neutral at10.5sec, becomes stationary at11.9sec and remains stopped for8.1sec, versus0sec in geometric's own trajectory. At first Neutral the car is not grounded; it later settles onto ground with ball roughly4170units away and rearward.

Critically, geometric itself predicts Neutral on566/570 of the failed policy's Neutral states when replaying the same real observation prefix. Restoring only the original mode head leaves570Neutral states; restoring the original encoder/GRU leaves566. This supports changed steering exposing an existing far/rearward-state blind spot, rather than a simple mode-head collapse alone. The ten-second student data ended before the demonstrated stop and contained no own Neutral states. It did contain high-speed coasting labels, which were not demonstrated safe and were removed in this campaign. These ablations are diagnosis only, not new playing models. This is a simulator finding, not a classification of the user's native callback error burst.

## What changed in the one new campaign

Started exactly from geometric, never rejected policies. Longer student-visited trajectories:44x20sec varied ground/moving-ball/kickoff/normal/attack/defence/recovery plus12x40sec normal-match trajectories. Both teams,60/120Hz and allfivekickoffs; original, A and geometric opponents.122400real collected decisions. No externally pretrained weights or labels. Independent geometric targets use pure-pursuit curvature toward a linearly extrapolated moving ball. Turning response calibrated from real RocketSim, four speeds/five steer levels. No simplified physics, actor-input additions or fake native prediction recordings.

Only existing325output-head parameters trained; encoder and64GRU frozen. Target steering is conditional on grounded approach feasibility. Selective Chase labels only for demonstrated low-speed, far-ball, grounded reference-Neutral stalls (1604data states); own reference mode KL outside that mask. No unconditional Chase labels, no Neutral coasting labels, no runtime heuristic. Compatible original supervisedAdam head moments carried over, frozen parameter states preserved in the starting fullcheckpoint. Complete recurrent prefixes determine each cached hidden state; frozen representation makes subsequent cached training exact. Whole seed groups held out across both rates/sides, not shuffled timesteps. Maximum32updates declared before training, checks after16.

The fit made16actualAdamupdates, zeroPPOdecisions, elapsed51.18sec including generation/development diagnostics. Held-out geometric target MSE0.339321→0.267788, about21.1%lower. Actual fixed-prefix steer change mean0.160307 per-sequence;318mode argmax changes across122400states. This is behavior change, not proof of improvement. Independent movement screen failed:contacts18/32→4/32, stationary16.1667→2.1333sec, circling3→10windows, goalward contacts42→3, backward24→4, goals4:0→1:1. New actor selectsChase on all86400development callbacks; fewer stops came with much worse paths. Kickoffs:geometric60Hz10/10,120Hz0/10; candidate0/10atboth rates. Conditional firstcontact median3.0833→3.1083sec, measured on different successful subsets. Postcontact observations dwindled as contacts failed: reduced postcontact stationary samples are not successful recovery. No useful Jump/Dodge behavior was learned.

The stage stopped immediately at its movement gate. No second16updates, PPO phase, hyperparameter retry or independent confirmation followed. Improving open-loop label error with a frozen representation did not improve closed-loop interception. This result does not prove every geometric target or the entire legalGRUarchitecture is incapable of improvement.

## Independent full-match screen

40games, two fresh972... seeds per pairing, both sides at60/120Hz. Frozen candidates, no tuning from these results. Standard realRocketSim3DSoccar; declared legacy simulation match clock:300active-sec regulation, floor-contact proxy,120sec capped overtime,900sec total safety cap. Clock waits during simulator kickoff until ball moves/touched; untapped kickoffs can therefore hit safety caps. All capped games unresolved even when a model leads. These timing limitations prevent claiming exact native RocketLeague rule parity.

This screen uses genuine RocketSim two-second ball forecasts in the original13D schema for BOTHagents. Forecastendpoint matched actual240physics-tick ball roll-forward with0error in a controlled test; prediction does not mutate current arena. These are simulation forecasts, not native RLBot recordings. Actor encoder forecast columns remain zero as original geometric; original/A may respond to the supplied forecast. Older zero-forecast scores are not directly comparable. Persistent unresolved original-bot games prevent claims of reliably beating original in this domain.

|Pairing|W/L/D/unresolved|Goalsfor:against|Stationarysec|Circlingwindows|Goalward/backwardcontacts|
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

Candidate vsgeometric overall3W5L, goals8:10. At60Hz0W4L,3:7; at120Hz3W1L,5:3. It fails the native-relevant60Hz screen and movement gate. Fourmatches/two pairedseeds provide broad uncertainty and no credible general improvement. The120Hz result is not a reason to promote despite these failures. No final confirmation seeds consumed. Match goalward/backward contacts and circling are proxies, not causal shotquality/saves; full records include postcontact goal associations, not proof of effective recovery.

## Compliance and readiness

Original13observations,26181parameters, eighttensor names, encoder/GRU dimensions, fourmode decoder, callback clock and hidden behavior unchanged. Bit-exact frozen encoder/GRU and same-input hidden state verified; fullsequence/callback inference, serialized actor restoration, two compatible headAdam states, allRNG/config/source/data hashes, forecast invariance and exact native controls passed. Official validator run unmodified on isolated candidate kit:PASS; native preflight:PASS, NOT actualgame acceptance. Tests cover realphysics steering sign, independent targets, two-second genuineforecast consistency and recurrent/architecture invariants.

Existing GitHubActions confirmed successful for unchangedmainc151310b5c36996aeb1fe192be5eea7cc328e846 (run38078636751) and geometricc2a8bc9478e454423a398cdd8fbba649e9a7769c (run38081508569). No newcandidate GitHubActions result is claimed. New experiment lives on private training/v13-interception-readout-20261011. Main was not pushed or replaced.

Native qualitative evidence remains that geometric is slightly better than original, but weak. Actual native session reported60Hz,89validforecastsamples,82Chase/12Neutral samples and135duplicateclock exceptions at oneT/frame. RawWindowsZIP stillawaited; paused-vs-active cause unresolved, no error suppression or invented delta. No cloudscore is actualRocketLeague evidence. Windows package provides Geometric-preferred, Original-V13, Validated-A and optionalRejected-readout-diagnostic, with phase/clock/action logging. See INTERCEPTION_WINDOWS_REVIEW.md. Actual newkitWindows execution UNTESTED. Geometric remains preferred reference, not a newly proven stronger submission. Allprevious checkpoints/backups preserved.

Speculative training already stopped before05:15IST. Deadline11October2026,07:00IST. No background training, automaticsubmission, judgeaccess or mainreplacement. Userapproval mandatory; native callback acceptance and formal Windows match scores remain blockers. The campaign produced a tested negative result, not a stronger candidate.
'''
(ROOT/'docs/INTERCEPTION_READOUT_INVESTIGATION.md').write_text(report)
