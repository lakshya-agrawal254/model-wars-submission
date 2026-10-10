# One bounded corrective V13 investigation — rejected candidate, preserved reference

Started from preserved geometric actor, not rejected PPO actors. Main c151310b5c36996aeb1fe192be5eea7cc328e846 unchanged. No external weights/labels, PPO, runtime override, promotion, merge, judge access or submission. The user reports genuine qualitative native improvement of geometric over original; there are no formal native scores/logs. Original and previous checkpoints were not changed.

## Identified versus unresolved causes

The old target is clip(2*atan2(right,forward),-1,1), Chase for every present ball. It ignores speed, turning radius, bearing rate and postcontact context. This is a concrete label limitation; it is not proof of the native stopping cause or every conceded goal. The official decoder cannot steer in Neutral, reverse, brake explicitly or boost. Selecting Neutral is coasting, not controlled braking.

Timing separation on five kickoff starts, both sides, fresh956... seeds, 20sec trials: unchanged60Hz10/10 contacts; same60Hzphysics with delta field forced0.5 ONLY0/10; doubleGRU updates at60Hz with delta unchanged10/10; unchanged120Hz0/10; same120Hz with delta forced1 ONLY10/10. These are diagnostic counterfactuals, never native runtime changes. Extra GRU updates and physics/control frequency alone do not reproduce the contact loss under the tested normal delta. This supports delta-feature sensitivity as a simulation kickoff bottleneck. It does NOT prove native callback errors or justify falsifying native observations. On identical fixed real-state prefixes, delta1 versus0.5 changes mean steering by0.119815 and mode on only0.0246212% of states (mean per-sequence rates): steering is the dominant measured difference. Extra updates change later contacts, so this does not establish their universal irrelevance.

Frozen geometric collection:39600 own deterministic visited decisions at60/120Hz, allfivekickoffs plus ground,movingball,normal,attack,defence,recovery. No absent-ball transitions or Neutral selections occurred in these initial10sec sequences; counterfactual reset-hidden mode selections also remained unchanged. This cannot explain later native stops. Longer20sec mechanical screens do show present-ball Neutral selections; the new model increased these351 to1140. No measured postcontact Neutral was observed in the diagnostic window; native callback/prediction failures remain unknown without actual logs. Simulation inference raised no errors during completed trials. Runtime h retains continuity across goal resets as original; no native reset rule was changed.

## Corrective experiment and early stop

One DAgger-style aggregation iteration: geometric drives the simulator; independent labels use only legal13D current ball bearing/distance/speed and chronological bearing history. Targets extrapolate bearing by0.15sec, use speed-dependent steering gain, and selective coasting for fast/rearward/close ball approximated with radius180+0.00022*speed². Reference modes preserved outside a groundlike relative-height mask; no unconditional Chase across aerial/recovery/near-contact states.3560 coasting labels. Approximation limitations: relative height is not a ground-contact sensor; bearing derivative mixes car rotation and ball motion; coasting cannot steer; these labels do not supply goal-relative tactical information. No self-imitation was used and no offpolicy data were represented as PPO.

44 full real prefixes,39600 simulator decisions, zero newPPO decisions. Compatible original geometric Adam moments carried over; learning rate0.00015,3epochs/33updates maximum. Complete GRU prefixes from real trial starts; no timestep shuffle or hidden resets at goals. Six native forecast input weights kept exactlyzero as original geometric training constraint, without modifying runtime forecasts. Label direction, overshoot/missing-ball cases, callback/fullsequence alignment, finite gradients, optimizer loading, strict state restoration, action mapping and serialized inference verified. After epoch1,11actualoptimizer updates, training stopped because stationary seconds increased beyond the declared10sec tolerance. There was no retry or budget increase.

Independent20sec trials (28total,560simseconds): contacts16/28→28/28; stationary6.1333→18.4083sec; circling5secwindows2→0; goalward contact ticks29→51; goals2:0→4:1. Conditional first-touch median3.15→3.3667sec; rates differ due additional successful trials. Candidate kickoff contacts10/10 at each rate. These are diagnostics, not complete-match wins. Contact direction uses observed signed y-velocity changes; not causal usefulness. The fitted actor changes steering on its own fixed prefixes(meanabs0.040846), not mode argmax there. Increased Neutral on later changed trajectories is a warning, not evidence of useful learned coasting.

## Fresh complete-match screen

Same9543011 initial seed, paired sides for each pairing/rate. Bothagents have the same callback rate and physics. Original regulation protocol300active seconds, ball-floor proxy,120sec capped overtime,900sec total safety cap; goals reset kickoff, hidden continuous. Capped games unresolved even when the score leads. No capped lead is a win. Domain: real RocketSim3D, no native ball predictions or actual RLBot game. Untouched original responds poorly without native predictions in these tests, making native transfer particularly uncertain.

| Pairing | W/L/D/unresolved | GF:GA | stationarysec | circlingwindows | goalward/backwardcontacts |
|---|---|---|---|---|---|
| candidate_vs_original_60 | 0/2/0/0 | 1:3 | 312.02 | 2 | 10/18 |
| candidate_vs_A_60 | 1/1/0/0 | 2:2 | 333.97 | 2 | 13/20 |
| candidate_vs_geometric_60 | 1/0/0/1 | 3:2 | 90.57 | 3 | 25/15 |
| geometric_vs_original_60 | 1/0/0/1 | 1:0 | 436.57 | 4 | 20/18 |
| geometric_vs_A_60 | 0/2/0/0 | 0:2 | 258.00 | 1 | 18/16 |
| candidate_vs_original_120 | 1/0/0/1 | 2:1 | 120.64 | 2 | 32/23 |
| candidate_vs_A_120 | 2/0/0/0 | 3:1 | 170.59 | 0 | 28/14 |
| candidate_vs_geometric_120 | 2/0/0/0 | 2:0 | 63.18 | 4 | 26/24 |
| geometric_vs_original_120 | 0/0/0/2 | 0:0 | 0.15 | 2 | 0/0 |
| geometric_vs_A_120 | 0/0/0/2 | 5:1 | 132.04 | 6 | 9/6 |

Only one paired seed per comparison: no credible win-rate inference, no promotion. Even2/2 wins has a Wilson95% interval approximately34.2%–100%, ignoring dependence within the seed pair; effective independent seed count is onlyone. Candidate lost both60Hz games to original.120Hz improvements are preliminary and mechanical stopping already failed. No independent confirmation seeds were consumed or repeatedly tuned. Geometric remains the preserved self-trained reference with modest qualitative Windows support; neither candidate is proven competitively stronger.

## Sampled versus deterministic

Screen and exported runtime always use argmax mode+mean steering. Separate fresh955... sampled diagnostics use categorical mode and tanhNormalstd0.15 steering; NOT the tournament behavior. Sampled20sec kickoff contacts: geometric60Hz10/10,120Hz5/10; candidate60Hz8/10,120Hz10/10. Deterministic development kickoffs: geometric10/10 and0/10; candidate10/10 and10/10 respectively. Sampled tests use a separate955... seed set, so this is not a matched stochastic/deterministic full-match comparison. Full sampled results are recorded in sampled-summary.json. No sampled metric is used for selection.

## Compliance, preservation and Windows

Official26181parameter13Dencoder/64GRU/5head unchanged; fourmode decoder unchanged, no new trainable modules, forecasts fabricated nowhere. Official validator run unmodified on isolated review kit:PASS; native preflight blue/orange, prediction error checks, action conversion:PASS. All previous sources/checkpoints/main retained. Experimental branch only. Logging forward hook tested bit-identical outputs/hidden state; no added inference. Review kits contain denser mode/stationary/periodic native telemetry; log cap16MiB, optional allcallback logging for short reproduction. Logging I/O overhead is a limitation.

Windows must establish actual modes/logits, callbacks delta/frame continuity, ball/prediction validity, errors, postcontact paths and formal complete-match scores. New candidate native execution UNTESTED. Review candidate is mechanically rejected, not a replacement. See DAGGER_WINDOWS_REVIEW.md for exact commands and pairings. Deadline11October07:00IST remains; readiness does not authorize submission. No active training continues after this completed bounded campaign.
