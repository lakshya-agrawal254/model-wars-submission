# Final V13 comparison — private review, no submission

Recommendation: test Outcome-128 first on Windows, with Geometric as the native-tested fallback. Outcome has the strongest overall fresh simulator screening profile; its narrow head-to-head edge over Geometric is not statistically convincing. Do not replace main or claim native acceptance. Ground-64 is rejected for competitive selection because long stationary behavior repeated independently.

## New frozen match evidence

72 fresh real RocketSim matches, four evaluation workers, zero training/optimizer updates. Six seed groups paired by team for each 60 Hz head-to-head; four paired groups against the original V13; two paired groups per 120 Hz head-to-head. Seeds 10501011 + 1009*i were declared before play. Both agents receive the same real two-second arena ball forecast and deterministic argmax controls. Weights are byte-identical before/after.

| Configuration | W/L/D/U | Goals for:against | Stationary seconds / observed seconds | Circling windows |
|---|---:|---:|---:|---:|
| ground64_vs_outcome128_60 | 4/7/0/1 | 16:15 | 603.73 / 4002.92 | 12 |
| ground64_vs_geometric_60 | 1/9/0/2 | 10:27 | 8.83 / 4116.80 | 10 |
| outcome128_vs_geometric_60 | 6/4/0/2 | 19:12 | 4.82 / 3994.12 | 21 |
| ground64_vs_original_60 | 6/2/0/0 | 30:11 | 4.50 / 2564.63 | 4 |
| outcome128_vs_original_60 | 6/1/0/1 | 17:9 | 3.05 / 2636.50 | 10 |
| geometric_vs_original_60 | 5/2/0/1 | 12:7 | 287.77 / 2702.72 | 3 |
| ground64_vs_outcome128_120 | 3/1/0/0 | 8:5 | 1.64 / 2399.62 | 6 |
| ground64_vs_geometric_120 | 3/1/0/0 | 4:3 | 1.95 / 2439.93 | 2 |
| outcome128_vs_geometric_120 | 0/0/0/4 | 1:6 | 1.73 / 3600.00 | 1 |

U means unresolved at an overtime/safety cap, not an observed match draw. Scores at a cap are still reported but do not establish a match winner. Stationary means own speed below 100 UU/s and includes match-start waiting. Circling, contact direction, saves and post-contact outcomes are proxies. Different match durations require reading stationary totals with observed time.

Evaluation wall time: 378.42 seconds. Independent small-screen uncertainty is broad; matched sides are correlated. Full per-seed results, IID Wilson ranges and exploratory seed-group bootstrap intervals are included. No candidate was promoted, and no 100-match confirmation was triggered by a convincing movement-safe improvement.

## Previously established movement and forecast evidence

| 60 Hz short-trial model | Contacts | Stationary seconds | Circling windows |
|---|---:|---:|---:|
| Geometric | 26/38 | 28.85 | 6 |
| Outcome-128 | 31/38 | 1.35 | 5 |
| Ground-64 | 38/38 | 1.50 | 1 |

Those 16-second trials did not predict Ground-64's long-match failures. Earlier Ground screening was 4W/3L/1U against Outcome and 4W/2L/2U against Geometric; earlier Outcome's 4/4 preliminary wins were only two seed groups. Keep these records separate from the fresh comparison rather than selecting whichever run looks best.

Ground-64's original 162-second stationary episode was predominantly car-to-car pinning: 99.75% of stationary samples were within 200 UU of the opponent, median distance 148 UU, median speed 45 UU/s, median steering 0.97 and throttle 1. The network selected Chase throughout. Only about 13% of those samples were near a wall. A training-only jump pulse disrupted that one consumed-seed pin, but the trigger occurred on a wall; this was a scripted physical feasibility probe, not a learned recovery, independent match improvement or permitted runtime intervention.

The new Ground-versus-Outcome seed 10506056 produced about 299 stationary seconds on each side while selecting Chase on every callback. These new summary records confirm recurrence, but do not contain collision geometry to prove the exact cause of each new stall. No extra recovery training was attempted after the deadline cutoff.

Geometric and Outcome retain zero forecast encoder columns. Ground learns forecast position and validity columns while retaining zero horizon/first-offset columns. On a real prefix, forecast invalidation changed mean steering by 0.02793, maximum 0.13451, without a mode change. In 22 separate controlled trials, actual forecasts yielded 22 contacts and 10 goals versus 21 contacts and 7 goals with in-memory forecast-column ablation; falls and recovery worsened slightly. This is partial interception evidence, not established competitive/native improvement.

Native Geometric logs contain seven Active-phase sampled Neutral outputs with a visible distant ball; this is distinct from Ground's all-Chase stalls. Native callbacks were approximately 60 Hz. The 135 repeated-clock errors lack the phase at the error timestamp, so paused versus active simulation remains unresolved. Neither 120 Hz failures nor those errors establish the cause of all poor native gameplay.

The actor has no direct opponent or goal-position input. Own speed and ball history may support collision inference, but useful recovery targets must be validated on multiple independent closed-loop trajectories. No runtime manoeuvre or extra input is added. No flip experiment was repeated; actual dodge counts are preserved in the raw fresh records.

## Checkpoints, compliance and preservation

| Candidate | Actor SHA256 | Exact actor / official validator |
|---|---|---|
| geometric | `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc` | 13D / 26181 / original GRU + decoder / PASS |
| outcome128 | `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043` | 13D / 26181 / original GRU + decoder / PASS |
| ground64 | `d79cac01aefe8d161b2ec5f8154fac608f554546a96644f0545b4cb254c6276a` | 13D / 26181 / original GRU + decoder / PASS |

Each already sealed kit passes the untouched official validator from its own directory. Original policy, observation contract, validator and workflow files remain unchanged. No external pretrained model, teacher output, incompatible checkpoint, additional control or runtime motion script is used. This establishes the supplied numerical/architecture checks; event-specific rules beyond the supplied materials are not invented.

Outcome-128 actor: `runs/outcome-flip-learning-v1/candidate.pt`; complete resumable state: `latest.pt` in that directory. Geometric: `model/V13.pt` and `runs/geometric-retained/latest.pt`. Ground-64: `screened-actor.pt` and complete `update-64.pt` under `runs/ground-trajectory-learning-v1/`. All 60 inventoried local checkpoints are unchanged; prior verified private Drive backups retain optimizer/RNG/data/source state. This comparison introduces no new learner checkpoint.

Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`. New evaluation evidence is on a separate private branch. Existing main Actions passed; new branch Actions has not run because API workflow dispatch was Forbidden and the unchanged push trigger only targets main. Do not merge/push main to work around this.

## Windows and submission blockers

Use FINAL_V13_WINDOWS_REVIEW.md for exact local tests and approval-dependent submission steps. Outcome/Ground actual Windows gameplay is UNTESTED. Geometric has qualitative user native evidence, not a formal win rate or complete acceptance report. Original V13 is included only as a benchmark and fails the no-baseline-weights requirement as a submission.

First test Outcome versus Geometric and original V13 on both native teams. Preserve scores, video and raw logs, inspect prediction/callback errors and compare stationary/circling recovery. If Outcome does not improve actual gameplay, retain Geometric as the candidate fallback; keep the currently validated main untouched until approval.

Deadline: 11 October 2026 07:00 IST (01:30 UTC). Stop speculative training by 05:15 IST; zero training was started in this task. Required remaining steps: Windows acceptance evidence, explicit user selection/approval, chosen-kit resealing and integrity checks after documentation, official GitHub Actions validation, and confirmed judge identity/read-access/handoff. User deferred judge identity. No merge, main replacement, organizer sharing/access or tournament submission has occurred. Deadline is not authorization.
