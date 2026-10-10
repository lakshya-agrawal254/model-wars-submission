# Experimental branch status: geometric V13 — NOT promoted

This branch's root model is an isolated native-test candidate, SHA256 `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`, named Lakshya V13 Geometry Candidate. Main remains c151310b5c36996aeb1fe192be5eea7cc328e846 with the retained A checkpoint. One eight-epoch geometric-supervised fit used 38,400 real RocketSim frames; exact original 13D/GRU/26,181 parameters and four controls. New candidate contact: 24/24 ground, 24/24 wide orientation, 20/20 all-five kickoffs. Match screening: 3W/5L vs untouched V13 (8GF/7GA); 3W/3L/2unresolved vs retained A (12GF/11GA observed, including capped play). Does not pass promotion. No further training or unused confirmation matches. Actual Windows RLBot gameplay is NOT TESTED. See GEOMETRIC_SUPERVISED_EXPERIMENT.md and GEOMETRIC_WINDOWS_TEST.md. Prior campaign results follow unchanged for historical comparison.

# V13 PPO feasibility results — no promotion

Private repository: `lakshya-agrawal254/model-wars-submission`. Base commit `83d92a69c80930652c4d26be8db9f7ebea062e38`, branch `training/v13-ppo-curriculum-20261011`. Official template pinned to `5d8801361766cae4584245e9188edc64fd1d79c3`.

## Selected artifact and legality

`model/V13.pt` is the **best screened feasibility candidate**, scratch-ground-A, trained for **16,384 decisions**, SHA256 **bda8a7a773d63526fea8a1449ff3971bba48317a05dd024eafc18827c1100011**. It is not a verified improvement or promoted champion. The original starter remains unchanged at baseline/v13-original-83d92a6 and in the complete private backup, SHA256 **579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7**. Prior RocketSim champion 3277796 / -0.25 and every old-project checkpoint were not imported, changed or trained.

Exact actor: unchanged official 13D encoder, GRU64, head5, **26,181 parameters**, original native decoder. No Nexto/Necto parameters, teacher outputs/labels, incompatible weights or externally pretrained assets. A scratch initialization of this same organizer architecture was used after starter continuations failed; all updates are PPO from self-generated real RocketSim physics. This is a legal architecture/path candidate, not a guarantee of organizer acceptance or gameplay quality.

## Training actually completed

**98,304 new learner decisions across bounded branches**, not one model's cumulative count. Latest unsuccessful lower-entropy A4 lineage reached **49,152** (C/D/E ended at 32,768); selected A lineage is **16,384**. Measured training time summed across processes: **210.61 seconds**, 466.8 learner decisions/s during training. This excludes evaluation/setup/export and is not end-to-end throughput. No training continues in the background.

| Stage/run | New decisions | Lineage decisions | Training seconds | Last entropy | Last approx KL | Last gradient norm |
|---|---:|---:|---:|---:|---:|---:|
| pilot-A | 8,192 | 8,192 | 15.01 | 0.5075 | 0.006180 | 1.2672 |
| ground-idle-A2 | 8,192 | 16,384 | 15.25 | 0.7616 | 0.004300 | 0.4958 |
| long-ground-A3 | 8,192 | 24,576 | 16.59 | 0.7533 | 0.001653 | 0.3577 |
| scratch-ground-A | 16,384 | 16,384 | 34.99 | 1.3399 | 0.000512 | 0.1899 |
| scratch-kickoff-B | 8,192 | 24,576 | 15.64 | 1.3320 | 0.000934 | 0.1127 |
| curriculum-C | 4,096 | 20,480 | 7.30 | 1.3019 | 0.000993 | 0.1714 |
| curriculum-D | 4,096 | 24,576 | 7.58 | 1.3195 | 0.000080 | 0.0529 |
| curriculum-E | 8,192 | 32,768 | 15.47 | 1.3167 | 0.000090 | 0.0617 |

Stage A is 15-second real-state ground trials, B kickoffs, C low-ball jump opportunities, D incoming-goal defence, E normal starts versus untouched starter and legal own-policy history. C/D contain one-third normal starts. The scratch A candidate contacted the ball in **10/16** matched-side trials (median **9.48s** among contacts) versus **0/16** for the untouched starter. Mean CPU actor inference latency across those trials was **0.1607 ms**. These are trial metrics, not full-match win evidence. PPO gradient/entropy/clip/value diagnostics are preserved; larger gradients and touch rewards never triggered promotion.

The first starter continuations disclosed a zero-value time-limit bootstrap error and failed deterministic ground trials. Fresh scratch training corrected that error. Kickoff continuation regressed; subsequent C/D/E continuation also failed screening. Optional opponent construction consumed RNG after resume; the future resume path now restores RNG after constructing the pool. Used training source snapshots are retained for exact forensic reproduction.

## Match evidence

All candidates used identical real RocketSim Soccar 1v1 conditions against the untouched V13 starter, seeds paired over both teams, greedy official inference. Active regulation: 300 seconds; kickoff time paused until contact. Five-minute expiration uses a documented ball-floor proxy. Sudden death capped at 120 active seconds and total physical simulation capped at 900 seconds. **Caps are unresolved outcomes, never completed draws or wins.** No native Rocket League result is claimed.

| Candidate / set | Completed W | L | Completed D | Unresolved | Observed GF–GA (including capped play) |
|---|---:|---:|---:|---:|---:|
| Scratch A screening (4 pairs) | 4 | 1 | 0 | 3 | 9–5 |
| Kickoff B screening (same pairs) | 0 | 3 | 0 | 5 | 1–4 |
| C/D/E continuation screening (same pairs) | 0 | 0 | 0 | 8 | 0–0 |
| Scratch A unused confirmation (12 pairs) | **4** | **9** | **0** | **11** | **8–13** |

Screen seed **831011**; unused confirmation seed **991011**, subsequent pair seeds +1009. The confirmation set was used once and has not been used for tuning. Raw evaluator counters were 5/9/10, but one raw W ended at the safety cap and is correctly counted as unresolved here.

Before confirmation, the screen eligibility rule was amended only to permit independent evidence gathering with censoring; it did not alter a model, reward or evaluation conditions. Promotion requires the one-sided exact independent-side-pair sign test p<0.05 after counting every unresolved match as a candidate loss. Confirmation: **one positive pair, nine negative pairs, two tied pairs; p=0.9990234375**. The candidate **fails** promotion. Censoring creates an observed win-fraction interval of 4/24 to 15/24 depending on unresolved outcomes, not a confidence interval; pairing/variance are handled by the declared sign test. Completed-only win fraction 4/13 is selected by completion and must not be represented as unbiased win rate.

Recurring limitation: first contacts often take **35–76 seconds**, with some never occurring. Training ground contacts did not translate to reliable kickoff or match wins. Recurrent trajectories and steering remain weak; no proven root cause beyond the observed behavior and substantial prediction-domain gap is established. Do not extend this campaign simply because shaped reward improved.

## Verification and native readiness

Five automated tests passed: exact actor/action contract; finite true-state observations and absent-prediction mask/clock; actual combined behavior likelihood with finite actor gradients; full-sequence versus callback GRU parity; strict loading of both optimizers/RNG and exact selected complete-checkpoint/inference parity. Numerical preflight passes both team configurations and native ControllerState serialization without importing the old project. Trusted official validator and kit integrity checks are run unchanged; their actual outputs are included alongside this report.

**GUI discovery/startup/live car control/actual Rocket League matches: NOT TESTED**, because this Linux host cannot run the Windows game. Training uses prediction-unavailable original columns (zero mask), while the unchanged live runtime consumes actual native RLBot prediction slices. Forecast-valid behavior and actual processed-callback timing transfer remain unverified. No fabricated prediction data or RLBot results were used. Follow `docs/WINDOWS_ACCEPTANCE.md` before submission or model promotion.

## Durable preservation and submission blockers

Complete private backup preserves original baseline, selected A and latest A4 actors, critics, both Adam states, RNG, counters, configuration, history and source snapshots. Windows inference package excludes optimizer/training run files, old archives and virtual environments. Private Drive exports must pass download SHA, cold extraction, strict complete-state loading and inference parity; the export record identifies those results and exact bytes. Rolling retention manages only this new project; original RocketSim backups are untouched.

Remaining organizer information: resource/runtime limits and judge collaborator identity/access (the official README specifies pushing to main). User-confirmed deadline: **11 October 2026, 07:00 IST (01:30 UTC)**. No architecture exception is assumed and external experts are not used. Validator passing establishes mechanical validation, not gameplay competitiveness. The current candidate is offered for local feasibility evaluation; **a genuinely improved tournament playing model has not yet been demonstrated**.

## Deadline-safe follow-up (rejected)

After the fallback was sealed and pushed, one additional bounded continuation tested the hypothesis that excessive mode entropy prevents sampled learning from becoming committed greedy behavior. It resumed the A actor, critic and both Adam states, changed only the entropy coefficient from .025 to .005, and collected **32,768** additional A decisions (lineage **49,152**, campaign total **98,304**). Steps512, rewards, observations, action controls, learning rates and starter opponent were unchanged. The new seed set **451011**, eight side pairs, was unused in earlier screening/confirmation.

Frozen A control: **3 completed wins, 5 losses, 8 unresolved**, observed goals 4–8. Lower-entropy A4: **0 completed wins, 0 losses, 16 unresolved**, goals 0–0, no first contact in any match. It fails screen qualification and receives no confirmation or promotion. A4 critic/optimizers/RNG remain as the latest complete resumable learner; the playing artifact stays A. These new development records were not used to retune either candidate. No direct head-to-head/confirmation was warranted after the failed baseline screen.

The fallback already passed the unmodified official GitHub Actions run **38078146088** on commit **ad710cb34db7173272cfa9b5f0da8b134bcf6e3c**; final-documentation changes are validated separately before publication to main. The original starter is preserved by immutable commit, a dedicated baseline branch and a complete private backup. The selected trained feasibility artifact is published to main solely to leave a mechanically compliant submission before the deadline, **not as a claim that match-based promotion or native acceptance succeeded**.


## Current private review status — 11 October 2026

Main remains unchanged; the user now requires explicit approval before submission or replacement. Geometric remains the preferred Windows playing reference. The private eight-update Jump/release/Dodge feasibility candidate is rejected: no deterministic learned flips, no wins against its Geometric start in screening. See FLIP_FEASIBILITY_REVIEW.md and FORECAST_SCREEN_CORRECTION.md for all results and the correction to earlier forecast-domain metadata. No organizer sharing, main merge or submission is authorized or performed. Native RLBot validation remains incomplete; raw callback logs are still awaited.
