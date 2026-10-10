# Bounded learned-flip campaign — private review, not submitted

Decision: preserve Geometric as preferred playing reference. Curriculum-128 is an OPTIONAL Windows diagnostic, not a promoted learned-flip policy. Phase-memory-144 is rejected and backup-only. Main and all historical checkpoints remain unchanged. No additional training is running.

## Exact models and method

Geometric plain actor SHA256: `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`.
Curriculum-128 plain actor SHA256: `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.
Rejected phase-memory-144 plain actor SHA256: `a00747c52b6c0c6c6767bbd90c76e33daf64419eacd1dad22122c5e27cc821b4`.
Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`.

Exactly 13 observations, original encoder/GRU64/head, 26,181 actor parameters, eight original tensor names, deterministic mode argmax and tanh steering, original decoder and hidden-state contract. No external pretrained assets, outputs or labels; no old RocketSim project models. No PPO run. The original geometric forecast-input weights remain zero via training-only constraints; the runtime observation contract still receives every real feature unchanged.

Legal controls: Neutral has no controls; Chase has throttle 1 and steering; Jump holds jump without throttle; Front Dodge holds jump and pitch -1 without throttle. All modes have boost false. A physical front dodge requires jump press, a Chase/Neutral release, then another jump press with pitch. No runtime sequence controller was installed.

## Previous failure audit and outcome selection

The old eight-update proof began with mean Jump/Dodge logit deficits 7.248/6.711, ending at 7.007/6.515: Chase remained the argmax. Target masking already balanced the event windows reasonably (96 Jump, 48 Dodge, 96 release rows); global row imbalance alone was not the explanation. Initially reference-KL gradient was effectively zero; after eight updates its weighted norm was 0.0396 versus supervised CE 5.646, whereas steering anchoring had norm 2.970 and opposed some CE movement. It was incorrect to assume reference KL alone prevented learning. Full chronological prefixes and actual callback-delta inputs were checked.

New real RocketSim search: 28 scenarios, 252 legal sequences plus matched Geometric controls, all five kickoff positions and both sides, moving balls/shots, initially low-airborne starts, misalignment and no-flip cases. Varied onset, jump hold and release durations. Selection required real flip physics, useful contact/goalward outcomes or faster contact without losing direction, stable landing and acceptable stationary/circling/challenge outcomes. Removing the second press on identical states retained attributable benefits in 21/28 scenarios. Four extra very-close ground cases supplied three no-benefit negatives. These are training-only independently generated examples, NOT evidence that the actor learned flips.

## Actual optimization and failure mechanism

Curriculum: 104 complete sequences / 54,075 collected observation rows at 60/120 Hz; 128 Adam updates, 789,289 supervised row presentations. Class-balanced mode loss on maneuver windows, own-policy driving/steering preservation outside them. Encoder/GRU and original heads all trained. Fit plus development checks took 174.50 seconds. Teacher controllers running at 60 Hz while recording authentic 120 Hz inputs were a TRAINING-ONLY augmentation; evaluated actors update at every actual requested callback. No runtime rate fixing or invented deltas.

Even with greatly reduced teacher-prefix deficits (Jump/Dodge about 0.90/0.85 at update 32), greedy Chase still won. Training-only least-squares probe on existing GRU memories failed its no-flip/generalization gate: held-out no-flip 69.68%, Jump 72.78%, Dodge 61.42%. This suggests phase discrimination and deterministic temporal control were not reliably represented; it does not prove the fixed architecture cannot learn them.

One predeclared conditional refinement used new student-visited prefixes and outcome-selected sequences, plus auxiliary phase supervision on three EXISTING GRU channels, not new actor heads or inputs. It carried compatible Adam state, added only 16 updates and 92,427 presentations, then stopped at the first guard. Total campaign: **144 actual optimizer updates, 881,716 supervised presentations, zero new PPO learner decisions**. Refinement fit plus checks: 18.16 seconds.

Refinement at 60 Hz: 4/18 contacts, 0/10 kickoff contacts, zero dodges; modes Neutral 0, Chase 107, Jump 7,453, Dodge 0; stationary 80.3 seconds. Holding Jump without release cannot execute a second-press dodge. Fixed-prefix hybrid replay showed both shared GRU/encoder changes and mode-head changes shifted Jump above Chase. Phase channels were chosen for low steering-head influence, but their negative Jump-minus-Chase readout meant the nominal ground-phase target could also raise Jump. This is a concrete objective/readout conflict, not a callback error or Neutral collapse. Hybrid replay isolates effects on that prefix, not every native gameplay cause. No further refinement or generic PPO was launched.

## Fresh independent physical checks

18 trials/model/rate, independent seeds/geometry, all five kickoff positions and both sides. Deterministic inference, actual RocketSim forecasts and controls; no demonstration scripts used by evaluated actors.

| Policy/rate | Contacts | Kickoffs | Median kickoff contact | Actual dodges | Stationary seconds | Goals for/against |
|---|---:|---:|---:|---:|---:|---:|
| Geometric 60 | 16/18 | 8/10 | 3.15 s | 0 | 0.783 | 4/0 |
| Curriculum-128 60 | 16/18 | 8/10 | 3.15 s | 0 | 0.667 | 5/0 |
| Geometric 120 | 3/18 | 0/10 | unavailable | 0 | 0.950 | 0/0 |
| Curriculum-128 120 | 3/18 | 0/10 | unavailable | 0 | 0.750 | 0/0 |

No learned flip exists, so useful-flip percentage and learned-flip recovery time are **not measurable**, not a demonstrated success rate. Actual falls/recovery/contacts are retained in JSON. Development 10/10 kickoff results are not substituted for the fresh 8/10 results. The separate 120 Hz simulator weakness remains and does not explain the user's measured 60 Hz native session.


### Fresh contact details (conditional on observed contacts)

| Policy/rate | First-touch advantage, kickoff median | First-contact ball speed, median | First-contact goalward velocity, median | Missed contacts |
|---|---:|---:|---:|---:|
| geometric_60 | 0.067 s | 1800.096 uu/s | 1539.216 uu/s | 2/18 |
| geometric_120 | unavailable s | 1595.443 uu/s | 1371.895 uu/s | 15/18 |
| candidate128_60 | 0.050 s | 1808.263 uu/s | 1556.372 uu/s | 2/18 |
| candidate128_120 | unavailable s | 1507.515 uu/s | 1257.240 uu/s | 15/18 |

Positive first-touch advantage means our actor touched before the opponent; unavailable means no jointly observed touch. These medians do not credit missed trials and are not flip improvements. Contact direction uses the actual team-goal direction in training/evaluation scoring, not an added actor input.

## Fresh complete-match screening

Two new seed groups, each side swapped, four matches/configuration; 24 matches total, 187.94 seconds wall time. Correct actual-forecast wrapper hook verified. Standard RocketSim Soccar physics, 300 active-second regulation, 120-second overtime cap and 900-second total cap. Capped matches stay unresolved even with a leading score. Ball-floor/kickoff clock handling is a simulator proxy, not established native-game clock parity. Both bots use the same conditions. No confirmation seeds reused for tuning.

| Policy versus opponent | Hz | W/L/D/unresolved | Goals for/against | Stationary fraction | Circling windows |
|---|---:|---:|---:|---:|---:|
| Curriculum-128 vs Geometric | 60 | 4/0/0/0 | 14/5 | 0.155% | 6 |
| Curriculum-128 vs original V13 | 60 | 2/2/0/0 | 7/6 | 0.142% | 13 |
| Geometric vs original V13 | 60 | 2/2/0/0 | 5/3 | 7.786% | 2 |
| Curriculum-128 vs Geometric | 120 | 0/0/0/4 | 3/3 | 0.060% | 2 |
| Curriculum-128 vs original V13 | 120 | 3/1/0/0 | 5/2 | 0.101% | 1 |
| Geometric vs original V13 | 120 | 1/3/0/0 | 7/6 | 3.687% | 5 |

**Zero actual dodges in all 24 matches.** The 4/4 win result is a promising incidental movement finding, not a verified learned-flip improvement. Independent-binomial Wilson 95% win interval for 4/4 is 51.0–100%; for 2/4, 15.0–85.0%. Side-paired seeds are correlated, leaving only two seed groups and less evidence than four independent matches. Candidate goal differential vs original is +1, control +2, with circling 13 vs 2. No 100-match promotion confirmation was justified for a candidate that failed the stated flip and circling gates. No promotion.

Stationary = car speed <100 uu/s; fallen = car z<60 and up-vector z<0.5. Circling is the implementation's trajectory-window proxy, not a manually adjudicated category. Goalward/backward contact counts are outcome proxies, not proof of useful shots: candidate vs original 60 Hz 53/44, Geometric 43/34. Logs preserve contacts, phases and outcomes for independent inspection.

## Verification and remaining limits

Complete save/load restores actor, all eight Adam states, RNG, data and source hashes. Finite observations, exact original action controls, deterministic repeat, whole-sequence versus single-step inference and hidden-carry parity pass. Unmodified official validator passes the isolated Curriculum-128 kit; SDK preflight passes without live game. Official actor/observation/validator/workflow files remain byte-identical to official sources. Mechanical validation is not organizer adjudication or native Windows acceptance.

Current experimental GitHub Actions invocation is blocked by Forbidden API access. Existing main's passing result is preserved; no new Actions pass claimed. User can run the unchanged workflow manually on the private experimental branch. Actual Rocket League Windows execution is UNTESTED for this new candidate. No judge access, main push/merge, model replacement or tournament submission occurred.

The user's 135 native duplicate-clock errors remain unclassified pending complete raw JSONL before/during/after T194.4583282470703/frame23335. Do not infer paused/active gameplay or change callback timing. Native logging remains diagnostic, with errors surfaced and original contract unchanged.

Complete private backup includes every checkpoint from this campaign, both training datasets, optimizer/RNG states, source and compact results. Existing prior verified backups stay separate; no cumulative archive nesting or deletion. The Windows kit defaults to Geometric and offers Original V13, A and optional Curriculum-128; the unsafe refinement is not a playing option. Follow OUTCOME_FLIP_WINDOWS_REVIEW.md. Submission still requires explicit user approval.
