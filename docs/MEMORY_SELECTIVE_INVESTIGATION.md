# V13 focused investigation — rejected candidate; native timing risk

Private experimental branch `training/v13-memory-selective-20261011`. Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`, model SHA `bda8a7a773d63526fea8a1449ff3971bba48317a05dd024eafc18827c1100011`. Every previous checkpoint and backup is retained. Official architecture, observation contract, decoder and validator/workflow are unchanged. No learner was active before this work. Latest committed checkout6087b7c was clean; original anchored files and saved diagnostics were inspected first. No completed campaign was restarted.

The user reports validated A was noticeably worse than original V13 in actual RLBot v5. This is genuine qualitative native evidence, not quantified scores/callback acceptance. Passing validation does not establish A as strongest. The geometric actor is the mechanically strongest retained self-trained reference and beats the failed anchored learner, but has not demonstrated superiority over original V13 or native acceptance.

## Critic experiment

40 frozen geometric-policy real RocketSim trajectories4096 decisions each, no actor updates. 24train/16heldout, split by entire side-paired seed groups. 163,840 diagnostic behavior decisions are NOT new PPO learner decisions. Each target uses discounted Monte Carlo reward to4096 plus the same preserved critic boundary bootstrap, and assesses only first2048 states. These are finite return proxies, not exact infinite-horizon targets. No native forecasts were fabricated. The old boundary bootstrap creates a limitation, particularly for comparison to that same critic; refit feature comparisons share identical targets.

The memory feature is h_t **after** consuming x_t, detached and aligned with V(x_t,h_t), r_t and nextstate. Fullsequence GRU features and action densities were compared to actual callbacks (including23 physical goal events); hidden states persist across goals. Next-boundary memory comes from actor(x_next,h_t), no hidden reset. Critic gradients cannot reach actor parameters. Capacity control has9073 parameters vs memory9217; exported actor remains26181.

| Critic | Mean heldout MSE (3 fits) | Mean explained variance |
|---|---:|---:|
| observation | 6.598899 | -0.057593 |
| memory | 6.985946 | -0.127757 |
| capacity | 6.617184 | -0.062705 |

Preserved unrefit critic: MSE 6.045217, EV 0.000339. All critics explain little heldout return variance. Memory loses all3 fitting seeds and has5.87% worse MSE than original-width refit. Paired-seed95% bootstrap difference vs observation: [-0.022130418755114078, 0.7967714101076123]; uncertainty includes0. **Reject memory augmentation.** No benefit is established; no claim of statistically proven harm or universal failure. This sparse-return regression cannot alone diagnose gameplay.

## One distinct bounded policy trial

Actor starts from preserved geometric weights SHA55fae517..., not failed anchored actor. The rejected memory architecture is NOT used in PPO. Original observation-only critic and its compatible prior Adam state are carried over; a fresh actor PPO Adam is created once for this new supervised-to-selective-PPO branch. No existing optimizer is overwritten or reset. Critic gets four warm-fit epochs on fresh on-policy rollouts. Actor optimizer includes every original parameter, no new actor tensors.

Global mode reference KL and all-ball-present CE are removed from this candidate loss. Suitable approach mask requires grounded car, low ball, target within300..2500UU, local-forward positive, angle<.65rad and >1sec since either car touched. CE(.01) favors Chase only there. Independent geometric steering MSE(.1) only there; conditional steering reference KL(.02) only when reference agrees with independent target within.2. Outside mask anchor gradients are exactly0, including recovery/post-contact. Categorical divergence is still monitored, and PPO uses actual combined mode/conditional-tanh steering density and clipping. Mode changes elsewhere are permitted, not taught to be useful by these labels.

The audit found legacy CE nonzero gradient outside approaches (.004439 norm on one prefix); new anchor gradient there0. At exact reference equality KL itself is0, so do not claim it immediately suppresses gradients; once mode probabilities change, the legacy all-state categorical KL opposes deviations everywhere. This is a training candidate, never a runtime scripted override. Reward coefficients remain unchanged.

First batch12×2048=24,576 decisions: E,E,O,F,B0,B1,B2,B3,B4,R,R,G. All required situations and all five kickoff positions occur before the first update, fixing the prior pilot schedule. Real recovery states have tilted airborne cars, not a scripted recovery policy. 60/120Hz rollouts alternate with duration-correct gamma/lambda. Chronological actor GRU continuity, fresh behavior densities and explicit time-limit next-value bootstrap are checked. Four PPO epochs, gamma/lambda.999, actor lr1e-4, critic.002, standard steer std.15. Frozen own pool: original V13, A, geometry; no external teachers/opponents. These anchor/schedule changes are tested together, so match effects cannot be attributed to one independently.

## Actual reward audit

Components are logged per simulation decision: +10/-10 goals; gammaPhi(next)-Phi(old); small .08 signed clipped goalward velocity-change contact term with cooldown; -.003 negligible-contact penalty; -.001×duration sustained stationary cost; explicit goal-reset potential correction. The obsolete sim_contract.reward(+2touch/+20goal) method is not called. Potential telescopes across actual nonterminal resets, rather than awarding free kickoff progress.

Frozen diagnostic totals: {"goal": -30.0, "potential": 7.291088405264578, "productive_velocity": 0.9691784973144532, "nonproductive_contact": -0.015, "stationary": -0.06100000000000005, "goal_reset_potential": 1.4978790521709189}. Contact events: {"goalward": 87, "backward": 72, "meaningless": 5, "touch": 267}. 23goals total, and11contact-to-goal-within5sec associations. Contact classification is signed y-velocity change, not shot accuracy; backward contacts can be useful defensively. Own-last-touch and danger-clear counts are proxies, not causal own-goal/save labels. Repeated meaningless touch terms remain negative and are cooldown-qualified.

Stationary control probe: 10real RocketSim seconds, no runtime/model changes; final8sec mean reward -0.0011158706543611943, zero positive rewards. No demonstrated stationary-reward exploit here. Circling remains a match telemetry proxy, not a proven reward exploit. No coefficient changed from these observations.

## Independent match screening

Same regulation300active seconds, overtime/floor proxy and physical safety caps; RocketSim60Hz with absent forecasts. Four fresh matched seed pairs per opponent. Caps are unresolved; zero completed draws here. Observed goals include capped play. These are not actual Rocket League tests.

| Candidate against | Completed W/L/D | Unresolved | Observed GF/GA |
|---|---|---:|---|
| candidate_vs_baseline | 2/4/0 | 2 | 2/4 |
| candidate_vs_A | 5/0/0 | 3 | 13/7 |
| candidate_vs_geo | 0/7/0 | 1 | 3/11 |
| geo_vs_baseline | 2/6/0 | 0 | 4/10 |

Candidate vs geometric conservative score0/8, descriptive win95% Wilson0–32.4%. Candidate-vs-A five completed wins are not enough to establish a stronger overall bot. Starting geometry itself has2W/6L against original V13 on the matched baseline seeds. **Reject the candidate; do not run final confirmation or extend training.** Reserved831... confirmation seeds remain unused.

Ground12/12 and all-five kickoff10/10 contacts at60Hz are preserved. New policy changed120/81,920 replayed real-state prefix mode decisions (0.1465%); mean absolute steering difference.005459; no difference>.05. CatKL mean.0000726. Recurrent prefixes replay each actor with its own memory. Weight changes are not treated as meaningful gameplay improvement. Greedy evaluation produced no useful Jump/Dodge learning; stochastic training sampled them but successful use is unproven.

## Verified callback-rate sensitivity

Fresh independent841... starts, identical opponents, starts and both sides; inference only, no tuning. Both geometric and selective candidates: ground4/4 at60 and120Hz; kickoff10/10 at60Hz but0/10 at120Hz. All five kickoff positions covered on both sides. Conditional kickoff median60Hz: geometry3.35sec, selective3.3667sec. This controlled comparison changes callback rate and corresponding original dt feature, so it does not isolate dt encoding from recurrent update frequency. It demonstrates a serious rate sensitivity. **Do not force60Hz, alter the runtime observation contract or claim that native failure has been established.** Native logs must identify actual callback frequency and real kickoff outcomes. No native recordings were supplied for the new candidates.

## Preservation, Windows review and approval

Strongest surviving self-trained reference for controlled Windows comparisons remains geometric checkpoint; its global strength is not verified. Original V13 is a required comparison given the user’s native finding. Validated A/main stays unchanged solely as protected submission state; it is not declared strongest based on validation. New candidate SHA `1177a04f30a3fc7ad10a5b51b491700a249d49a6dd08ebf38e61638a57a8faf4`, complete critic/both optimizers/RNG/config/state preserved at runs/memory-selective-24576/latest.pt. Existing failed anchored32,768 state remains untouched. One new bounded trial only. Training 30.95sec (~794.0PPOdecisions/s excluding evaluation/diagnostics/export); actor PPOlineage40,960. Diagnostic163,840 frames and evaluation are separately counted.

Read docs/MEMORY_SELECTIVE_WINDOWS_REVIEW.md for the review ZIP’s three separate kits: selective candidate, geometric control and untouched original V13. Record both-side full native scores, prediction/callback/control evidence, actual dt and all-five kickoff contacts. Code/tests/validator pass cannot replace native acceptance. Deadline11October2026 07:00IST; no further speculative training. User approval remains mandatory before main changes, organizer access or submission, even if deadline passes.
