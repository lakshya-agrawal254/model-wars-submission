# Post-deadline failure-prefix PPO: rejected; retain Outcome-128

All new experimental weights are post-deadline research. Nothing was submitted, merged, shared with organizers or written to main by this work. Native Rocket League acceptance is untested.

Research branch: `training/v13-failure-prefix-ppo-20261011`.
Protected remote main: `738e875877bec6ce84965a607cc15af2341f9a87`.
Retained Outcome-128 SHA256: `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.

## Audit and rationale

The forecast pilot made four actor updates. Its successor made 36 updates on 196,608 decisions, received 22 goals, but lost the common-opponent comparison 3 wins versus Outcome's 8. It accumulated 275.88 stationary seconds versus 11.35. Insufficient update count alone therefore does not explain failure. Its deterministic actor always chose Chase on recorded states, whereas approximately 5.5% of sampled actions were Jump/Dodge. None of its training trajectories triggered the sustained-stationary penalty despite long pins in greedy evaluation. Forecast ablation changed steering by only 0.000986 on average.

Earlier unconditional Chase anchoring and selective/memory-critic variants failed competitive qualification. No rejected actor initializes this work. These observations motivated collecting genuine student-visited failures with the actual greedy policy before fresh stochastic recovery rollouts, rather than repeating ordinary-start PPO with more updates.

## Training method and invariants

Both stages start independently from unchanged Outcome-128. Four workers, two learner threads, 60 Hz, real RocketSim, actual two-second ball forecasts. Exactly 13 actor inputs, 26,181 parameters, eight original tensors, original GRU and decoder. No external teacher, opponent-position input, goal-coordinate input or runtime controller. Legal opponents are Outcome-128, Ground-64, Geometric and Original V13, on both sides in each batch.

Eight trajectories per batch each supply 8,192 fresh stochastic suffix decisions. Four first drive greedily in normal games to reach a failure or a bounded timeout; the complete chronological prefix remains in GRU replay but receives no PPO loss. Other rows use normal and attack starts in batch one, normal and defence in batch two. Moving-ball/recovery starts were only scheduled for conditional continuation, which did not happen. Goal resets preserve memory and callback time. Nonterminal rollout cuts bootstrap from the next observation. No old rollout is reused. Per-row replay verifies stored memory and exact behavior likelihoods.

Suffix sampling uses categorical temperature 0.10 and tanh-transformed Gaussian latent steering standard deviation 0.02. PPO uses those actual distributions, with steering likelihood applied only to Chase. Gamma and lambda are 0.999; actor learning rate starts at 0.00005, clip 0.15. Mode reference KL coefficient is 0.02. A latent-steering squared anchor of 1 applies only to fast, forward approaches; slow/failure states have no steering target. There are no unconditional Chase labels. This combined intervention is not a factorial causal study.

Rewards are unchanged: +/-10 goals, potential shaping, small signed productive-contact terms, nonproductive-contact penalty and sustained-stationary penalty. True goal/team coordinates are reward/evaluation-only. The compatible training-only observation critic and its Adam state come from the last legal PPO run; none of its actor weights do. Actor Adam starts fresh because the preserved Outcome actor was trained under a different objective. All subsequent actor/critic optimizer states, RNG states, configuration, histories and raw batches are saved.

## Pilot failure and optimizer fix

Pilot: 131,072 PPO decisions + 12,153 deterministic prefix decisions, eight accepted actor updates, 16 critic updates, 175.76 seconds. It reached brief kickoff collisions and circles but no prolonged stall; stationary reward remained zero. All 149 pre-existing checkpoint hashes passed preservation checks.

The smaller steering noise exposed a real optimizer safeguard defect: inherited code checked KL before a step and stopped only after another step. The first next-pass measured KL was 0.18147. Actual stored action probabilities replay correctly; this was excessive update size, not an on-policy density mismatch. This pilot is rejected and fully preserved.

The corrected training-only optimizer measures post-proposal KL against the frozen rollout actor on complete histories: categorical KL plus Chase-probability-weighted latent-Gaussian KL. It accepts only mean <=0.01 and maximum trajectory mean <=0.025. Otherwise both actor and every Adam moment, step and learning rate are restored before retrying at a smaller step. Tests verified exact rollback with both empty and populated Adam. Copied test actors were discarded.

The corrected collector also removes transient kickoff collision triggers: it requires three seconds of low speed, and two roll-ins specifically seek stalls for up to 180 seconds. Two others allow circling. It starts again from Outcome with disjoint fresh seeds, not pilot weights or trajectories.

## Corrected training evidence

- 131,072 fresh PPO decisions + 38,675 deterministic prefix decisions = 169,747 new learner decisions.
- 19 optimizer proposals: three exactly rolled back, 16 accepted actor updates; 24 critic updates.
- 367.24 seconds of training; finished 2026-10-11 01:43:12.564 UTC (07:13:12.564 IST).
- Maximum accepted mean KL 0.009602; maximum accepted trajectory KL 0.010140.
- A genuine near-car pin was reached. It never recovered during its 8,192-decision suffix and generated approximately -8.192 stationary reward. Reaching the failure did not teach an escape.
- On 169,747 recorded observations, the final deterministic mode was Chase every time. The stochastic suffixes contained 131,069 Chase, one Jump and two Dodge selections. No new useful deterministic recovery mode was learned.
- Forecast gradients were finite (0.0353 to 1.288). Zeroing forecast columns changed mean steering by 0.0003756 and changed no greedy modes. Mean steering change versus Outcome was 0.0008026. These are training-history diagnostics, not independent match evidence.
- Training saw 10 own and 19 opposing goals including greedy prefixes; PPO-suffix goal reward was -60. Prefix goal events are not silently counted as PPO training targets.

The strong Chase preference, very rare alternative exploration and small forecast influence remain plausible learning limitations. This does not establish that the GRU architecture is incapable of recovery or prediction use. The official inputs lack direct opponent position and explicit absolute goal identity; no direct opponent tracking or universally reliable own/enemy-goal inference is claimed.

## Fresh full-match screening

Deterministic 60 Hz, both teams, four fresh side-paired seeds per matchup. Each stage runs 40 matches: candidate against three references plus unchanged Outcome against Ground and Geometric. Same RocketSim physics, 300 active seconds, floor-end proxy, capped overtime/safety; capped games are unresolved, never counted as draws. These are simulated full matches, not native Rocket League tests. 120 Hz was not reevaluated in this 60 Hz investigation.

Pilot seeds: `11801011 + 1009*i`, i=0..3. Corrected seeds: `12201011 + 1009*i`, i=0..3. Training seeds and the two screens are disjoint. All outcomes, goals, contacts, circle/stall metrics and censored events remain in raw records.

### Rejected pilot

| Matchup | W/L/D/unresolved | Goals for:against | Stationary seconds | Circling windows |
|---|---:|---:|---:|---:|
| candidate vs outcome | 2/5/0/1 | 9:15 | 14.17 | 14 |
| candidate vs ground | 3/5/0/0 | 9:16 | 242.28 | 5 |
| candidate vs geometric | 3/4/0/1 | 9:11 | 4.53 | 13 |
| outcome vs ground | 4/2/0/2 | 9:10 | 1201.75 | 3 |
| outcome vs geometric | 5/2/0/1 | 18:11 | 3.72 | 12 |

### Corrected candidate

| Matchup | W/L/D/unresolved | Goals for:against | Stationary seconds | Circling windows |
|---|---:|---:|---:|---:|
| candidate vs outcome | 1/7/0/0 | 4:14 | 102.53 | 13 |
| candidate vs ground | 2/6/0/0 | 9:17 | 178.62 | 10 |
| candidate vs geometric | 5/3/0/0 | 30:12 | 42.05 | 3 |
| outcome vs ground | 5/2/0/1 | 11:12 | 373.12 | 7 |
| outcome vs geometric | 6/2/0/0 | 21:5 | 3.73 | 14 |

Across common opponents the corrected candidate won 7/16, versus Outcome's 11/16 (one Outcome game unresolved). Candidate goal difference was +10 versus +15. Stationary fraction improved from 7.31% to 4.34%, and circling rate fell from 0.244 to 0.154 windows/minute. These aggregates hide a regression against Geometric: 42.05 stationary seconds versus Outcome's 3.73. Lower motion-proxy totals do not offset losing 1-7 directly to Outcome or conceding 29 versus 17 goals against common opponents.

The predeclared gate required >=6/8 direct wins, no worse common-opponent wins/goal difference, stationary-fraction increase <=1 percentage point and circling-rate increase <=25%. The corrected candidate fails the first three. No extension, confirmation campaign or promotion occurred. Outcome remains selected. There were 80 full matches total across the two stages.

Uncertainty: only four independent seed pairs per matchup, not eight independent samples. Exploratory seed-pair bootstrap for corrected candidate-versus-Outcome win-minus-loss is [-1, -0.25]; it is not a calibrated small-sample significance guarantee. The match-wise Wilson interval for 1/8 wins is 2.2%-47.1%, optimistic because it ignores pairing. No claim of precise population strength is made. Outcome itself still suffers seed-dependent car pins (up to 170.6 stationary seconds here); it is not declared robust or native-accepted.

Contact direction uses ball-velocity projections, not proof of useful shots/saves. Collision recovery is grounded speed >400, not tactical recovery. Already-moving collisions can report zero seconds; censored unrecovered collisions and long near-car stalls are retained separately. No inflation of recovery success is inferred from those zeros.

## Integrity, preservation and selection

Read-only verification passed strict original tensor loading, exactly 26,181 parameters, actor/critic Adam restoration, full versus actor-only inference, chunked versus complete recurrent replay, recorded behavior probabilities, prefix exclusion and finite observations. All 171 checkpoint files present before the corrected run remained byte-identical. Original policy/observation source, decoder and official validation tools were not changed by this work. Main still points to the protected Outcome commit.

Corrected rejected actor: `runs/failure-prefix-trust-v1/candidate-batch-2.pt`, SHA256 `2e1e4b220ab3fd32770bef977f09fc2633d30f4236a023dfcaa9832211321fd4`.
Pilot rejected actor: `runs/failure-prefix-ppo-v1/candidate-batch-2.pt`, SHA256 `7016fe0945d731101dd155d182cfbc283b8f6a151878349a91dcbf4326651640`.
Selected unchanged actor: `runs/outcome-flip-learning-v1/candidate.pt`, SHA256 `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.

Combined collection was 262,144 PPO decisions plus 50,828 greedy prefix decisions (312,972 total), 24 accepted actor updates, approximately 543.00 training seconds. Held-out match decisions are not counted as training. No learner remains active after the report.

Four raw data parts are separately uploaded and downloaded with SHA256/finite-array verification; see `docs/evidence/failure-prefix-v1/data-backup-receipts.json`. The private research core holds all new checkpoint snapshots, optimizers, code/configuration and evaluation results. No older archives are nested and no previous checkpoint or backup was deleted. Final core checksum/cold-restore receipt is recorded separately. The public tournament submission process was not invoked.

For read-only reproduction after restoring the backup set, use the project's existing Python environment:

```bash
python training/failure_prefix/verify_trust.py
python training/failure_prefix/summarize_trust.py
python training/failure_prefix/analyze_trust.py
```

The preservation check additionally needs earlier checkpoints at their original paths; they are retained in the original repository and existing verified backups. Do not run the training command to reproduce a report or overwrite the experiment directories. Neither failed actor is recommended for continued training without a revised evidence-based design.

The untouched official validator was rerun read-only on the existing sealed Outcome single-bot repository: PASS, original 26,181 parameters, model hash above, starter cosine similarity 0.05932. It fetched and verified the official upstream workflow. This is a local validator result, not a GitHub Actions run or native Windows acceptance. Experimental candidates were not installed or sealed over the selected model. The existing research branch's root runtime pin still points to its preserved Geometric reference; never deploy the multi-model research tree as Outcome or merge this branch wholesale into main. Use the already sealed Outcome single-bot package for Windows.

Final backup verification: [private research core](https://drive.google.com/file/d/1K9YoOse50pnlyP6jGh-HM_gLGAUBKYYc/view), SHA256 `ad5e9b5608ba0953316e657da338b8578b9971ecddacff57b011105b6ac96146`. Downloaded and verified before extraction; 289 member checksums, all 24 numbered complete checkpoints, actor/critic/both optimizers and RNG restoration, and exact actor-only/full inference on every archived trajectory passed. Four separately listed raw data parts also passed. The complete backup set is 103,690,357 bytes, with owner-only permissions on all five Drive files. No prior backup was deleted. Archive code/evidence commit is `8f1e979b16113dd2ccd5114de377f704499c50db`; the subsequent receipt-only commit adds this verification without altering model or training source.
