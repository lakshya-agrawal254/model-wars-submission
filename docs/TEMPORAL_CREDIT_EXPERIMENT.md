# Bounded V13 temporal-credit experiment — no promotion

Private NEW repository `lakshya-agrawal254/model-wars-submission`; branch `training/v13-temporal-credit-20261011`; starts from main `c151310b5c36996aeb1fe192be5eea7cc328e846`. Main, original baseline and old RocketSim expert-backed project remain unchanged.

## Protocol and actual compute

Five independently restored continuations from the same scratch-A complete checkpoint (16,384 lineage decisions), each 16,384 new decisions: **81,920 new decisions**, not a repeat of the old 65,536-decision campaign. Each fork ends at 32,768 lineage decisions. Exact official actor, 13D observation, GRU64, 26,181 parameters and original decoder. No external assets, pretrained outputs or teacher labels. All inherit actor, critic, both Adam states and RNG; no optimizer resets. Fresh on-policy rollouts; verified joint mode/conditional tanh-steer probabilities. Same rewards, learning rates, entropy .025, ground starts and untouched-starter opponent.

| Cell | Trajectory | Gamma / lambda | Forecast penalty | Training seconds |
|---|---:|---|---:|---:|
| short_original | 256 (4.27s) | 0.99 / 0.95 | 0.0 | 30.29 |
| long_original | 2048 (34.13s) | 0.99 / 0.95 | 0.0 | 27.97 |
| short_credit | 256 (4.27s) | 0.9995 / 0.9995 | 0.0 | 31.18 |
| long_credit | 2048 (34.13s) | 0.9995 / 0.9995 | 0.0 | 30.79 |
| long_credit_regularized | 2048 (34.13s) | 0.9995 / 0.9995 | 0.01 | 28.42 |

Total measured training 148.64s (551.1 decisions/s), excluding evaluation and preservation overhead. All cells collect 4,096 decisions per update and take four sample-weighted gradient-accumulation epochs: 16 Adam steps each. This common cadence differs from the old trainer and isolates budget/optimizer-step counts; it is not an exact replay of the old campaign. Short trajectories produce more independent starts, a disclosed consequence of this design.

The 256-step setting was the old default; the retained A was trained with 512, and E used 4096. This experiment tests 256 versus 2048 (34.13s). GRU sequences remain chronological within each trial, continue through goals, and restart only for genuinely separate trials. Time-limited final critic values bootstrap GAE. Unit checks cover delayed rewards and nonzero bootstrap. Gamma .99 gives an approximately 1.66s discount e-folding time, with gamma×lambda credit about .27s. The .9995/.9995 treatment gives approximately 33.33s and 16.66s respectively.

## Fresh matched results

All policies use identical seed sets, both sides and untouched baseline opponents. Real RocketSim, not a native RLBot match. All five standard kickoff positions receive four trials each. Ground cap 30s; kickoff 90s. First-touch medians are conditional on contact and must not conceal failures. Full matches use 300 active seconds with overtime and a documented ball-floor proxy; 900 physical seconds safety cap. Caps are **unresolved**, never credited as wins or draws. No native predictions are fabricated.

| Policy | Ground /16 | Touch median if contacted | Kickoff /20 | Completed W/L/D | Unresolved /8 | Observed GF/GA |
|---|---:|---:|---:|---|---:|---|
| frozen_A | 4 | 9.57s | 18 | 2/3/0 | 3 | 5/13 |
| long_credit | 0 | — | 0 | 0/0/0 | 8 | 0/0 |
| long_credit_regularized | 0 | — | 0 | 0/0/0 | 8 | 0/0 |
| long_original | 0 | — | 12 | 2/0/0 | 6 | 2/0 |
| short_credit | 0 | — | 0 | 0/0/0 | 8 | 0/0 |
| short_original | 0 | — | 12 | 1/0/0 | 7 | 1/0 |
| untouched_baseline | 0 | — | 0 | 0/0/0 | 8 | 0/0 |

Machine-readable summaries include Wilson 95% intervals, kickoff outcomes by position, latency, censored touch-time estimates and actual flip-start counts. With 16 ground trials, 0 contacts has Wilson upper bound 19.4%; frozen A 4/16 has interval 10.2–49.5%. These small samples are diagnostic, not a superiority claim. Team pairs are correlated; the per-trial Wilson intervals are descriptive and not promotion tests. Zero observed flips in these greedy trials cannot demonstrate an improved ability to choose useful flips.

No candidate improves matched ground contacts or passes the predeclared conservative complete-match threshold. **No unused confirmation seeds consumed; no candidate promoted.** Retained A remains the playing model, unchanged SHA256 `bda8a7a773d63526fea8a1449ff3971bba48317a05dd024eafc18827c1100011`.

## Findings and limits

Longer trajectories and longer credit did not help at this bounded budget. Long-credit final critic loss rose to about 182 versus .15 in long-original; inherited value calibration and larger target scale are plausible issues, not proof that long horizons cannot work. Greedy continuations mostly choose Neutral on ground starts. The original A top-two logit median margin is .0584 and mean top-mode probability .3435 on 9,216 real states: mode decisions can be brittle despite small PPO distribution changes. This supports investigating policy/return calibration in future, not launching another campaign before the deadline.

Native recordings were not available in this new project. Counterfactual six-field stress on fixed real-state sequences changed frozen-A modes 12.08%, with joint behavior KL .01207; this is sensitivity only, not native forecast validity or match performance. Regularization acts only on the existing six input columns and changes no runtime contract. It reduced counterfactual joint KL from .009157 to .008293 versus its unregularized matched cell (~9.4%) but still changed 16.83% of greedy modes versus 17.77%. These columns are zero in simulation, so those two cells have exact identical nonforecast weights and identical zero-forecast inference; sensitivity reduction does not imply gameplay improvement.

## Validation, preservation and native acceptance

Nine automated contract/temporal/kickoff/regularization tests pass. All actors strict-load with 26,181 parameters; complete retained experimental learner includes finite critic, both Adam states, RNG, config/history and exact source hashes. Bounded experiment preservation retains one full resumable learner (`long_credit_regularized`) and four inference-only alternatives; all compact results remain. This is not a champion/rollback change. Existing main backups and old completed-campaign checkpoints are not overwritten. Recovered `selected_original_ppo.py` matches the original A checkpoint source hash `4256e3dd03b6ca5a8ecdd342709010d4bd7d8ce2f522460792856459c8c8044e` exactly; no checkpoint metadata was rewritten.

Actual Rocket League/Windows acceptance is unavailable on this Linux host. Follow existing `docs/WINDOWS_ACCEPTANCE.md` with the current main package; no experimental candidate should replace the submission without independent match evidence and actual Windows acceptance. Organizer read access remains deferred at the user’s request. Stop training here; main is already validated and pushed well before the deadline.

Reproduce only if intentionally requested: `python training/experiments/temporal_credit/train_all.py --help`; use a new output directory and follow the predeclared protocol. Do not rerun speculative training automatically.
