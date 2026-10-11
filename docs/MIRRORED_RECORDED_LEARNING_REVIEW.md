# Mirrored and recorded-match learning — bounded review, no submission

The existing final candidate package remains frozen and verified privately at https://drive.google.com/file/d/1nTPJq8eq_L2NNsq-1bG8TCXv6KBs4LPW/view . Its SHA256 is `a740aa0923d74a7f779cb17735eaa8c4a8994226f99bde31653e3150f1ed2f1c`. Outcome-128 remains the recommended existing Windows candidate and Geometric the native-tested fallback. Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`. The new user instruction “its ok continu the training” authorized one short mirrored experiment past the previous training cutoff; it did not authorize submission, main replacement, judge access or unlimited training.

## Legal reflection and recurrence

For left/right reflection across global x, retain the team and goal ends. Negate only original 13D columns **1 and 4**, the current/predicted ball's local right components. Negate steering labels; retain Neutral/Chase/Jump/Front Dodge modes, pitch, throttle, jump, masks, distance, speed and timing fields. There is no new action or input. This is not global y/team inversion.

World position and linear velocity transform with `M=diag(-1,1,1)`. Angular velocity is axial and transforms with `-M`. Proper car orientation reflects forward/up and negates the reflected right axis; native Euler representation is pitch unchanged, yaw `pi-yaw`, roll negated. Kickoff indices map `0↔1`, `2↔3`, `4↔4` on both teams.

The unmodified native observation builder passed 200 random 3D orientations with all three availability cases, giving zero float32 feature error. Four actual RocketSim paired kickoff/curve cases on both sides, 60 chronological callbacks each, matched reflected observations within `4.30e-6` and car positions within `0.0021 UU`. All four decoder modes pass controller reflection. These are mapping/physics checks, not trained gameplay improvement.

Mirror an entire chronological sequence and its prefix; never shuffle timesteps or split an original and its reflected sibling across training/validation. Start GRU memory at None only at the original match boundary and recompute it from the reflected prefix. An arbitrary sign flip or reuse of the original 64D hidden state is invalid. Whole-sequence and carried-chunk inference agree on the audited prefixes. Goal resets do not reset actor memory. Native replay/countdown transitions are not fully reproduced by simulator recordings.

## What can actually be reused

- Geometric dataset: 88 sequences / 42,240 rows, consisting of 80 real geometric-controller demonstrations and eight counterfactual missing-ball augmentations. They are not autonomous Geometric model recordings. Do not reuse unconditional Chase/pure-pursuit targets as a new improvement method.
- Outcome dataset: 104 sequences / 54,075 rows: 62 Geometric-driven sequences and 42 selected training-only flip-schedule demonstrations, with both callback rates represented. They are not all Outcome-128 autonomous trajectories. Do not repeat failed flip imitation.
- Ground dataset: 55 sequences / 44,501 rows: 34 autonomous Outcome-128 sequences and 21 independent searched-controller demonstrations. They are not Ground-64 autonomous recordings. Ground-64's 38/38 short contacts did not predict its long pinning failures.
- Prior full-match JSON files contain outcomes and aggregate counters, not complete chronological observations/controls. Downsampled collision traces and native JSONL samples likewise cannot reconstruct full GRU context. Do not invent labels or forecasts from them.

Recorded-match learning should retain original timestamps, predictions, both actor identities, actual controls, match boundaries, goal events and chronological prefixes. Selecting a winning match does not make its circling, stationary intervals or backward touches useful labels. Own-policy imitation is off-policy supervised learning, not PPO; any later PPO would need new on-policy rollouts and actual behavior-policy probabilities.

Mirrored starting-state self-play can generate diverse own-policy data without a mirrored runtime wrapper. It remains a data-collection proposal: no mirrored self-play PPO campaign was launched. Identical-policy self-play alone cannot establish strength; use fixed legal historical rivals and independent opponent matches for evaluation.

## One approved experiment, stopped at four updates

Initialization: preserved Outcome-128, SHA256 `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`. Only 15 successful autonomous Outcome source sequences qualified: contact, no circling, at most 0.5 seconds stationary, no conceded goal, limited falls and a productive goalward contact. Source rows: **11,239**. Independently searched controller/flip labels and external pretrained outputs were excluded.

Both original and reflected full sequences trained the unchanged encoder/GRU/heads using mirrored Chase steering self-imitation, an original-side steering anchor and soft preservation of actual reference mode distributions. No unconditional Chase labels, actor redesign, runtime steering heuristic, reward change or PPO was used. Existing six zero forecast encoder columns stayed zero; this isolated mirroring from new forecast learning. Fresh supervised Adam preserved the original optimizer states separately.

Maximum eight updates were authorized. The fit stopped at the first check, **four real Adam updates / 22,688 supervised row presentations / zero new PPO decisions**, after its declared contact guard failed. Collection/fit/development wall time was 18.95 seconds. There were zero source mode-label mismatches, and source steering replay error was `1.23e-7`.

| Fresh 60 Hz 34-trial diagnostic | Outcome-128 | Mirrored-4 |
|---|---:|---:|
| Contacts | 26/34 | 24/34 |
| All-five-kickoff contacts, both sides | 10/10 | 10/10 |
| Stationary seconds | 1.267 | 0.883 |
| Circling windows | 4 | 3 |
| Fallen seconds | 3.35 | 2.50 |
| Short-scenario goals for:against | 5:0 | 11:0 |
| Median turn recovery seconds | 3.367 | 2.908 |

These are mixed physical results. More short-scenario goals and lower supervised loss do not override the failed guard or prove full-match improvement. No additional updates followed. On 19 source sequences excluded from fitting (related scenario families, not independent match confirmation), mean absolute steering non-equivariance **worsened from 0.07145 to 0.08041**. Valid reflection did not guarantee this tiny fit would learn useful symmetry.

The frozen independent 60 Hz full-match screen used fresh seed 10801011 + 1009*i and both sides:

| Mirrored-4 opponent | W/L/unresolved | Goals |
|---|---:|---:|
| Starting Outcome-128 | 4/3/1 | 11:10 |
| Original V13 | 8/0/0 | 42:3 |
| Geometric | 1/3/0 | 2:6 |

An additional fixed Outcome-versus-original benchmark on the **identical** eight seeds/sides was 4/3/1, goals 11:11. On the same four Geometric match conditions, Outcome was 0W/4L, goals 1:7, versus the candidate's 1W/3L, goals 2:6. The original-opponent result is promising, but improvement against the starting reference or diverse opponents remains unproven. Do not compare the new 42:3 directly with the prior candidate package's different seeds. Side pairs are correlated and the sample is small; no 100-match confirmation or promotion occurred.

Separate fresh 120 Hz all-five-kickoff checks gave **2/10 contacts for both** Mirrored-4 and Outcome. Native Geometric's measured 60 Hz callbacks must not be explained by this separate simulator-rate diagnostic. No callback timing or input contract was changed.

## New complete recordings and preservation

The first paired candidate-versus-Outcome seed now has two complete NPZ recordings, **18,411 and 19,023 chronological callbacks per actor**, with exact 13D inputs, strictly increasing physics ticks and actually transmitted controls. Replaying both saved actors from their match start reproduces deterministic modes and steering. The recording manifest pins both actor/checkpoint hashes. These are real RocketSim recordings, not native Rocket League packets, and no recorded-match learning from them occurred in this fit.

Candidate plain actor SHA256: `407e8fe69c510f7947aca1146985384059fb4d2c43093a6533f82e85a54a4f65`. Complete four-update state is `runs/mirrored-own-8-v1/latest.pt`, with original `update-4.pt`, actor snapshots, all eight finite Adam states, Torch/sampler RNG, protocol and source-data hash retained. Inference and complete states match; carried GRU chunk error is `2.38e-7`. Every previously inventoried checkpoint and the existing final ZIP remain unchanged.

The separate MirroredOwn diagnostic Windows kit is sealed and passes non-live Blue/Orange SDK preflight and the untouched official validator: exactly 13 inputs, 26,181 parameters and original decoder. Actual Windows gameplay remains **UNTESTED**. The experimental kit is optional and does not replace Outcome/Geometric or main. No new GitHub Actions success is claimed.

## Remaining submission actions

First finish Windows tests of Outcome and Geometric; Mirrored-4 is only an optional challenger because it improved the original-opponent result but failed the declared contact guard and did not convincingly beat its starting reference. Retain actual scores and complete native prediction/callback logs on both sides. The frozen-clock error classification remains unresolved where original error records lack phase context.

Use the existing `FINAL_V13_WINDOWS_REVIEW.md` for SETUP/CHECK, RLBot bot.toml registration, provenance extraction, sealing, integrity/preflight/official-validator commands and approval-dependent private-repository handoff. New documentation must be resealed. GitHub Actions on a prepared candidate branch still needs to run; previous API dispatch was Forbidden. Main push/merge, selected-model replacement, judge access and any organizer handoff require explicit user approval. Judge identity remains deferred. Deadline: **11 October 07:00 IST / 01:30 UTC**. No further fitting is running or scheduled.
