# One supervised geometric V13 candidate — no promotion

Repository `lakshya-agrawal254/model-wars-submission`; isolated branch `training/v13-geometric-supervised-20261011`, based on passing main `c151310b5c36996aeb1fe192be5eea7cc328e846`. Main model SHA256 stays `bda8a7a773d63526fea8a1449ff3971bba48317a05dd024eafc18827c1100011`. No old RocketSim expert-backed data/models or temporal-credit training are reused. Original baseline and all previously retained checkpoints remain unchanged.

## Training method and provenance

One fit only: exact official 13D, GRU64, 26,181 parameters, tensor names and four-action decoder. Initialization is the existing legally trained scratch-A 16,384-decision checkpoint; its complete PPO state is preserved separately. This is supervised continuation, not a new PPO run. A fresh supervised Adam is appropriate for a different loss; no PPO optimizer state is reset or overwritten.

`Chase` when original observation `ball_present > .5`; `steer = clip(2 * atan2(ball_right, ball_forward), -1, 1)`. Unavailable ball is labeled Neutral/zero steering. Labels use only original observation fields, independent of baseline/opponent outputs. The untouched official baseline is an opponent during data collection, never a source of labels. No pretrained teachers or external outputs. Positive and negative steering signs were verified with actual RocketSim physics; mirrored targets and all five positions/both sides were tested.

80 real, 480-step sequences: 40 wide-orientation ground, 10 kickoff, 20 moving-ball ground, 10 normal starts. **38,400 real simulation frames**, plus eight transparently counterfactual missing-ball augmentations (3,840 samples), not native RLBot recordings. No synthetic forecast labels. Eight epochs, four full independent sequences per accumulated gradient update, 176 updates, lr .001, mode CE + 2×Chase steering MSE, grad clip1. **337,920 supervised sample presentations; zero new PPO decisions.** Do not call repeated supervised samples additional learner decisions.

Data generation 19.44s; fitting 43.92s; combined 63.36s. Complete recurrent trial prefixes retain continuity; independent trial starts use h=None. No shuffled timesteps or hidden resets inside a sequence. Candidate-only forecast encoder columns [3,4,5,9,10,11] are initialized and retained at zero as a training constraint. All 13 runtime inputs are still assembled, delivered and checked normally; no runtime mask or extra parameter/control. Counterfactual forecast perturbations produce bit-identical candidate inference. Actual native callbacks remain untested.

## Independent diagnostic results

Fresh development seeds, matched starts, both sides. No training seed reuse. Ground and wide-orientation cap30s; kickoffs cap90s, four trials per each of five positions. Medians are conditional on contact; machine-readable files also provide capped time estimates and Wilson intervals.

| Policy | Ground contacts /24; median | Wide orientation /24; median | Kickoff /20; median |
|---|---|---|---|
| untouched_baseline | 0; — | 0; — | 0; — |
| retained_A | 2; 9.48s | 10; 21.45s | 18; 50.58s |
| candidate | 24; 0.91s | 24; 7.12s | 20; 3.35s |

Candidate contact Wilson95: 24/24 ->86.2–100%; 20/20 ->83.9–100%. Side pairs are correlated, so per-trial intervals are descriptive. Candidate ground Neutral fraction 2.24% vs retained A91.65%; wide orientations4.25% vs41.73%. Kickoff Neutral fraction9.45% vs8.32%, yet first contact is much faster. Zero Jump/Dodge decisions and actual flips were observed for all policies: no demonstrated improvement in flip choice. Ground latency average .156ms on this cloud CPU (not native latency).

Steering fits imperfectly: final train MSE .1719; independent initial-state MAE .882 and MSE .819 with fresh h=None for each real trial. The candidate can recover through recurrent pursuit despite imperfect initial predictions; do not claim exact imitation of the geometric rule or perfect steering. Precontact closed-loop label MAE .401 standard ground, .408 wide orientations and .117 kickoffs.

## Full-match evidence and stopping decision

Standard RocketSim Soccar, 300 active seconds, kickoff pause, overtime and original documented ball-floor proxy. Existing 900-physical-second safety cap and 120-active-second overtime cap remain unchanged. All caps are unresolved, not wins or draws. Both agents have absent native prediction fields in simulation. Cloud matches are not RLBot/Rocket League execution.

| Comparison | Completed W/L/D | Unresolved | Observed goals for/against |
|---|---|---:|---|
| Candidate vs untouched V13 | 3/5/0 | 0 | 8/7 |
| Candidate vs retained A | 3/3/0 | 2 | 12/11 |
| Retained A vs untouched V13 | 0/4/0 | 4 | 0/4 |
| Untouched V13 vs itself | 0/0/0 | 8 | 0/0 |

Each comparison uses four seed pairs/eight matches; candidate vs baseline and retained A vs baseline share seeds1913011+i×1009. Candidate-vs-A uses distinct matched pairs1914011+i×1009. Candidate score under conservative cap-as-loss treatment is .375 in both comparisons; baseline win Wilson95 is13.7–69.4%. Three wins versus three completed losses in head-to-head provides no credible superiority. Positive aggregate goal difference is insufficient for promotion.

**Screening fails the predeclared >.5 conservative match score. No further fitting, no confirmation seed consumption and no promotion.** Current best retained/legal submission stays scratch-A on main, not this candidate. Contact learning is demonstrated; a stronger winning policy is not. The labels teach pursuit without goal-aware shot selection, defensive positioning, challenge timing, aerials or recovery. Those omissions limit competitive claims. No causes of individual conceded goals are asserted without corresponding telemetry.

## Validation, packaging and submission safety

Ten tests cover labels, real steering direction, all-five kickoffs, exact actor/controls, native observation projection, GRU continuity, candidate full-state reload/optimizer and forecast invariance, original A complete state, and original PPO probability accounting. The old selected-checkpoint test now compares the preserved selected inference asset to its preserved complete state; a separate test explicitly compares the candidate root model to its own full state. No validator checks are weakened.

The experiment branch root is named Lakshya V13 Geometry Candidate for isolated native testing. It does not replace main. Only sanctioned identity and checkpoint hash pins change in runtime source; architecture, observation encoding, decoder and inference sequence behavior remain unchanged. Official validator/workflow/tools are unmodified. The candidate has its own training provenance, sealed inference model and full supervised optimizer/RNG state. Existing A and prior PPO checkpoint files are untouched.

Use `docs/GEOMETRIC_WINDOWS_TEST.md` for clean installs, candidate vs untouched V13, candidate vs current main, both sides and real prediction/callback log inspection. Windows/Rocket League is unavailable here; GUI discovery, launch, gameplay and native acceptance are NOT TESTED. Any promotion remains blocked pending convincing independent results and successful native acceptance. Judge repository access remains deferred at the user’s request. Stop speculative training now; main already has passing official validation well before the 11 October2026 07:00IST deadline.
