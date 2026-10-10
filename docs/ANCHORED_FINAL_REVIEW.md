# Anchored PPO review — stopped, no promotion

Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`. The user must explicitly authorize any replacement, merge, organizer access or submission. This experimental branch is review-only.

One configuration, 32,768 new on-policy decisions on four workers. Actor initialized from our geometric V13 (no external weights/labels), frozen own reference KL and independent geometric supervision anchors. Exact 13D observation contract, GRU64, 26,181 actor parameters and original four-action decoder remain unchanged. Six forecast encoder columns remain zero; native observation delivery is still required. Both PPO optimizers, critic, actor, RNG and reference/data/source hashes are preserved. Fresh PPO Adam was created once at the supervised-to-PPO transition; no reset comparison or repeated campaign.

4096 chronological decisions per worker, actual 60/120Hz callback durations, gamma and GAE lambda .999 at60Hz, correct time-limit bootstrapping and nonterminal goal reset potentials. GRU hidden states persist across goal resets. Old rollout log probabilities are verified against recomputed actual behavior densities before updating. Goal rewards ±10 dominate smaller velocity-qualified contacts and potential shaping. Training-only privileged state never enters the actor.

The declared 16-rollout scenario cycle is 8 normal-match, 3 attack, 3 defense, 1 all-five kickoff and 1 wide-ground starts. **This stopped pilot completed only the first eight normal-match rollouts.** Supervised anchors included all-five kickoffs, but no targeted attack/defense RL starts were reached. Do not interpret this pilot as testing the complete scenario mix.

## Fresh matched-side screening

Full RocketSim Soccar regulation 300 active seconds plus overtime; existing floor proxy and safety/OT caps preserved. Native forecast data are absent. These are not actual Rocket League/RLBot matches. Four seed pairs per comparison; caps are unresolved. Small samples cannot establish tournament strength.

| Comparison | Completed W/L/D | Unresolved | Observed GF/GA |
|---|---|---:|---|
| candidate_vs_baseline | 3/3/0 | 2 | 4/4 |
| candidate_vs_A | 5/2/0 | 1 | 5/4 |
| candidate_vs_geo | 0/8/0 | 0 | 1/10 |
| geo_vs_baseline | 1/7/0 | 0 | 3/9 |

| Policy | Ground contacts/12; median seconds | Kickoff contacts/10; median seconds |
|---|---|---|
| candidate | 12/12; 0.8916666666666666 | 10/10; 3.35 |
| geo | 12/12; 0.8916666666666666 | 10/10; 3.35 |
| A | 0/12; None | 9/10; 50.583333333333336 |
| original | 0/12; None | 0/10; None |

Training measured 42.25s, 775.6 learner decisions/s excluding evaluation/export. Total actor lineage PPO49,152; prior337,920 supervised presentations are not PPO decisions. Evaluation timing and all120 raw records retained.

**Stop: no extension, no unused-seed confirmation, no promotion.** The candidate loses 0W/8L to its frozen geometric reference. Winning some matches against retained A does not overcome that failure or establish superiority. Reference KL and supervised anchoring preserved contact ability but did not demonstrate competitive improvement. Tiny joint-KL changes and weaker head-to-head suggest the bounded PPO update did not resolve strategic limitations; this does not prove a fundamental architectural bottleneck. No further hyperparameter tuning from this screen.

Telemetry distinguishes productive/backward velocity changes, nonproductive contacts, stationary time, circling and own-last-touch conceded goals. Clears, circling and own-last-touch are explicitly proxies; they do not prove causal saves/own goals. Raw compact results include controls and latency.

## Proposed selection and approval boundary

Retain current validated A on main for review; no new candidate is justified for replacement. Original starter, all retained legal states and prior private backups remain unchanged. Geometric candidate remains an isolated mechanical improvement, not a verified stronger match-playing model. Review the existing Windows package and test candidate/control separately using `docs/GEOMETRIC_WINDOWS_TEST.md`; native game execution remains untested. Never infer native acceptance from RocketSim or validator success.

Official unmodified validator is run on a separate sealed anchored-candidate staging kit, plus repository tests and integrity restoration. No judge access or organizer distribution. Any future submission requires explicit user approval; the deadline does not authorize submission.
