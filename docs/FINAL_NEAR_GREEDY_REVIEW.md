# Final bounded V13 PPO campaign — rejected; retain Outcome-128

Private preparation only. No main replacement, merge, organizer access or tournament submission is authorized. No further training campaign is scheduled. Outcome-128 remains the provisional playing recommendation; actual Windows Rocket League acceptance is pending.

## Identity and budget

Repository: lakshya-agrawal254/model-wars-submission (private). Research branch: training/v13-near-greedy-final-20261011. Training/evaluation source commit: 7bede3c3c8f7bc8ba7d430328c3df48345f79b72. Main remains c151310b5c36996aeb1fe192be5eea7cc328e846.

Initializer: runs/outcome-flip-learning-v1/candidate.pt, SHA256 8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043. No rejected actor was used as initialization.

Rejected candidate: runs/near-greedy-final-v1/candidate.pt, SHA256 0e484068fda33199302ccf8de96cfbae67a5e071c2041128aedad82908c4b8ed. Complete state: latest.pt and all36 update-N.pt checkpoints, with actor/critic optimizers and RNG states; every actor-N.pt is retained.

Training completed 11 October2026 at06:20:09.274IST (00:50:09.274UTC), before both user cutoffs. Actual elapsed549.939seconds (9min10sec);196,608 additional learner decisions,36 actor and48 critic optimizer updates;357.51 effective learner decisions/second including collection and optimization. Evaluation took169.47seconds. No learner remains running.

## What changed and what was verified

The previous forecast run used24,576 decisions, four actor updates, and approximately35% sampled Jump/Dodge while greedy behavior remained Chase. That mismatch and sparse outcomes were hypotheses, not established sole causes of its failure.

This one bounded experiment used three fresh policy/data cycles, each8 complete8192-step trajectories at60Hz, four simulation workers, two learner threads and12 actor updates. Training-only categorical temperature0.35 and steering standard deviation0.10 brought exploration closer to deterministic play; PPO likelihoods used exactly that behavior distribution, including the tanh Jacobian and steering only for Chase. ActorAdam learning rate0.0001; existing gamma/lambda0.999, clip0.15, small reference joint-KL coefficient0.02. No geometric mode-cross-entropy labels or external teachers were added. Runtime argmax/steering remained unchanged. The final complete checkpoint was selected for screening, not a reward-selected intermediate snapshot.

All four frozen legal opponents (Outcome, Ground64, Geometric, Original) occurred on both sides in every batch. First-batch scenarios already included normal matches, attack, defence, moving-ball and recovery starts; second batch covered all five kickoff positions. Actual RocketSim two-second forecasts reached all recorded observations. These are simulator forecasts, not invented native RLBot recordings, and cannot anticipate future car interventions.

Chronological GRU trajectories began with h=None only at new physical trajectories. Memory and clock continuity were preserved across goal resets. Time limits used nonterminal value bootstrapping; goal resets included the potential correction. Fresh batches used the current actor and newly collected rollouts. Streaming hidden states and exact stored action log-probabilities matched chronological replay for all24 trajectories. Full-sequence and chunked inference matched. Gradients were finite for all eight actor tensors. Plain/full checkpoint inference matched exactly, and both optimizers restored. All71 pre-existing checkpoint files remained byte-identical.

The actor remains the original13-observation,26,181-parameter encoder/GRU/head, original four controls and tensor names. Official policy, observation contract, validator and workflow files are unchanged. The actor PPO optimizer was newly created from Outcome's supervised actor; its original supervised optimizer remains preserved. Only the existing compatible training-only critic and criticAdam state were carried from the earlier own-project anchored run; its rejected actor was not loaded.

## Learning diagnostics and limitations

Training encountered11 goals for and11 against,125 contact callbacks,37 goalward and36 backward qualified contact events. Goal rewards remained+10/-10; no coefficients were tuned. Separately summed components: goal0 (22 nonzero events, not no goal signal), potential7.14457, productive velocity0.323297, meaningless-contact-0.006, stationary0, goal-reset potential-0.437271.

Sampled modes were Neutral2, Chase185815, Jump5788, Dodge5003. On all196,608 recorded states, the final deterministic actor still chose Chase, with zero mode changes against Outcome. Mean steering change against Outcome was0.002253. Zeroing the six forecast-related fields changed steering by only0.000986 on average and changed zero modes. This shows a small forecast influence, not demonstrated anticipation or tactical learning. No actual dodge events occurred in the40 screening matches.

The sampled rollouts never incurred the sustained-stationary penalty, yet deterministic matches contained long near-car stalls. The remaining sampled-versus-greedy difference therefore failed to expose/train away a relevant evaluation failure. This is evidence of a coverage/behavior mismatch, not a controlled proof of the sole cause. More updates and forecast gradients did not solve the competitive plateau. No further training is justified now.

True goal coordinates were used only for rewards/evaluation. The legal13D inputs have no opponent positions or global goal-relative heading. Direct opponent tracking is unavailable; general reliable own/enemy-goal identification is not guaranteed. Earlier real-state rotational aliasing evidence remains applicable. Recurrent context can help in known situations but does not remove this observability limitation universally.

## Fresh deterministic full-match screen

Seeds11401011+i*1009, i=0..3; both sides for every seed and matchup.40 unique matches at60Hz; real RocketSim120Hz physics,300active-second regulation, floor proxy,120second overtime cap and900total simulation-second safety cap. Capped games are unresolved, never relabeled as draws. No120Hz or native RocketLeague result is claimed for this candidate.

| Playing model | Opponent | W/L/D/U | Goals for:against | Stationary seconds (8 matches) | Circling windows proxy |
|---|---|---|---|---:|---:|
| New candidate | Outcome-128 |6/0/0/2|12:5|4.72|14|
| New candidate | Ground-64 |1/5/0/2|3:11|274.12|12|
| New candidate | Geometric |2/6/0/0|5:10|1.77|18|
| Unchanged Outcome-128 | Ground-64 |6/2/0/0|14:9|7.10|12|
| Unchanged Outcome-128 | Geometric |2/4/0/2|11:11|4.25|13|

On the same common opponents the candidate won3/16, scored8 and conceded21; Outcome won8/16, scored25 and conceded20. Candidate stationary time275.88seconds (5.004% of observation time) versusOutcome11.35seconds (0.208%). Candidate near-other-car stationary time againstGround alone268.60seconds; two games each had119.58seconds stationary. These are serious movement regressions, not evidence of improved collision recovery. Circling remained30 windows versus25 forOutcome across common opponents. Contact, missed-approach, collision-recovery and defensive-clear measurements are proxies; full raw records retain their definitions and censoring.

Four independent side-paired seeds give substantial uncertainty. The optimistic independent-match Wilson95% win interval for6/8 againstOutcome is40.9–92.9%; pairs are correlated. Pair-bootstrap estimates are exploratory with only four clusters and do not establish strong confirmation. The gates were recorded before screening: at least6/8 direct wins, common wins and goal difference not worse, stationary-fraction increase at most1percentage point, and circling rate increase at most25%. The candidate failed common wins, goal difference and stationary criteria. It does not qualify for the100-unused-match confirmation stage. No repeated tuning or intermediate-checkpoint cherry-picking was performed.

Outcome's2/8 wins againstGeometric in this fresh set also cautions against overstating its strength from earlier favorable small samples. Outcome is retained as the stronger available provisional entry, not certified as a strong native-game bot.

## Preservation and final preparation

Compact evidence is in docs/evidence/near-greedy-final-v1. All campaign checkpoint and optimizer files are retained locally and in a verified private Drive research backup set; each of the three chronological NPZ datasets is a separate checksum-verified part. No cumulative old-export ZIPs are nested and no old checkpoint was deleted. The inference-only Outcome preparation remains separate and contains one bot and one checkpoint.

Outcome preparation branch: preparation/outcome-128-provisional-20261011. Selected model SHA remains8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043. Local unchanged official validator, sealed manifests and non-live RLBot SDK preflight are checked separately from native acceptance. Cloud Actions dispatch previously returnedForbidden; the owner must run the unchanged workflow on the exact preparation commit through GitHub UI. Do not treat old main's passing CI as this branch's CI.

Windows: extract the single-bot package into a new directory, run SETUP.cmd and CHECK.cmd, register root bot.toml, then play complete Blue/Orange matches against separately installed OriginalV13/Geometric and preserve scores/prediction/callback/error logs. Update the results, reseal and revalidate if those documents change. Only after explicit user approval may main be replaced or organizers receive access. The07:00IST deadline is not authorization to submit.
