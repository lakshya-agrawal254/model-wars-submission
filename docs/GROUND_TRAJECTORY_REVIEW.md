# Ground trajectory experiment: private review, no promotion

Ground-64 improved short ground-movement trials but failed the declared competitive and full-match movement gates. Geometric remains the preferred playing reference; Outcome-128 is preserved. Ground-64 is an optional Windows diagnostic, not a verified stronger model. Training has stopped. Main and every historical checkpoint remain unchanged.

## Models

- Geometric plain actor SHA256: `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`.
- Outcome-128 plain actor SHA256: `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.
- Ground-64 plain actor SHA256: `d79cac01aefe8d161b2ec5f8154fac608f554546a96644f0545b4cb254c6276a`.
- Ground-64 inference: `runs/ground-trajectory-learning-v1/screened-actor.pt`; complete actor/Adam/RNG state: `update-64.pt` in that folder.
- Rejected latest 96-update learner: `latest.pt` in that folder. All snapshots at 16, 32, 64 and 96 updates are preserved.
- Main remains `c151310b5c36996aeb1fe192be5eea7cc328e846`.

## Native logs: actual findings and limits

The supplied ZIP contains two JSONL files. The active process loaded the correct Geometric checkpoint and captured only Blue. Its 94 processed packets and 94 logged submissions are periodic samples: retirement reports **11,197 actual submissions and 11,332 callbacks**. The sampled 82 Chase / 12 Neutral counts are not exhaustive frequencies.

Seven sampled Active-phase packets between T=60.35 and 72.36 show Neutral, a visible ball and zero horizontal velocity. This establishes policy-driven stopping at those samples, separately from later callback errors. It does not prove uninterrupted stationary time between samples. Five other Neutral samples are GoalScored/Replay with the ball absent. The logs omit complete chronological observations, raw logits and scores, so they cannot support faithful GRU replay or native win-rate claims.

All 135 duplicate/backward-clock errors report T=194.4583282470703 and frame 23335. The last sampled successful packet is Countdown at T=193.81666565. Error records omit phase and last accepted timestamp; no successful packets follow the burst. Repeated reported clocks establish stale/frozen packet timestamps, **not whether the actual game was paused or actively simulating, or the backend cause**. The guard rejects before inference/submission. Retirement's `diagnostic_errors=0` counts logging I/O errors, not callback exceptions.

No callback timing, observation delta, hidden resets or error suppression was changed. Review logging now records error phase/code/last accepted time and reraises the original exception; a synthetic SDK fixture verified this behavior. That fixture is not actual-game evidence.

## One changed training method

The previous steering readout fit froze the encoder/GRU and supplied pointwise pure-pursuit targets even when movement deteriorated. This experiment started from preserved Outcome-128 and trained the existing encoder, GRU and heads on complete, **outcome-selected closed-loop trajectories**. No PPO, external teacher or old-project model was used.

Real RocketSim search covered 34 scenarios and 408 candidate controllers: all five kickoffs on both sides, curved moving-ball approaches, behind/side-on turns, post-miss recovery and ground shots. Outcome-128 drove each 2–3 second prefix. Training-only controllers varied forecast lead, speed-scaled turn gain, angular-change damping and persistent turn direction. They read only legal observation features; simulator geometry scored outcomes. None is exported into inference.

Twenty-one manoeuvres qualified: eight kickoffs, five curved approaches, two U-turns, three side-on turns, two recoveries and one shot. On their matched starting states the controllers produced 21 contacts versus Outcome's 12, zero circling windows versus four, and goals 8–0 versus 0–3. These are **demonstration results, not learned-policy results**. Aggregate stationary time was 0.917 versus 0.717 seconds; per-trial selection allowed 0.1-second tolerance. Only one of six shot scenarios qualified, so shot coverage was limited.

The dataset contained 55 full chronological sequences and 44,501 real observation rows. One fit completed **96 Adam updates and 611,668 supervised row presentations; zero PPO decisions**. Collection, fitting and development checks took 124.26 seconds; physics search took 75.54 seconds. Fresh supervised Adam was used for this different objective, while the original optimizer states remained preserved. Sequence-level reference mode KL and driving/steering examples protected existing behavior. This is supervised off-policy learning, not on-policy PPO.

Existing forecast-position inputs 3–5 and validity flag 9 could learn from actual RocketSim two-second forecasts. Encoder columns 10 and 11 remained at their original zeros to avoid relying on simulator/native horizon-offset differences. Training-only forecast-availability dropout preserved the runtime's 13D contract. No fabricated RLBot predictions were used.

At update 96 the fall guard failed: 4.65 fallen seconds versus control's 2.27. The development-only selection rule froze update 64 **before fresh match seeds were consumed**: 29/34 contacts and two circling windows, versus control's 25/34 and five. No later retuning or second fit occurred.

## Fresh independent movement trials

There were 38 trials per policy/rate, with unused seeds, changed geometry, both teams and all five kickoffs. Four long-range side-on cases were motivated by native stopping. Actors used deterministic inference on every callback; no search driver was present.

| Policy / Hz | Contacts | Kickoffs | Conditional median contact (s) | Stationary (s) | Circling windows | Median turn recovery (s) | Goals for/against |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_60 | 38/38 | 10/10 | 3.625 | 1.500 | 1 | 2.683 | 12/0 |
| candidate_120 | 23/38 | 2/10 | 5.533 | 1.250 | 4 | 2.750 | 4/0 |
| outcome128_60 | 31/38 | 10/10 | 3.817 | 1.350 | 5 | 3.217 | 7/1 |
| outcome128_120 | 9/38 | 0/10 | 10.633 | 1.408 | 4 | 3.142 | 1/1 |
| geometric_60 | 26/38 | 10/10 | 3.350 | 28.850 | 6 | 3.033 | 10/1 |
| geometric_120 | 13/38 | 2/10 | 10.483 | 33.100 | 3 | 3.225 | 1/0 |

Contact medians exclude misses; Geometric's lower 60 Hz median excludes 12 failed trials. Turn recovery means bearing below 0.55 radians with speed above 150 after a side/behind state; it does not measure every collision or landing recovery. Ground's 60 Hz fallen time was 4.267 seconds versus Outcome's 4.200 and Geometric's 4.150. Post-contact stationary time was 0.083/0/0 seconds. Ground contacted in all four long-range cases; Geometric stalled in two. Those states were motivated by the native observations, not an exact reconstruction of native GRU history.

## Full-match evidence: reject promotion

Forty-four deterministic matches used fresh seed groups, paired team sides and identical physics/forecast/decoder conditions. Eight matches per 60 Hz comparison; four per 120 Hz comparison. Evaluation took 343.85 seconds. Regulation was 300 active seconds, overtime capped at 120 seconds, total time capped at 900 seconds. Kickoff/ball-floor clock handling remains a declared simulator proxy, not verified native-game clock parity. Capped matches remain unresolved even when someone leads. Hidden state carried across goals/kickoffs.

| Matchup / Hz | W/L/D/unresolved | Goals for/against | Stationary (s) | Circling windows | Goalward/backward contacts |
|---|---:|---:|---:|---:|---:|
| candidate_vs_outcome128_60 | 4/3/0/1 | 10/12 | 4.750 | 11 | 142/137 |
| candidate_vs_geometric_60 | 4/2/0/2 | 14/9 | 168.450 | 6 | 154/167 |
| outcome128_vs_geometric_60 | 4/4/0/0 | 13/8 | 3.600 | 10 | 78/75 |
| candidate_vs_original_60 | 4/3/0/1 | 25/15 | 4.467 | 5 | 219/161 |
| candidate_vs_outcome128_120 | 0/3/0/1 | 3/6 | 1.892 | 6 | 59/34 |
| candidate_vs_geometric_120 | 2/2/0/0 | 4/5 | 2.092 | 8 | 37/34 |
| outcome128_vs_geometric_120 | 0/0/0/4 | 2/6 | 1.933 | 1 | 7/14 |

Four wins out of eight against Outcome is not convincing superiority: the IID Wilson 95% interval is 21.5–78.5%, and only four seed groups were side-paired, making that IID interval optimistic about effective evidence. The declared screen required at least 6/8 wins against both references plus reliable movement. It failed; no confirmation seeds were consumed or tuned.

The candidate's 142 goalward versus 137 backward contacts and 10–12 score against Outcome show why more contacts were insufficient. Contact direction, circling and own-last-touch are proxies, not causal shooting/save judgments. Outcome itself went 4–4 against Geometric on these fresh seeds, rather than the earlier 4/4 preliminary result: the old records are preserved, and neither result should be treated as a settled ranking.

### The full-match stopping failure

One Orange-side match against Geometric, seed 10303029, produced **162.03 stationary seconds** despite all 18,378 actor callbacks selecting Chase and throttle 1 reaching the engine. Read-only telemetry replay reproduced the original score, end and stationary result. Median stationary speed was 45.2 uu/s, steering 0.972, bearing 1.940 and car-to-car distance 148.1 uu. Nearly 99.75% of stationary samples had the opponent within 200 units; wheels were grounded, with no handbrake, demo or autoflip. Only 13% were wall-near. This is predominantly car-to-car pinning/high-steer creeping, not Neutral collapse or callback failure. The short curriculum did not cover this late full-match failure. The bot won that match 1–0, illustrating why a win alone does not establish reliable movement.

### Optional Jump physics check — not a learned fix

On that consumed seed, one training-only Jump pulse while wheel-contacting the wall at height 966.6 uu broke the contact sequence. A 0.10-second pulse reduced stationary time from 162.03 to 3.43 seconds, preserving the 1–0 score. A 0.15-second pulse produced 3.78 stationary seconds and a 2–2 result. Both showed physical jumping and zero dodges. This establishes a possible recovery action under the decoder, **not learned short-hop behavior or improved match performance**. The trigger used privileged diagnostic state; it is not an eligible runtime controller. No further optimization occurred, and no pulse/script is in any playing package. Front-flip training remained separate.

## Prediction inputs: useful influence, limited evidence

On one real recorded prefix, forecast invalidation changed mean absolute steering by 0.02793 (maximum 0.13451), with no mode changes on that prefix. Learned forecast-column norm was 0.12564. This establishes influence, not winning improvement.

A separate 22-trial, paired-side controlled ablation used only in-memory model copies: actual forecasts produced 22 contacts and ten goals versus 21 contacts and seven goals with four forecast columns removed. Median contact time was 2.483 versus 2.617 seconds. Tradeoffs were worse fallen time (1.783 versus 1.200 seconds) and slower turn recovery (3.567 versus 2.350). No model was saved or tuned from that test. These inputs can help some interception cases; global competitive and native benefits remain unproven.

## Compliance, preservation and remaining limits

The exact original 13D inputs, 26,181 parameters, GRU64, eight tensor names, four modes, decoder and callback/hidden-state contract are preserved. There are no new controls, external weights/labels or runtime movement scripts. Complete 64/96 states restore all eight Adam states, RNG, data and source snapshots. Finite data, whole-sequence/single-step/chunk carry parity and deterministic repeat pass. A synthetic missing-ball interval after a real prefix selected Neutral on 117/120 callbacks for Ground/Outcome and 116/120 for Geometric; this is not native goal/replay acceptance.

The unmodified official validator and SDK preflight pass Ground-64. Actual Windows GUI/gameplay acceptance remains **untested**. New GitHub Actions execution is blocked by Forbidden API permissions; no new Actions pass is claimed. Main is unchanged. No judge access, main push/merge, model replacement or submission occurred.

Private Windows review offers Geometric preferred, Outcome reference, Ground diagnostic, original V13 and optional A. Full backup contains every new checkpoint, optimizer/RNG/data/source state, compact results and the supplied raw JSONL. Existing backups remain separate; no export ZIP is nested, and nothing was deleted. Follow GROUND_WINDOWS_REVIEW.md. Integrity receipts accompany verified private Drive exports.
