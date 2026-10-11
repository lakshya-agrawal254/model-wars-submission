# Outcome-128 provisional entry — measured results

Status:PRIVATE PREPARATION ONLY; NOT SUBMITTED. Actual Outcome-128 RocketLeague/RLBotv5 GUI discovery, launch and full-match acceptance:NOT TESTED. User feedback that Geometric was slightly better than OriginalV13, and validatedA worse, concerns different weights and is not Outcome acceptance.

Checkpoint model/V13.pt SHA256:8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043. Exactly13 observations/26,181 parameters/originalGRU/fourmode decoder. Existing verified sealed weights copied unchanged; preparation performed zero optimizer updates. Identity:lakshya-agrawal254/v13-outcome-128.

## Training

Outcome128 used128 recurrent supervised Adam updates (789289 sequence-row presentations), starting from preservedGeometric. Independently generated realRocketSim legal examples and own-policy trajectories; classbalanced mode and steering learning, original recurrent sequences, no external teachers or runtime controllers. Its training phase added0 PPO decisions. Data/configuration/source hashes and optimizer checkpoint remain preserved privately; inference requires none of those training files. Forecast-related input weights remain at their original trained zeros; Outcome128 cannot claim learned use of two-second predictions. No learned useful front-dodge behavior was demonstrated.

## Frozen full-match RocketSim evidence

These are simulation tests, not actualRocketLeague. Real3D RocketSim2.2.1/RLGym RocketLeague2.0.1, PyTorch2.11CPU/Python3.12.14, deterministic argmax mode and tanh steering, original controls. Both team sides use matching seeds and actual2second RocketSim forecasts (not invented native packets).300active-second regulation; floor proxy, capped overtime and total duration. Censored matches remain unresolved, not draws.

| Seed set / opponent | W/L/D/Unresolved | Goals for:against |
|---|---|---|
|10501011+i*1009 / OriginalV13,8matches|6/1/0/1|17:9|
|10501011+i*1009 / Geometric,12matches|6/4/0/2|19:12|
|11101011+i*1009 / Ground64,8matches|6/1/0/1|10:4|
|11101011+i*1009 / Geometric,8matches|7/1/0/0|23:7|

Small side-paired samples have substantial uncertainty. These opponents are independently preserved legal project models; none is included as a second bot here. The latest forecastPPO challenger won5/8 againstOutcome but failed common-opponent qualification,7/16 wins versusOutcome's13/16; its weights are excluded. Corrected read-only telemetry reproduced all44 original outcomes; these are44 unique conditions, not88 independent samples. Evidence source commit:f7e872f0f3a2bb47fa73d5d239e0c85f01d66f60, docs/evidence/forecast-final-v1 in the private development branch.

Important remaining weaknesses:Outcome spent892.78seconds stationary over8 latestGround64 matches;887.70seconds were near the other car. Winning records do not prove reliable collision recovery. Circling and backward/unproductive contacts remain. Four separate120Hz forecast-candidate-versusOutcome matches reached safety caps scoreless; neither agent made useful contact. Native logs previously supplied by the user were approximately60Hz, so120Hz failures are not asserted to explain native poor gameplay.

The 13D local-ball/speed/timing contract lacks opponent positions and global goal-relative heading. It cannot guarantee general own/enemy-goal identification; no privileged coordinates enter runtime inputs. No claim of direct opponent tracking or strong defending is made.

## Validation and native acceptance

Sealed source participant-manifest verification:PASS before copying. Preparation checks cover exact checkpoint bytes, runtimePIN, consistentidentity, originalpolicy/observations/decoder, all8tensors/26181parameters, full/recurrent inference parity, integrity and officialvalidator. The final preparation verification receipt records actual pass/fail outputs. Cloud SDK preflight is NON-LIVE and must not be called native acceptance. GitHubActions status is distinct and must be checked for the exact preparation commit; dispatch access may be blocked in this cloud session.

Native acceptance evidence to complete:Windows/Python/RLBot/RocketLeague versions; exact package/commit/hash; Blue/Orange full-match scores; actual prediction receipt and callback/error logs; GUI discovery/launch/control success. Do not mark acceptance passed without those observations. Preserve any failures in this document, reseal and rerun validation after edits.

Preparation results: exact4096-callback real-state inference replay matched sealedOutcome controls, modes andGRU hidden states bit-for-bit. Local officialvalidator fetched from current organizer main passed; packagedvalidator was byte-identical. Native SDK preflight passed bothteam configuration and originalactions; nativegame remainsUNTESTED. Actionsdispatch returnedForbidden through the cloud API before a run could start. Do not transfer the passing main commit's CI result to this preparation commit.

## Final bounded PPO challenger — rejected,11October2026

Training stopped at06:20:09IST after549.94seconds,196608 new decisions and36 actor updates. The new actor was isolated; this package still contains the unchanged Outcome-128 weights. All71 prior checkpoints were preserved. The challenger passed exact13D/26181-parameter and recurrent/on-policy/optimizer-restoration checks, but failed competitive qualification.

Fresh seeds11401011+i*1009, four side pairs per matchup, deterministic60Hz realRocketSim full matches:

| Playing model / opponent | W/L/D/Unresolved | Goals for:against | Stationary seconds |
|---|---|---|---:|
| Challenger / Outcome128 |6/0/0/2|12:5|4.72|
| Challenger / Ground64 |1/5/0/2|3:11|274.12|
| Challenger / Geometric |2/6/0/0|5:10|1.77|
| Unchanged Outcome128 / Ground64 |6/2/0/0|14:9|7.10|
| Unchanged Outcome128 / Geometric |2/4/0/2|11:11|4.25|

Common-opponent totals:challenger3/16 wins,8:21goals,275.88seconds stationary; Outcome8/16 wins,25:20goals,11.35seconds stationary. Challenger near-car stalls againstGround lasted268.60seconds overall. The direct6/8 screen does not offset these regressions and does not warrant independent confirmation. Its optimistic independent-match95% Wilson win interval againstOutcome is40.9–92.9%; four correlated side pairs do not establish strong statistical evidence. No more training is scheduled.

Outcome remains the provisional recommendation. Its fresh2/8 wins againstGeometric also show substantial seed sensitivity; do not extrapolate earlier small winning samples to native strength. No actualRocketLeague acceptance is claimed. No learner, experimental weights or other bots are included here. Full immutable research evidence is preserved on private branch training/v13-near-greedy-final-20261011, docs/FINAL_NEAR_GREEDY_REVIEW.md.
