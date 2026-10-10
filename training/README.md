# Separate V13 PPO feasibility campaign

This private project is independent of ModelWars-RocketSim. It uses the organizer's unchanged 13-input, one-layer 64-unit GRU Policy with exactly 26,181 actor parameters and its original four modes / steering decoder. The critic is training-only, never part of the submitted actor. There are no Nexto/Necto assets, pretrained outputs, labels, old checkpoints, external datasets, runtime scripted overrides or additional actor controls.

The original V13 initialization is preserved at `runs/baseline/V13.pt` and original Git commit `83d92a69c80930652c4d26be8db9f7ebea062e38`. The selected scratch candidate is a fresh permitted initialization of exactly that Policy class, subsequently trained only with PPO from real RocketSim experience. It is not a padded imported network or imitation model.

## Domain limitation

RocketSim training calls the original observation builder and selects exactly its first 13 features. Native world orientation is used on both teams. Predictions are **explicitly unavailable**: prediction positions, validity and time fields are zero. No native RLBot BallPrediction is synthesized. The game-facing official runtime is unchanged and will use real native predictions when delivered. Therefore this campaign does not establish transfer to normal prediction-valid Rocket League callbacks. Simulator trials use 60 Hz decisions, versus the actual callback cadence in RLBot. These are readiness blockers, not equivalence claims.

## Reproduce

Use Python 3.12 and CPU PyTorch 2.11.0; install the official `requirements.txt`, then `training/requirements.txt`. Run from this repository root (PowerShell paths work with the same arguments):

```
python -m unittest discover -s tests -v
python training/ppo.py --scratch --stage A --steps 512 --decisions 16384 --output runs/reproduced-A
python training/evaluate.py --model runs/reproduced-A/candidate.pt --kind matches --seed 831011 --pairs 4 --output runs/reproduced-screen.json
```

After restoring the complete private training backup, continue the selected policy **only for a justified bounded experiment**, for example:

```
python training/ppo.py --resume runs/retained/selected/latest.pt --stage A --steps 512 --decisions 8192 --output runs/next-bounded-A
```

This command continues the actor, critic, Adam states, RNG and episode counter, not the starter. It is an example, not an instruction to repeat the failed campaign. Do not reuse screening or confirmation seeds for tuning. No training is left running after this task.

## PPO and reproducibility

Behavior is categorical mode plus conditional tanh-Normal Chase steering; likelihood includes the squash Jacobian only when steering is executed. PPO recomputes full chronological GRU trajectories from each trial's new-instance state; goals do not reset memory. Four epochs, clip 0.2, gamma 0.99, GAE 0.95, Adam actor 3e-4 / critic 1e-3, entropy 0.025, gradient clip 1, KL stop 0.03. Artificial time limits bootstrap the next value. No pre-change rollouts are reused.

Atomic complete checkpoints contain actor/critic, both optimizer states, Torch/Python/NumPy RNG, episode and decision counters, configuration, source hashes and history. Saves occur at completed trial/update boundaries: there is no in-flight trajectory or live hidden state to recover. Both policies start new match instances with hidden=None; within-match goals preserve it.

The first starter continuations used the earlier zero-bootstrap bug, disclosed in their compact results. Their source snapshots are retained with evidence. The fresh scratch campaign fixed time-limit bootstrap. Optional-opponent constructor RNG consumption was corrected after training; original used-source snapshots are preserved. All matches are scored from RocketSim goals, never training reward.

## Bounded retention

Private backups retain: original baseline, selected complete A checkpoint, latest complete A4 learner checkpoint, the required actor dependencies, at most four legal historical actor policies, used-source snapshots and compact results. Failed candidate weights are not permanent archives. Upload a new managed set to a new file, download it, checksum and strict-load complete state, update the retention manifest, then retire only unreferenced managed backups. Do not delete old-project backups or unrelated Drive files. ZIPs never contain previous ZIPs.
