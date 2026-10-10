# V13 legal flip feasibility review — NOT SUBMITTED

Selected playing model remains Geometric V13, plain actor SHA256
`55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`.
Main remains c151310b5c36996aeb1fe192be5eea7cc328e846. No old checkpoint deleted.

## Physical decoder proof
Original modes: Neutral (no inputs), Chase (throttle1 and continuous steering),
Jump (jump true), Front Dodge (jump true, pitch -1). Both Jump and Dodge remove
throttle and steering. Boost/brake/reverse/roll control are unavailable. Mode3
on the ground does NOT imply an actual dodge. RocketSimEngine with rlbot_delay
advances one tick BEFORE applying each tick's controls. This existing delay is
unchanged. At60Hz each callback control lasts two120Hz ticks; at120Hz one tick.

Across both sides and rates, firstJump0.1s → Chase-release0.1s → Dodge0.05s
created actual RocketSim is_flipping transitions. HeldJump without release,
secondpress after1.6s release, and a single groundDodge produced zero dodges in
these tests. We did not determine every possible dodge-window boundary.
The initial physics proof used fixed straight steering; the demonstration
curriculum used Geometric steering on kickoffs. This distinction matters.

Centred kickoff at60Hz: straight Chase first contact3.90s, first-hit ballspeed
1926.39UU/s; releaseDodge3.4833s,2359.16UU/s. Straight approach fromspeed1000:
Chase1.45s/1945.89UU/s, releaseDodge1.3333s/1168.82UU/s. Thus faster contact can
be a WEAKER contact. These are symmetric constructed cases, not match results.
Additional close-contact timing onset0.6/0.9/1.1s succeeded physically on both
sides/rates but did not establish a useful learned skill. First-wheel-contact
recovery is a proxy, not stable useful recovery. Raw trace and ballvelocity
changes are preserved. No runtime flip script was installed.

## One bounded learning proof
Start: unchanged full Geometric checkpoint, including compatible Adam states.
14 of24 legal demonstration trajectories had both an actual flip and contact;
allfive kickoff positions, bothteams and60/120Hz were represented in collection.
Training-only independently generated sequence labels; no pretrained bot or
teacher outputs. Reference anchors are from our own Geometric policy.
Eight full-chronological-sequence Adam updates (58,392 row presentations), existing8 actor parameter
states carried fromstep176→184, lr0.0001; no critic or PPO rollouts.
Input remains13D; actor architecture26181parameters, tensor names, original
mode decoder, callback timing and recurrent inference are unchanged. GRU is
trained, not reset on every example timestep. Each chronological episode starts
withNone hidden; whole trajectories include pursuit, jump, release, second
press and aftermath. The inference evaluator preserves hidden over goal resets.
CORRECTION: the preceding interception full-match screen actually retained zero prediction fields; its forecast-world injection targeted the torch decorator wrapper rather than the underlying match function. Original raw records are preserved, with this correction. The current corrected flip screen hooks the underlying function and verifies the observation domain. Its complete-match forecasts come from realRocketSim ballphysics, not fabricated
native RLBot recording. Real Windows acceptance remains untested.

{
  "baseline": {
    "trials": 24,
    "contacts": 12,
    "flips": 0,
    "stationary_seconds": 1.4166666666666665,
    "airborne_seconds": 4.05,
    "kickoffs": {
      "60": {
        "contacts": 10,
        "trials": 10
      },
      "120": {
        "contacts": 0,
        "trials": 10
      }
    }
  },
  "candidate": {
    "trials": 24,
    "contacts": 12,
    "flips": 0,
    "stationary_seconds": 1.4166666666666665,
    "airborne_seconds": 0.0,
    "kickoffs": {
      "60": {
        "contacts": 10,
        "trials": 10
      },
      "120": {
        "contacts": 0,
        "trials": 10
      }
    }
  },
  "reject_reasons": [
    "No deterministic learned dodge events"
  ],
  "optimizer_updates": 8,
  "new_ppo_decisions": 0,
  "training_seconds": 18.958635240007425,
  "selected_demonstrations": 14,
  "candidate_sha256": "7b0f26dc751c0f87867d2101b883419e6005c9d26a41e687322edcc404c67f19",
  "native_game": "UNTESTED",
  "promotion": "NONE"
}

The new-seed kickoff/straight-start diagnostics reuse the prescribed fixed start geometries; they are not unseen-state generalization tests. A separate unused-speed/distance/lateral-offset ground set and fresh full matches are retained.

No deterministic learned dodge events: REJECTED. A decrease in CE loss is not
successful flip learning. No additional training or PPO extension followed.
Demonstration selection by flip+contact is insufficient to ensure productive
contacts; the weaker straight-approach hit demonstrates that limitation.
The eight-update budget and strong movement anchors also limit what can be
concluded: this failure does NOT prove the GRU cannot learn flips.

## Fresh full-match screen
Bothteams, matched seed9873011, deterministic argmax/continuous mean steering,
60/120Hz, same original/ownA/Geometric opponents.300active-second regulation,
ball-floor ending proxy,120second OT cap,900second global safety cap. Existing
kickoff clock pauses until the ball moves; consequently long untapped kickoffs
can censor matches. Every cap is unresolved even if one agent leads. This is
not verified exact native Soccar timing or actual RocketLeague gameplay.
One seedpair per configuration is a small screening sample, not an independent
statistical confirmation or quantified native winrate. No confirmation seeds
were spent, no candidate promoted.

{
  "candidate_vs_original_60": {
    "W": 2,
    "L": 0,
    "D": 0,
    "unresolved": 0,
    "GF": 8,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 90.88333333332966,
    "circling_windows": 2,
    "goalward_contacts": 25,
    "backward_contacts": 15,
    "falls_seconds": 9.299999999999978
  },
  "candidate_vs_A_60": {
    "W": 2,
    "L": 0,
    "D": 0,
    "unresolved": 0,
    "GF": 6,
    "GA": 1,
    "flips": 0,
    "stationary_seconds": 35.95000000000016,
    "circling_windows": 5,
    "goalward_contacts": 20,
    "backward_contacts": 22,
    "falls_seconds": 6.73333333333332
  },
  "candidate_vs_geometric_60": {
    "W": 0,
    "L": 1,
    "D": 0,
    "unresolved": 1,
    "GF": 1,
    "GA": 3,
    "flips": 0,
    "stationary_seconds": 318.3000000000596,
    "circling_windows": 1,
    "goalward_contacts": 11,
    "backward_contacts": 9,
    "falls_seconds": 237.76666666665574
  },
  "geometric_vs_original_60": {
    "W": 2,
    "L": 0,
    "D": 0,
    "unresolved": 0,
    "GF": 4,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 40.98333333333254,
    "circling_windows": 1,
    "goalward_contacts": 26,
    "backward_contacts": 22,
    "falls_seconds": 9.149999999999977
  },
  "geometric_vs_A_60": {
    "W": 0,
    "L": 2,
    "D": 0,
    "unresolved": 0,
    "GF": 0,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 177.18333333332583,
    "circling_windows": 1,
    "goalward_contacts": 19,
    "backward_contacts": 21,
    "falls_seconds": 5.249999999999991
  },
  "candidate_vs_original_120": {
    "W": 1,
    "L": 1,
    "D": 0,
    "unresolved": 0,
    "GF": 2,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 138.80000000002542,
    "circling_windows": 2,
    "goalward_contacts": 18,
    "backward_contacts": 5,
    "falls_seconds": 15.366666666666557
  },
  "candidate_vs_A_120": {
    "W": 0,
    "L": 1,
    "D": 0,
    "unresolved": 1,
    "GF": 0,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 185.77500000001157,
    "circling_windows": 5,
    "goalward_contacts": 10,
    "backward_contacts": 12,
    "falls_seconds": 8.875000000000036
  },
  "candidate_vs_geometric_120": {
    "W": 0,
    "L": 0,
    "D": 0,
    "unresolved": 2,
    "GF": 2,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 37.95833333333246,
    "circling_windows": 1,
    "goalward_contacts": 8,
    "backward_contacts": 9,
    "falls_seconds": 27.93333333333256
  },
  "geometric_vs_original_120": {
    "W": 0,
    "L": 0,
    "D": 0,
    "unresolved": 2,
    "GF": 4,
    "GA": 4,
    "flips": 0,
    "stationary_seconds": 81.86666666667136,
    "circling_windows": 6,
    "goalward_contacts": 18,
    "backward_contacts": 17,
    "falls_seconds": 42.283333333332216
  },
  "geometric_vs_A_120": {
    "W": 2,
    "L": 0,
    "D": 0,
    "unresolved": 0,
    "GF": 5,
    "GA": 2,
    "flips": 0,
    "stationary_seconds": 42.033333333331626,
    "circling_windows": 5,
    "goalward_contacts": 16,
    "backward_contacts": 8,
    "falls_seconds": 21.783333333332976
  },
  "elapsed_seconds": 181.7342107190052
}

## Preserved and validated
Full eight-update checkpoints, Adam, RNG, demonstrations, used sources and
compact results retained. Exact original13D/26181 actor, finite optimizerstate,
strict reload and chronological vs single-callback inference parity passed.
Unmodified official validator passed on isolated diagnostic kit, workflow
compared to official upstream. Local nonlive SDKpreflight passed. These do
not prove actual Windows acceptance. Previously verified main and Geometric
GitHubActions remain preserved; no new Actions result claimed until checked.
No main push, merge, judgeaccess, organizercommunication or submission.
Native duplicateclock cause remains unresolved until the rawJSONL arrives.
User's actual60Hz evidence must not be attributed to120Hz simulation behavior.

Earlier interception-readout candidate was also rejected: sixteen supervised
updates improved label MSE but reduced independentcontacts18/32→4/32 and
increased circling3→10. It lost allfour60Hz matches againstGeometric.
The earlier failure's stationary increase was a steering-trajectory exposure
of a pre-existing reference blindspot: reference replay itself selectedNeutral
on566/570 failed-path Neutral states. This is not the established cause of
native duplicateclock errors. Those rawlogs remain unavailable.
