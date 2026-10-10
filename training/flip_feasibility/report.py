"""Compact reproducible results; no promotion based on supervised loss."""
from proof import *
learn=ROOT/'runs/flip-learning-v1';result=json.loads((learn/'result.json').read_text());physics=json.loads((ROOT/'runs/flip-feasibility-v1/physics-proof.json').read_text());matches=json.loads((ROOT/'runs/flip-screen-corrected-v1/summary.json').read_text())
text=f'''# V13 legal flip feasibility review — NOT SUBMITTED

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

{json.dumps(result,indent=2)}

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

{json.dumps(matches,indent=2)}

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
'''
(ROOT/'docs/FLIP_FEASIBILITY_REVIEW.md').write_text(text)
