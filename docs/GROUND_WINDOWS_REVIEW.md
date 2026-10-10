# Windows comparison — private review, not submission

Geometric remains preferred. Ground-64 is not a verified competitive improvement and has a known car-pinning failure. Outcome-128 remains a reference. No model is automatically replaced.

1. Install actual Rocket League, RLBot v5 and Python 3.12 x64. Extract to `C:\V13GroundReview` (outside OneDrive preferred).
2. Run `SETUP.cmd`, then `CHECK.cmd` in Geometric-preferred, Outcome-128-reference and Ground-64-diagnostic. Original-V13 and Validated-A are optional comparisons; install only the folders you need. Dependencies and the game are installed separately.
3. In RLBot v5, add each folder's `bot.toml`. Test Ground-64 versus Outcome-128 in default 1v1 Soccar, then swap teams. Repeat against Geometric and original V13. Keep game settings identical. Do not change callback timing, observations, decoder or hidden resets.
4. Record final scores, video, all kickoff positions, side/behind-ball turns, stops, bumps and recovery. Watch for Ground-64 being pinned against another car while Chase/throttle remain active. More contacts alone are not acceptance.
5. JSONL logs are in each bot's `logs` folder. Default logging records periodic packets, action/stationary transitions and error phase/last accepted timestamp. Errors still raise. For short, full-callback captures, start RLBot from PowerShell after setting `$env:V13_LOG_EVERY_CALLBACK='1'`; ensure bot processes inherit it. Restart RLBot if needed. Files are capped at 16 MiB: use 60-second detailed captures and default logging for full matches.
6. From each bot folder run:

   `.venv\Scripts\python.exe docs\native_clock_audit.py logs`

   `.venv\Scripts\python.exe docs\native_stop_audit.py logs`

   Preserve both teams' complete logs and video, including before/during/after errors. The older T=194.4583 error records lack phase; we have not classified that event as paused or active gameplay.
7. Report scores and callback failures before selecting a model. Cloud RocketSim and SDK checks are not actual Rocket League acceptance. The experimental Jump pulse is not included in inference. Main replacement, merging, organizer access and submission still require explicit user approval.
