# Windows review: compare policies and verify callback timing

No model has been promoted or submitted. Main and every prior checkpoint are preserved. The candidate failed its RocketSim match screen and must not replace main. The geometric control is the strongest surviving self-trained mechanical reference, not a verified better tournament bot. Your observation that validated A was worse than original V13 is recorded as qualitative real-game evidence.

The Windows ZIP contains three independent folders:
- `Selective-review`: Lakshya V13 Selective Review, actor SHA1177a04f30a3fc7ad10a5b51b491700a249d49a6dd08ebf38e61638a57a8faf4.
- `Geometric-control`: V13 Geometry Control, actor SHA55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc.
- `Original-V13`: unchanged original V13 starter weights SHA579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7; evaluation control, not an eligible trained submission.

Install Python3.12, Rocket League and the official RLBot v5 Launcher. Extract the ZIP into a writable local folder. For each folder, open PowerShell there and run:

    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    .\.venv\Scripts\python.exe tools\kit.py verify
    .\.venv\Scripts\python.exe src\preflight.py

These commands load/check models and configurations; they do NOT prove native game execution. In the official RLBot launcher use Add File to register each folder's `bot.toml`. Select 1v1 Soccar, standard arena/physics, five-minute regulation, normal mutators and no human intervention.

First play Selective Review vs Original V13 and Geometry Control vs Original V13, one complete match on each team side. Keep the original validated A installation separate. Record actual scores, own-goal-like touches, defensive retreats, contact times, kickoff positions and errors. Do not just test launch or declare wins based on contact counts. Both self-trained models made0/10 kickoff contacts at120Hz in the controlled RocketSim probe, versus10/10 at60Hz on identical starts. Ground contact succeeded4/4 at both rates. This is a serious transfer risk, not proof of native failure. **Do not force a new tick skip or change observation12 to conceal it.** Record the real runtime callback rate and actual native kickoff behavior.

The kits include `tools/native_log_audit.py`. After actual full matches on both sides, from each kit run:

    .\.venv\Scripts\python.exe tools\native_log_audit.py --logs logs --checkpoint CHECKPOINT_SHA_ABOVE

Use the exact hash for that folder. This reads only actual JSONL logs, checks delivered predictions, valid forecast observations, control submissions, both teams and callback errors. It reports observation12 median and estimated callbacks per game-time second. Packet logging is periodic; paused/countdown/replay phases can distort estimates. Logs cannot establish scores or engine acknowledgement. Native forecasts must remain real SDK deliveries; no synthetic prediction fallback.

For live error inspection:

    Get-Content (Get-ChildItem logs\*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1).FullName -Tail 20 -Wait

If callbacks and controls pass, record at least ten full matches for each new/control policy against original V13 (five per side), plus ten Selective Review vs Geometry Control (five per side). Do not tune on these results or claim superiority from a tiny sample. Keep the blank `native_results.csv` beside logs and fill actual observations only. Send private scores/logs for review; no organizer access, merge, checkpoint replacement or submission without explicit user approval.

Cloud RocketSim/preflight/validator tests are not native RLBot acceptance. Native execution of these new kits remains NOT TESTED. Deadline11October2026 07:00IST remains in effect, but never authorizes automatic submission.
