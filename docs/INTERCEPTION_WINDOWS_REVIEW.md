# Private Windows review: keep Geometric as the playing reference

Folders: Geometric-preferred, Original-V13, Validated-A and Rejected-readout-diagnostic. The new readout candidate failed independent movement screening and is NOT proposed for replacement. Geometric weights remain unchanged. Main and all historical checkpoints are preserved. This is private testing, not submission.

In each folder run SETUP.cmd, then CHECK.cmd. Alternatively:

    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    .\.venv\Scripts\python.exe -I .\src\preflight.py

Use official RLBot v5 Launcher and actual Rocket League. Add each bot.toml. Select standard five-minute 1v1 Soccar, normal physics. Test Geometric-preferred vs Original-V13 on Blue and Orange, then Geometric vs Validated-A. Save full scores, replay/video and both bots' logs. Rejected-readout-diagnostic is optional diagnosis only; do not treat it as stronger. Actual Windows execution of the new package is UNTESTED.

Do not alter callback timing, observation12, the decoder or hidden resets. Diagnostic logging adds no inference. It records original logits/controls/observations/hidden norm on mode/stationary transitions and every12callbacks. Errors include phase, phase code, remaining time, game speed and last accepted timestamp; exceptions still propagate. Logs cap at16MiB. Logging adds I/O overhead.

Analyze real logs:

    .\.venv\Scripts\python.exe .\tools\native_stop_audit.py .\logs
    .\.venv\Scripts\python.exe .\tools\native_clock_audit.py .\logs

For about30seconds of every-callback recording, close RLBot and relaunch it from PowerShell after setting:

    $env:V13_LOG_EVERY_CALLBACK='1'

Then remove that setting:

    Remove-Item Env:V13_LOG_EVERY_CALLBACK

Raw Geometric logs aroundT194.4583282470703/frame23335 remain awaited. The reported native session was approximately60Hz; do not blame the separate120Hz simulation problem. Pause-versus-active classification remains unresolved. Older error entries omit phase; new telemetry exposes it without suppressing errors. Logged samples are not win rates or complete stationary-time measurements.

This campaign's match screen uses genuine RocketSim two-second ball forecasts in the original13D schema for both agents. These are simulation forecasts, not native RLBot recordings. Scores are not directly comparable with older zero-forecast screens. No external pretrained model or teacher was used. Do not replace main, merge, grant judge access, publish or submit without explicit approval. Deadline11October2026,07:00IST.
