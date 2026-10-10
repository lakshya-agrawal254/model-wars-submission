# Private V13 corrective review — not a tournament submission

The original Geometric V13 remains the selected self-trained playing reference. The corrective model is an early-stopped, unpromoted diagnostic candidate. Main and every earlier checkpoint are preserved. User approval is mandatory before model replacement, merging, sharing with organizers or submission.

## Install and play on Windows

Extract the ZIP somewhere outside OneDrive if possible. It contains separate Corrective-review, Geometric-control, Validated-A and Original-V13 folders. In each folder run SETUP.cmd, then CHECK.cmd. Alternatively, in PowerShell:

    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    .\.venv\Scripts\python.exe -I .\src\preflight.py

Install/use the official RLBot v5 Launcher and Rocket League. Add each folder's bot.toml to the launcher. Do not use Nexto, custom decoders or callback-rate patches for this acceptance comparison. Select normal five-minute 1v1 Soccar, standard physics, same map/settings. First play Geometric-control vs Original-V13, then Corrective-review vs Geometric-control, then Corrective-review vs Original-V13 and Validated-A. Swap teams for each pairing. Preserve full final scores, replays/video, both bots' logs and callback error console output. No cloud result below is an actual Rocket League test.

## Measure stopping instead of guessing

All review folders include diagnostic-only logging: every12callbacks, mode transitions and stationary transitions; no added model inference, control overrides or hidden resets. Entries contain mode, four logits+steering raw head, actual steering/throttle/jump/boost, observation13 including ball presence/prediction validity/delta, game time/frame/phase, velocity, hidden norm and errors. Logs cap at16MiB per bot process; restart the bot for a fresh diagnostic recording if the cap is reached. Logging adds I/O overhead: compare against the uninstrumented preserved geometric kit if callbacks slow.

To capture every callback during a short stop reproduction, launch RLBot from the same PowerShell session after setting:

    $env:V13_LOG_EVERY_CALLBACK='1'

Close the launcher first so the new process inherits the variable. Record only about30seconds, then remove it:

    Remove-Item Env:V13_LOG_EVERY_CALLBACK

Summarize that folder's real logs:

    .\.venv\Scripts\python.exe .\tools\native_stop_audit.py .\logs

The summary's sample counts are NOT time fractions or win rates. At a stop inspect raw_head: does Neutral really win? Check ball_present column8, prediction_valid9, actual dt12/60 and callback_error. If Chase remains selected with throttle1, the stop cannot be attributed to Neutral on that recorded callback. Inspect collision/replay/airborne state and packet continuity. Hidden norms are correlations, not proof of a GRU defect. Preserve raw JSONL and score CSV for assessment; do not edit observations or native callback timing.

The key simulation sensitivity is the callback-delta feature, not a proven native bug. Never force dt=1 at runtime: this would change the official observation contract. The original 13D/26181 actor, four-action decoder and continuous GRU behavior remain unchanged.

## Acceptance status

Native qualitative evidence: user reports Geometric slightly better than original, but weak. No quantified native match result is claimed. New candidate's Windows execution is UNTESTED. Callback/prediction acceptance, real-game scores and stopping causes must be established from your Windows recordings before reconsidering it. The candidate was stopped when independent stationary-time diagnostics worsened; it is not proposed as a replacement.
