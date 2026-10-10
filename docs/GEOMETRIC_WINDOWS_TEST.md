# Isolated geometric candidate — native Windows test still required

The experimental model is `Lakshya V13 Geometry Candidate` (`lakshya-agrawal254/v13-geometry`), exact 13D/26,181-parameter V13. It has been fit once on independent geometric targets. Improved RocketSim contact is not proof of stronger match play. Do not replace the main submission without independent match evidence AND actual RLBot Windows acceptance.

Install Python 3.12, Rocket League and the official RLBot v5 launcher. In PowerShell:

```powershell
git clone --branch training/v13-geometric-supervised-20261011 --single-branch https://github.com/lakshya-agrawal254/model-wars-submission.git V13-geometry
Set-Location V13-geometry
powershell -ExecutionPolicy Bypass -File .\setup.ps1
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' verify
& '.\.venv\Scripts\python.exe' -I -u -B '.\src\preflight.py'
Set-Location ..
git clone --branch baseline/v13-original-83d92a6 --single-branch https://github.com/lakshya-agrawal254/model-wars-submission.git V13-original
Set-Location V13-original
powershell -ExecutionPolicy Bypass -File .\setup.ps1
Set-Location ..
git clone --branch main --single-branch https://github.com/lakshya-agrawal254/model-wars-submission.git V13-current
Set-Location V13-current
powershell -ExecutionPolicy Bypass -File .\setup.ps1
Set-Location ..
```

Alternatively extract the private candidate ZIP and run the same setup/verify/preflight commands in its `V13` folder. The existing private Windows main package remains the safe fallback.

Open the **official RLBot launcher shortcut**. Add each folder's `bot.toml`. Choose **Lakshya V13 Geometry Candidate** versus **V13**, standard 1v1 Soccar, 5 minutes, Start Match. Repeat with team sides swapped, then compare against **Lakshya V13 PPO** from `V13-current`. Do not manually start the Python bot to stand in for native acceptance.

In a separate PowerShell window, inspect actual native logs after the game has started:

```powershell
Set-Location V13-geometry
Get-Content (Get-ChildItem .\logs\*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1).FullName -Tail 20 -Wait
```

After playing both sides, Ctrl+C stops only the log viewer. Run:

```powershell
& '.\.venv\Scripts\python.exe' '.\training\geometric\check_native_logs.py' --logs '.\logs' --checkpoint 55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc
```

Inspect `callback_error`, `ball_prediction_received`, `processed_packet`, `controller_submitted`, and finite `observation13`. Valid prediction fields must reflect actual native delivery. The candidate's six forecast input columns have zero trained weights, so it does not use forecasts; the original 13D runtime still receives and checks them. A native callback/prediction failure remains a failure; no heuristic fallback or ignored exception has been introduced.

Record Windows, RLBot, Python and game versions; configuration; W/L/D; goals for/against; both team sides; first contact; circling/idle episodes; replay/kickoff behavior; releases; and errors. Jump/Dodge remain original modes; boost, yaw, roll and handbrake stay disabled. Hidden state persists through goals/replays and resets only for a new bot instance. Logs alone do not establish gameplay quality. Report native results before promotion. Cloud RocketSim and ControllerState serialization are **not** an actual Rocket League match test.

Submission main stays at `c151310b5c36996aeb1fe192be5eea7cc328e846`, with its passing validator. Judge access is still deferred at the user's request. Deadline: 11 October 2026 07:00 IST; final tested push target 06:30 IST.
