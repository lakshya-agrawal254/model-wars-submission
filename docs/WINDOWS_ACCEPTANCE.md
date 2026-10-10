# Required local Rocket League acceptance (NOT TESTED in cloud)

1. Install Python 3.12, Rocket League and the official RLBot v5 launcher. Extract the private Windows ZIP into a writable folder. Open PowerShell in its `V13` directory and run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   & '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' verify
   & '.\.venv\Scripts\python.exe' -I -u -B '.\src\preflight.py'
   ```

2. Open the official **launcher shortcut**, not the internal GUI executable. Add File -> this extracted root `bot.toml`. Select **Lakshya V13 PPO**, a compatible opponent and 1v1 Soccar -> Start Match. Do not manually launch the Python bot as a substitute for this test.
3. For the untouched starter comparison, separately clone the private repository at `83d92a69c80930652c4d26be8db9f7ebea062e38`, run its setup and add its `bot.toml` as **V13**. Do not overwrite either installation. Community experts may be optional local evaluation opponents only; no expert outputs are used in this training or package.
4. Play full regulation matches on both sides, using new seeds/configurations. Record actual scores, goal prevention, touches, first contact, stationary/circling episodes, recovery, latency and callback errors. Inspect bounded local JSONL logs for actual prediction receipt, finite observations, successful controller submission and no clock exceptions. Preserve logs privately. Native real prediction-valid behavior has not been trained/evaluated here.
5. Check Chase throttle/steer, mode-controlled jump/front dodge and releases. Boost, yaw, roll and handbrake must remain disabled by the original tournament controls. Check kickoff/replay/countdown memory and team changes; hidden state must reset at a new bot instance, not every goal.
6. Repeat from a different clean extraction to check relocation. Record Windows/Python/RLBot/game versions and package SHA. Do not mark `live_acceptance` successful without actual evidence. Add truthful native results and reseal with `tools/kit.py seal` before final tournament submission.

Cloud preflight validates shapes, config and native ControllerState serialization. RocketSim results are not an actual-game test. Organizer confirmation is still required for runtime/resource limits and judge account/access (the official README specifies pushing to main). The user confirmed the deadline as **11 October 2026, 07:00 IST (01:30 UTC)**. Do not use old Nexto-derived assets or assume architecture exceptions. The organizer's validator has no guarantee of gameplay quality.
