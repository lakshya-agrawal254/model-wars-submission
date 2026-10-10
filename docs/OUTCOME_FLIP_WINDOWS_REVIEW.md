# Windows learned-flip review — Geometric preferred, no submission authorized

Keep Geometric-preferred as the playing reference. The Outcome-128-diagnostic
folder is an optional diagnostic candidate: 128 updates did NOT teach deterministic
flips. Do not replace the preferred model with it. Original-V13 and Validated-A
are separate comparison bots. No Nexto/Necto assets are used.

1. Install Python3.12 x64 and RLBot v5, with actual Rocket League installed.
2. Extract the ZIP to a short local folder such as C:\ModelWarsReview (outside
   OneDrive is preferable for clean logs). Double-click SETUP.cmd and CHECK.cmd
   inside each bot folder you want to use. Dependencies are separately installed;
   packages do not contain a virtualenv or the Rocket League game.
3. Open RLBot v5, add Geometric-preferred\bot.toml and Original-V13\bot.toml.
   Choose a normal1v1 Soccar match, start, then swap teams for the return match.
   Use the same game settings and deterministic models. Do not alter callback
   rate, observation delta, decoder or hidden resets.
4. Only for optional diagnosis, add Outcome-128-diagnostic\bot.toml. It is NOT a
   recommended improvement. Test against Geometric with sides swapped, then
   original/A if desired. Record final scores, kickoff spawns, stopping, circling,
   missed contacts, falls and recovery. A Dodge label is not evidence of a flip:
   require real in-game movement/video and jump release/secondpress sequence.
5. For full callback logs, start RLBot from a PowerShell session with the variable
   $env:V13_LOG_EVERY_CALLBACK='1'. The bot process must inherit this environment;
   restart RLBot if needed. Logs are under each bot's logs folder and capped at
   16MiB; use short recorded tests. Mode transitions are logged without this flag.
   Logging hooks do not do extra inference or change controls. Errors still raise.
6. Run, from the bot folder:
   .venv\Scripts\python.exe docs\native_clock_audit.py logs
   .venv\Scripts\python.exe docs\native_stop_audit.py logs
   Preserve complete JSONL files before/during/after errors, bothteams, phase,
   timestamps/frame, observations, raw heads, hidden norms and submissions.
   Do not infer paused/active gameplay from a summary alone. The earlier135
   duplicateclock errors atT194.458328/frame23335 remain unclassified until the
   user's complete rawlogs are supplied.

Cloud tests are real RocketSim and SDK checks, not actual Windows/Rocket League.
Official competition rules/validator are preserved. This package is private
review material; nothing has been submitted, merged into main or shared with judges.

The 16-update phase refinement is unsafe (continuous Jump), rejected and NOT included as a playing bot. Do not replace model files manually. Curriculum-128 won 4/4 small 60Hz Geometric screening matches but learned zero flips; test it optionally, not as a proven improvement. Record both team sides, at least full matches and all kickoff spawns.
