# Final V13 candidate review — not submitted

No new training was started. Preserve the validated main commit, all actor and optimizer snapshots, and all three separate review kits. Actual Rocket League acceptance is still required for Outcome-128 and Ground-64; cloud RocketSim and SDK checks cannot provide it.

## Local Windows test

1. Install Rocket League, RLBot v5 and Python 3.12 x64. Extract the review ZIP outside OneDrive, for example `C:\V13FinalReview`.
2. In each desired bot folder run `SETUP.cmd`, then `CHECK.cmd`. Start with `Outcome-128-reference`, `Geometric-preferred`, and `Original-V13`; `Ground-64-diagnostic` is optional because it has repeated long stationary episodes. The folder names identify preserved roles, not a new promotion.
3. In RLBot's launcher add each folder's root `bot.toml`. Run default 1v1 Soccar, then swap teams, keeping all conditions identical. Test Outcome against Geometric and original V13, and Geometric against original V13. Record complete scores and both bots' logs. Do not change callback timing, observation inputs, decoder or hidden resets.
4. Check prediction/callback errors, distant-ball Neutral, circling, collisions, useful contacts, scoring and recovery. Actual score evidence takes priority over short contact trials. A GUI start alone does not establish competitive improvement.
5. From each bot folder run:

```powershell
& '.\.venv\Scripts\python.exe' 'docs\native_clock_audit.py' 'logs'
& '.\.venv\Scripts\python.exe' 'docs\native_stop_audit.py' 'logs'
```

For a short detailed trace, start RLBot from PowerShell after `$env:V13_LOG_EVERY_CALLBACK='1'`, ensuring bot children inherit it. The 16 MiB log cap remains; use roughly 60-second detailed captures and default logging for full matches. Save raw JSONL and video around failures. Logs now include error-phase context but errors still raise. Do not force 60 Hz or suppress duplicate-clock errors.

## Approval-dependent submission steps

These are instructions for after explicit user approval; none was executed. Deadline: 11 October 2026 07:00 IST / 01:30 UTC. Training cutoff: 05:15 IST / 23:45 UTC on 10 October. Missing approval remains a blocker even at the deadline.

1. Select a separately sealed candidate only after reviewing Windows scores and callback behavior. Retain main and the other checkpoints. The three kits already have distinct bot names/agent IDs and their own model, runtime SHA pin, `submission.json`, provenance, participant report and integrity manifests.
2. Prepare the approved kit as the repository contents on a private preparation branch. Keep its relative paths and root `bot.toml`. Do not copy weights alone or retain another candidate's runtime pin. Record actual Windows results in `docs/PARTICIPANT_RESULTS.md`, including native prediction reception and remaining errors. Do not describe a simulator run as a GUI test.
3. Retain the chosen kit's already initialized unique identity. The official `init` command is for a fresh uninitialized template and refuses these initialized kits; do not run it again here. Seal and check from the approved kit's own directory:

```powershell
& '.\.venv\Scripts\python.exe' -c "import json,pathlib; s=json.loads(pathlib.Path('submission.json').read_text()); pathlib.Path('docs/review_training_provenance.json').write_text(json.dumps(s['training_provenance'],indent=2))"
& '.\.venv\Scripts\python.exe' -I -u -B 'tools\kit.py' seal --checkpoint 'model\V13.pt' --provenance 'docs\review_training_provenance.json'
& '.\.venv\Scripts\python.exe' -I -u -B 'tools\kit.py' verify
& '.\.venv\Scripts\python.exe' -I -u -B 'src\preflight.py'
& '.\.venv\Scripts\python.exe' 'tools\validate_submission.py' '.'
```

The first command extracts the chosen kit's existing, documented provenance from `submission.json`; it does not invent new training results. Verify that it belongs to the chosen candidate. Reseal after documenting new evidence. Run the untouched official validator, not a modified check.
4. Run GitHub Actions on that preparation branch using the Actions UI / Run workflow, and inspect its result. This session's workflow-dispatch API permission was previously Forbidden; no new Actions pass is claimed. Experiment-branch pushes do not automatically trigger the preserved workflow, which filters push/PR to main. Do not push main just to obtain a check.
5. Only when explicitly approved, merge/push the approved prepared contents to main and require the official Actions result to pass. Preserve the previous passing commit. Obtain the designated judge account and grant only the instructed read access; user previously deferred this identity. No organizer access has been granted.
6. If the organizer requests a ZIP, use the official archive command, which creates a `V13/` wrapper. Create its destination parent first, verify/extract/install the archive in a fresh directory, and share its SHA256 separately. Do not send the multi-candidate review ZIP or optimizer/training backups as the submission.

```powershell
& '.\.venv\Scripts\python.exe' -I -u -B 'tools\kit.py' archive --output 'C:\Submissions\ApprovedV13.zip'
```

The official README describes private-repository submission through main and read-only judge access; it does not supply a separate registration form in these materials. Confirm the judge identity and any event-specific organizer handoff rather than inventing one.

## Limits

Exactly 13 inputs, original 26,181-parameter GRU actor, eight tensor names, four-mode decoder and native hidden/callback contract remain unchanged. No external pretrained weights, model outputs, opponent inputs, runtime motion/flip scripts or new controls are present. The decoder cannot reverse, brake, handbrake or boost; Jump/Dodge sacrifice throttle and steering while active. That constrained support and missing opponent/goal-position inputs limit recoveries and tactical scoring. Legal architecture checks are not proof of native competitive strength.
