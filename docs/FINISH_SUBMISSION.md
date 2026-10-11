# Finish Outcome-128 submission — approval required

Provisional preparation branch: `preparation/outcome-128-provisional-20261011`.
Private repository: `lakshya-agrawal254/model-wars-submission`.
Deadline supplied by participant:11October2026,07:00IST (01:30UTC).
Training is stopped; the final challenger finished at06:20:09IST and failed qualification. Retain this exact Outcome-128 checkpoint. Do not run any learner.

## Permitted preparation and Windows acceptance

1. Download this branch or the single-bot provisional review ZIP. Verify model/V13.pt SHA256:
   `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.
2. Windows: install Python3.12x64 and official RLBotv5/RocketLeague. Extract to a fresh directory; run SETUP.cmd and CHECK.cmd there. Register only this root bot.toml (V13 Outcome-128 Provisional, agent ID lakshya-agrawal254/v13-outcome-128).
3. Complete standard1v1 full matches as Blue and Orange against separately installed OriginalV13/Geometric. Capture actual scores and native callback/prediction/control errors. Do not describe RocketSim or non-live preflight as native acceptance. Duplicate/backward callback clocks must be investigated, not suppressed.
4. Update docs/PARTICIPANT_RESULTS.md with actual evidence. On the preparation branch only:

```powershell
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' seal --provenance '.\docs\training_provenance.json'
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' verify
& '.\.venv\Scripts\python.exe' -I -u -B '.\src\preflight.py'
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\validate_submission.py' .
git add docs/PARTICIPANT_RESULTS.md submission.json candidate_manifest.json participant_manifest.json participant_manifest.sha256 src/runtime.py
git commit -m "Record actual Outcome-128 Windows acceptance"
git push origin preparation/outcome-128-provisional-20261011
```

Only record acceptance if it actually passes. Do not run kit init again; identity is already initialized. Do not replace the model file.

5. GitHub: repository → Actions → Validate Bot Submission → Run workflow → choose preparation/outcome-128-provisional-20261011 → Run workflow. Confirm success for its exact latest SHA. The unchanged official workflow does not automatically run on preparation-branch pushes. If the cloud dispatch is denied, the repository owner must use the UI or authorized Actions credentials. Do not modify the validator/workflow or push main to work around that denial.
6. Review the exact branch diff, checkpoint hash, sealed manifests, results, CI run and remaining weaknesses. Obtain the user's explicit approval for main replacement/submission and organizer access. Missing time or an approaching deadline is not approval.

## ONLY AFTER explicit user approval

If Windows acceptance and CI are satisfactory and main still points to the preserved commit, the following fast-forward preserves Git history. If it refuses, stop for review; do not force-push or overwrite concurrent work.

```powershell
git fetch origin
git switch main
git merge --ff-only origin/preparation/outcome-128-provisional-20261011
git push origin main
```

Watch the official main-commit Actions run and confirm success. The original validated main commit is c151310b5c36996aeb1fe192be5eea7cc328e846; previous model/checkpoints remain in development branches, Git history and verified private backups. This clean inference tree omits training archives; it does not erase those preserved copies.

After separately authorized organizer sharing: confirm the designated individual judge username, grant only the instructed Read access to this private repository and follow the organizer's documented handoff channel. The judge account is still unresolved; do not guess or invite an arbitrary similarly named account. No additional submission form, email address or automatic handoff is invented here.

If the organizer specifically requests a ZIP, create its parent directory, then use tools/kit.py archive --output C:\Submissions\Approved-Outcome128.zip after resealing. Share only the approved single-bot artifact and its actual SHA256. Do not send the multi-model review or complete private training backup.

If approval or native acceptance is missing at07:00IST, keep the entry provisional and report the blocker. Do not auto-submit.
