# V13 Outcome-128 — provisional private entry

One bot, one inference checkpoint. This preparation branch is for review and Windows acceptance. It is NOT submitted. No main replacement, merge, judge invitation or organizer handoff is authorized.

- Bot: `V13 Outcome-128 Provisional`
- Agent ID: `lakshya-agrawal254/v13-outcome-128`
- Checkpoint: `model/V13.pt`
- SHA256: `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`
- Original 13D inputs, encoder13→64/ReLU, one-layer GRU64, head64→5: exactly26,181 parameters. Four original modes and continuous steering. No added runtime features, controls, scripts or external pretrained models.
- Neural weights copied byte-for-byte from the verified sealed Outcome-128 package. Training is stopped. This bundle contains no trainer, other bot, alternative checkpoint, optimizer, recording or environment.

## Windows acceptance — still required

1. Install Python3.12 x64, RocketLeague and official RLBotv5. Extract the entire package to a fresh folder.
2. Run `SETUP.cmd`, then `CHECK.cmd` from that folder. Setup installs dependencies; it never trains.
3. In RLBotv5 add this folder's `bot.toml`. Select a separately installed OriginalV13 or Geometric opponent. Do not register the old Outcome review ID by mistake.
4. Run standard1v1 full matches on both Blue and Orange. Record final scores, prediction/callback errors and `logs/` evidence. No fixed callback timing override is permitted.
5. Update `docs/PARTICIPANT_RESULTS.md` with actual evidence. Reseal after documentation changes and revalidate. A cloud preflight or RocketSim result does not establish Windows gameplay acceptance.

See `docs/FINISH_SUBMISSION.md` for the explicit-approval gate and exact remaining commands. Deadline:11October2026,07:00IST. A deadline never authorizes submission.

## Evidence and limits

Outcome128 remains weak: circling, unproductive contacts, collision pins and120Hz failures remain. Complete simulator results and native acceptance status are in `docs/PARTICIPANT_RESULTS.md`. No claim of tournament superiority or native acceptance is made.

Training provenance is in `docs/training_provenance.json` and `submission.json`. Model and package hashes are enforced by the runtime/candidate/participant manifests. Official validation tools and workflow are unchanged. Third-party attribution is in `docs/PROVENANCE.md` and retained license files.

The official workflow runs on main pushes, PRs targeting main and manual dispatch. A push to this preparation branch alone does not trigger it. Do not create a PR or alter main merely to bypass missing dispatch permissions.
