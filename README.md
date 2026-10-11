# V13 Geometric Control — restored playing bot

This restores the preserved Geometric V13 requested after Outcome-128's poor native gameplay. One bot, one unchanged trained checkpoint; no new training or runtime manoeuvres.

Bot name: **V13 Geometric Control**. Agent ID: `lakshya-agrawal254/v13-geometry-control`.
Model: `model/V13.pt`, SHA256 `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`.
Original 13 observations, exactly 26,181 parameters, GRU64, four modes and original steering decoder. No external pretrained weights, extra controls or inputs. Forecast-related weights remain zero: this preserved version does not use ball prediction despite receiving the original forecast features.

## Play on Windows

1. Extract the ZIP into a **new folder**. Keep the existing Outcome installation as rollback.
2. Run `SETUP.cmd`, then `CHECK.cmd`. Setup installs dependencies without training.
3. In RLBot v5, remove the old Outcome entry from the selected matchup and add this folder's `bot.toml`.
4. Select **V13 Geometric Control**, choose a separately installed opponent such as Original V13, and click **Start Match**.

The user previously found Geometric slightly better than Original in actual Rocket League, but still weak. This does not establish a formal native win rate or guarantee the repackaged copy's acceptance. Circling, overshooting and callback errors remain possible. Cloud verification is not an actual-game test.

The repository restoration preserves Outcome-128 at commit `738e875877bec6ce84965a607cc15af2341f9a87` and its preparation/research branches. The original V13 and all other checkpoints remain preserved. This package changes checkpoint/identity metadata; observation, GRU, decoder and callback behavior are preserved. See `docs/RESTORATION.md` and `docs/PARTICIPANT_RESULTS.md`.

This is a private playing-model restoration, not a tournament handoff. Organizer access and submission still require separate explicit authorization. No date is backdated to imply a pre-deadline submission.
