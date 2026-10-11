# Provenance and attribution

This is Outcome-128, derived from the preserved Geometric V13 through128 supervised Adam updates on independently generated realRocketSim legal manoeuvres and the project's own trajectories. No external pretrained bot weights, outputs or labels were used. The exact source/data hashes, objective/configuration and presentation count are retained in training_provenance.json. This preparation changed no neural weight.

The official Model-Wars/Model-Wars-Base-Repository supplies the original13D/GRU/fourmode framework and starter. The original pretrained starter is the organizer-provided baseline, not an external teacher. Official source snapshot:5d8801361766cae4584245e9188edc64fd1d79c3. Original V13 framework attribution:shryssssss. Participant training and preparation:lakshya-agrawal254.

RLBot/python-example revisionfd061f457bf19175b4a9b3b3d7811a987044c64d is the upstream attribution retained by the official starter for example-derived material. LICENSE_RLBOT.txt and third_party/LICENSE_RLBOT_PYTHON_EXAMPLE.txt are preserved. The optional runnable PythonExample reference has been omitted so the preparation package contains exactlyone bot; runtime code has no reference-folder dependency. Dependencies retain their own licenses. No blanket license is asserted for participant weights.

The sealed Outcome reference remains unchanged outside this preparation worktree. Only identity literals in bot.py/preflight.py, metadata and documentation changed in this copy. policy.py, observation_contract.py, prediction.py and runtime.py are byte-identical to the sealed source; the model SHA/runtimePIN is unchanged. No observation, inference, decoder, callback timing or GRU reset behavior changed.
