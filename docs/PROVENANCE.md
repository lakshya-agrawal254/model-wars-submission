# Provenance and attribution

Restored Geometric V13, trained within the new competition project using independently generated geometric steering/mode labels on real RocketSim states. The actor inherited the project's own legal scratch-A checkpoint and received 176 supervised Adam updates. No Nexto/Necto assets, external pretrained outputs or old RocketSim-project checkpoints were used. Exact original training configuration and source/data hashes are in training_provenance.json. Restoration performs zero optimizer updates and preserves the actor file byte-for-byte.

Official framework: Model-Wars/Model-Wars-Base-Repository, preserved source snapshot 5d8801361766cae4584245e9188edc64fd1d79c3; original V13 attribution shryssssss. Participant training lakshya-agrawal254. RLBot/python-example attribution revision fd061f457bf19175b4a9b3b3d7811a987044c64d is retained in LICENSE_RLBOT.txt and third_party/LICENSE_RLBOT_PYTHON_EXAMPLE.txt. No blanket license is asserted for participant weights; this package is private.

Only checkpoint pin, bot identity and documentation/sealing metadata change relative to the preserved Outcome playing package. The model architecture, observation encoder, action decoder, prediction selector and native hidden-state contract remain unchanged. Required third-party attribution is retained.
