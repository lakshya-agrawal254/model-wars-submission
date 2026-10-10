# Correction to the interception full-match observation-domain claim

The previous interception screen's metadata and report described enabled physical
ball forecasts. That description was incorrect for its full matches. Its hook
assigned World in match_impl.__globals__, but match_impl is the PyTorch
inference_mode decorator wrapper. The underlying function retained common.World,
whose observation builder supplies zeros for the six prediction fields.

The old 40 match records and archived exports are preserved unchanged. Those
scores describe matched zero-prediction-field RocketSim evaluations; they must
not be represented as native-schema forecast evaluations or native RLBot results.
The former numerical outcomes remain recorded, but interpretation is limited.
The interception actor/data verification and direct ForecastWorld checks did not
catch the wrapper injection error. No model was promoted from those outcomes.

This issue was discovered while adding actual-dodge and landing tracking for
flip screening. The initial flip screen also hit the wrong wrapper namespace;
its tracking failed at completion and no successful result set was preserved.
That failed job was stopped; no training or checkpoint creation occurred in it.
The corrected flip screen installs the tracked forecast-enabled world into
match_impl.__wrapped__.__globals__. A direct check confirmed the actual class,
finite forecast input, tracker identity and unchanged decoder before the fresh
screen. New corrected results are under runs/flip-screen-corrected-v1.

Independent inference/physics trials explicitly constructed ForecastWorld;
they are unaffected. The eight-update learning proof uses direct ForecastWorld
and no duplicate learner was launched. Missing native raw JSONL remains a
separate unresolved issue. No simulator result constitutes Windows acceptance.
