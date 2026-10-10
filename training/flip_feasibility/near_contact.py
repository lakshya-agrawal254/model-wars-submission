"""Additional close-contact timing test; independent physical proof, NO optimization."""
from proof import *
out=ROOT/'runs/flip-nearcontact-v1';assert not out.exists();out.mkdir();rows=[]
for rate in [60,120]:
 for side in [0,1]:
  for onset in [.6,.9,1.1]:
   r,*_=trial(rate,side,'nearball','release',seed=9880011,onset_override=onset);r['first_jump_seconds']=onset;r['setup']='Flatground speed1000UU/s, ball2000UU ahead; independent timing sweep; training-only controls';r.pop('trace');rows.append(r)
(out/'records.json').write_text(json.dumps(rows,indent=2));print([(r['rate'],r['side'],r['first_jump_seconds'],r['first_contact'],len(r['flip_events']),r['landing_recovery_seconds']) for r in rows])
