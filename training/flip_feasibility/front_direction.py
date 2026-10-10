"""Actual second-press dodge direction, legal unchanged control mapping only."""
from proof import *
w=ForecastWorld(9900011,'A',0);c=w.state.cars[w.our];c.physics.position=np.array([0,0,17],np.float32);c.physics.euler_angles=np.zeros(3,np.float32);c.physics.linear_velocity=np.array([1000,0,0],np.float32);c.physics.angular_velocity=np.zeros(3,np.float32);c.wheels_with_contact=(True,True,True,True);w.state.ball.position=np.array([0,4000,93.15],np.float32);w.state=w.engine.set_state(w.state,{})
rows=[]
for stage,mode,count in [('prefix',1,24),('jump',2,12),('release',1,12),('dodge',3,6),('recover',1,180)]:
 for n in range(count):
  before=w.state.cars[w.our].physics.linear_velocity.copy();w.step(decode(mode,0.),decode(0,0.),ticks=1);raw=w.engine._cars[w.our].get_state();after=w.state.cars[w.our].physics.linear_velocity.copy()
  if stage=='dodge' or n==0:rows.append(dict(stage=stage,tick=n,mode=mode,controls=decode(mode,0.),has_flipped=bool(raw.has_flipped),flipping=bool(raw.is_flipping),velocity=after.tolist(),delta_velocity=(after-before).tolist()))
w.close();events=[r for r in rows if r['stage']=='dodge' and r['flipping']];assert events and max(r['delta_velocity'][0] for r in events)>200
out=ROOT/'runs/flip-front-direction-v1';assert not out.exists();out.mkdir();(out/'proof.json').write_text(json.dumps(dict(status='PASS',definition='Initial forward axis+x; actual flipped flag and positiveforward velocity impulse confirm front dodge',rows=rows),indent=2));print('PASS actual frontdodge direction; max+x impulse',max(r['delta_velocity'][0] for r in events))
