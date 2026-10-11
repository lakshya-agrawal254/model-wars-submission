"""Native 3D feature reflection proof; no learner or runtime changes."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from observation_contract import basis,build
def main():
 rng=np.random.default_rng(10651011);M=np.array([-1.,1.,1.]);worst=0.
 for _ in range(200):
  p,y,r=rng.uniform(-3.14,3.14,3);B=np.asarray(basis([float(p),float(y),float(r)]));C=np.asarray(basis([float(p),float(np.pi-y),float(-r)]));expected=B*M;expected[1]*=-1
  assert np.max(np.abs(C-expected))<1e-12
  pos=rng.uniform(-3000,3000,3);vel=rng.uniform(-1200,1200,3);ball=rng.uniform(-4000,4000,3);future=rng.uniform(-4000,4000,3)
  for present,valid in [(True,True),(True,False),(False,False)]:
   context=dict(elapsed=5.,previous_elapsed=5.-1/60,previous_mode=1,previous_steer=.3,prediction_valid=valid,prediction_position=future.tolist(),selected_time=7.,first_time=5.)
   raw=dict(position=pos.tolist(),velocity=vel.tolist(),basis=B.tolist(),ball_present=present,ball_position=ball.tolist())
   x=np.asarray(build(raw,context))[:13];other=dict(raw,position=(pos*M).tolist(),velocity=(vel*M).tolist(),basis=C.tolist(),ball_position=(ball*M).tolist());ctx=dict(context,prediction_position=(future*M).tolist(),previous_steer=-.3)
   yy=np.asarray(build(other,ctx))[:13];target=x.copy();target[[1,4]]*=-1;worst=max(worst,float(np.max(np.abs(yy-target))))
   assert np.max(np.abs(yy-target))<1e-6
 report=dict(random_orientations=200,availability_cases_per_orientation=3,max_feature_error=worst,native_basis_mirror='pitch unchanged, yaw pi-yaw, roll -roll; globalx positions/polar velocities reflected, angular velocity axial -M*omega',training_updates=0)
 out=ROOT/'runs/mirror-recording-audit-v1/three-dimensional-basis.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
