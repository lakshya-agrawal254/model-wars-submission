"""Evaluation-only genuine RocketSim ball forecasts. NOT native RLBot prediction recordings."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training/memory_selective'))
from common import World as Base
class ForecastWorld(Base):
 def observe(self,aid):
  x=super().observe(aid)
  # Fresh forecast of the current real arena, shared by bothcars. Index240 is2seconds.
  key=id(self.state)
  if getattr(self,'forecast_key',None)!=key:
   predictions=self.engine._arena.get_ball_prediction(241,1);assert len(predictions)==241
   self.future_ball=np.asarray(predictions[240].pos.as_numpy(),dtype=np.float32);assert np.isfinite(self.future_ball).all();self.forecast_key=key
  p=self.state.cars[aid].physics;delta=self.future_ball-p.position;x[3:6]=[np.dot(delta,axis)/6000 for axis in [p.forward,p.right,p.up]];x[9]=1.;x[10]=1.;x[11]=0.
  assert np.isfinite(x).all();return x
