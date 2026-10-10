"""Independent geometry from original V13 fields only. Training utility, never runtime."""
import numpy as np

def targets(x):
 x=np.asarray(x,dtype=np.float32);assert x.shape[-1]==13 and np.isfinite(x).all()
 present=x[...,8]>.5;angle=np.arctan2(x[...,1],x[...,0]);steer=np.where(present,np.clip(2*angle,-1,1),0).astype(np.float32)
 return np.where(present,1,0).astype(np.int64),steer
