"""Independent training-only curvature targets; existing13Dactor+history only at runtime."""
import numpy as np

def target(position,velocity,forward,right,ball_position,ball_velocity,speed,table):
 # Linear extrapolation is only a label calculation, never a native forecast feature.
 distance=float(np.linalg.norm(ball_position-position));tau=float(np.clip(distance/max(speed+300,300),.12,.8));aim=ball_position+ball_velocity*tau-position
 a=float(np.arctan2(np.dot(aim,right),np.dot(aim,forward)));r=max(250,float(np.linalg.norm(aim[:2])))
 # Pure pursuit requires a forward lookahead. Behind-target cases use a maximal feasible turn.
 if abs(a)>np.pi/2:return float(np.sign(a)),tau
 curvature=2*np.sin(a)/r;idx=np.argmin(np.abs(np.asarray(table['speeds'])-speed));values=np.asarray(table['curvatures'][idx]);steers=np.asarray(table['steers']);order=np.argsort(values)
 return float(np.sign(curvature)*np.interp(abs(curvature),values[order],steers[order])),tau
