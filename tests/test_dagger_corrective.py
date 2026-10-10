import sys,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training/dagger_corrective'))
from campaign import labels,load,GEO,COLS
from runtime import decode

def test_corrective_direction_and_selectivity():
 x=np.zeros((5,13),np.float32);x[:,8]=1;x[:,12]=1;x[:,6]=.3;x[:,0]=.3;x[0,1]=.1;x[1,1]=-.1;x[2,0]=-.08;x[2,1]=.02;x[2,6]=.08;x[2,7]=.8;x[3,8]=0;x[4,2]=.5
 m,s,mask,coast=labels(x,np.array([1,1,1,1,2]),np.zeros(5));assert s[0]>0 and s[1]<0;assert m[2]==0 and coast[2];assert m[3]==0;assert m[4]==2 and not mask[4]

def test_recurrence_logging_and_forecast_invariance():
 torch.set_num_threads(1);p=load(GEO);q=load(GEO);torch.manual_seed(51);x=torch.randn(37,13);captures=[];p.head.register_forward_hook(lambda module,args,output:captures.append(output.detach().tolist()));h=None;ls=[]
 with torch.inference_mode():
  y=p(x);z=q(x)
  for a,b in zip(y,z):assert torch.equal(a,b)
  for row in x:
   l,s,h=q(row[None],h);ls.append(l)
  assert torch.allclose(y[0],torch.cat(ls),atol=2e-6) and torch.allclose(y[2],h,atol=2e-6)
  stress=x.clone();stress[:,COLS]=100
  for a,b in zip(y,p(stress)):assert torch.equal(a,b)
 assert sum(v.numel() for v in p.parameters())==26181

def test_official_sources_and_decoder_unchanged():
 official=ROOT.parent/'official'
 for name in ['src/policy.py','src/observation_contract.py','tools/validate_submission.py','.github/workflows/validate.yml']:
  assert (ROOT/name).read_bytes()==(official/name).read_bytes()
 assert [decode(i,.4) for i in range(4)]==[[0.,0.,0.,0.,0.,False,False,False],[1.,.4,0.,0.,0.,False,False,False],[0.,0.,0.,0.,0.,True,False,False],[0.,0.,-1.,0.,0.,True,False,False]]
