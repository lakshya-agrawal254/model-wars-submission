from world import *
from search import process
import json
from concurrent.futures import ProcessPoolExecutor
out=ROOT/'runs/outcome-flip-negatives-v1';assert not out.exists();out.mkdir();rows=[]
for side in [0,1]:
 for i,kw in enumerate([dict(speed=1000,gap=220,ball_y=2000),dict(speed=1400,gap=240,ball_y=1000,ball_vx=60)]):rows.append(dict(kind='close_ground',side=side,seed=9920011+side*10000+i*1009,**kw))
with ProcessPoolExecutor(max_workers=4) as pool:r=list(pool.map(process,rows))
(out/'search.json').write_text(json.dumps(r,indent=2));print('CLOSEGROUND',[(x['scene']['side'],bool(x['best']),x['baseline']['first_contact']) for x in r],flush=True)
