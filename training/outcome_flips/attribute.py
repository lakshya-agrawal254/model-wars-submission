"""Separate actual-dodge benefit from first-jump benefit; no actor updates."""
from world import *
from search import useful
import json
from concurrent.futures import ProcessPoolExecutor
OUT=ROOT/'runs/outcome-flip-attribution-v1'
def process(row):
 if row['best'] is None:return dict(scene=row['scene'],selected=False,reason='No sequence better than baseline')
 best=row['best'];no,*_=trajectory(row['scene'],best['schedule'],disable_second_press=True)
 rv=best['contacts'][0]['goalward_velocity'];nv=no['contacts'][0]['goalward_velocity'] if no['contacts'] else 0
 contact_gain=best['first_contact'] is not None and (no['first_contact'] is None or best['first_contact']<no['first_contact']-.1)
 keep=best['goals_for']>=no['goals_for'] and best['goals_against']<=no['goals_against'] and (rv>nv+100 or contact_gain and rv>=nv-50 or best['goals_for']>no['goals_for'])
 return dict(scene=row['scene'],selected=keep,best=best,no_second_press=no,reason='Actual dodge improves outcome over same Jump/release without secondpress' if keep else 'Benefit not attributable to secondpress dodge')
def main():
 assert not OUT.exists();OUT.mkdir();rows=json.loads((ROOT/'runs/outcome-flip-search-v1/search.json').read_text())
 with ProcessPoolExecutor(max_workers=4) as pool:result=list(pool.map(process,rows))
 (OUT/'records.json').write_text(json.dumps(result,indent=2));print('DODGE-ATTRIBUTED',sum(r['selected'] for r in result),'of',len(result),[(r['scene']['kind'],r['scene'].get('index'),r['selected']) for r in result],flush=True)
if __name__=='__main__':main()
