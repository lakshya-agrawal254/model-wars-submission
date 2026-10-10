from world_ground import *
import hashlib
OUT=ROOT/'runs/ground-trajectory-learning-v1'
def main():
 b=aggregate(json.loads((OUT/'development-control.json').read_text()));rows=[]
 for p in sorted(OUT.glob('development-*.json')):
  if p.name=='development-control.json':continue
  n=int(p.stem.split('-')[-1]);s=aggregate(json.loads(p.read_text()));ok=s['contacts']>=b['contacts'] and s['stationary']<=b['stationary']+2 and s['circling']<=b['circling'] and s['fallen']<=b['fallen']+2;rows.append(dict(update=n,summary=s,eligible=ok))
 qualified=[v for v in rows if v['eligible']]
 best=min(qualified,key=lambda v:(v['summary']['circling'],-v['summary']['contacts'],v['summary']['GA']-v['summary']['GF'],v['summary']['stationary'])) if qualified else max(rows,key=lambda v:v['update']);chosen=OUT/f'actor-{best["update"]}.pt';selected=OUT/'screened-actor.pt';assert not selected.exists();selected.write_bytes(chosen.read_bytes());r=dict(selected_update=best['update'],eligible=bool(qualified),actor=str(chosen.relative_to(ROOT)),selected_actor=str(selected.relative_to(ROOT)),SHA256=hashlib.sha256(selected.read_bytes()).hexdigest(),all_checks=rows,baseline=b,selection='training/ground_trajectories/selection.json',freeze_before_fresh_seed_evaluation=True,promoted=False);(OUT/'selection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
