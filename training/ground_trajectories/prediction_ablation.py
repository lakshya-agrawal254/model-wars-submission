"""Diagnostic inference-only copy ablation. No saved model or runtime changes."""
from world_ground import *
from concurrent.futures import ProcessPoolExecutor,as_completed
OUT=ROOT/'runs/ground-prediction-diagnostic-v1';SELECTED=ROOT/'runs/ground-trajectory-learning-v1/screened-actor.pt'
def worker(args):
 scene,ablate=args;import world_ground as w;original=w.load
 def loading(path):
  p=original(path)
  if ablate and Path(path)==SELECTED:
   with torch.no_grad():p.encoder[0].weight[:,[3,4,5,9]]=0
  return p
 w.load=loading
 try:return w.run(scene,path=SELECTED,hz=60)[0]
 finally:w.load=original

def main():
 assert not OUT.exists();OUT.mkdir();cases=[r for r in scenes(10401011) if r['kind'] in ['curve','uturn','beside','shot']];rows={'actual_forecast':[],'forecast_weight_ablation':[]}
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks={pool.submit(worker,(s,a)):('forecast_weight_ablation' if a else 'actual_forecast') for a in [False,True] for s in cases}
  for f in as_completed(tasks):rows[tasks[f]].append(f.result())
 (OUT/'records.json').write_text(json.dumps(rows,indent=2));report=dict(summary={k:aggregate(v) for k,v in rows.items()},scope='Actual13D realforecast states. Only private in-memory copiesof selectedactorzero4forecastencodercolumns; referenceopponentunchanged; no models saved/tuned. Two-sided controlled scenes, notfullmatches or nativeRLBot evidence.',candidate_unchanged=True);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
