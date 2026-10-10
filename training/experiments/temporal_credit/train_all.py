import argparse,json
from pathlib import Path
from train import run,PROTOCOL
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
protocol=json.loads(PROTOCOL.read_text())
for cfg in protocol['cells']:
 run(cfg,a.output/cfg['name'],protocol['budget_per_cell'])
