"""Frozen paired comparison; no selection using training loss or rewards."""
import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/failure-prefix-trust-screen-v1'
def main():
 records=json.loads((OUT/'records.json').read_text());summary=json.loads((OUT/'result.json').read_text())['summary'];rng=np.random.default_rng(12299999);paired={}
 for key,rs in records.items():
  groups={}
  for r in rs:
   groups.setdefault(r['seed'],[]).append(0 if 'cap' in r['end'] else 1 if r['result']=='W' else -1 if r['result']=='L' else 0)
  assert len(groups)==4 and all(len(v)==2 for v in groups.values());means=np.array([np.mean(v) for v in groups.values()]);samples=rng.choice(means,(100000,len(means)),replace=True).mean(1)
  paired[key]={'pair_mean_win_minus_loss':means.tolist(),'pair_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),'limits':'Only four independent seed pairs. Exploratory bootstrap, not a reliable small-sample significance guarantee; unresolved assigned zero, not declared a draw.'}
 common={}
 for a in ['candidate','outcome']:
  ss=[summary[f'{a}_vs_{b}_60'] for b in ['ground','geometric']];common[a]={k:sum(s[k] for s in ss) for k in ['matches','W','L','D','U','GF','GA','stationary_seconds','observed_seconds','circling_windows','goalward_contacts','backward_contacts']};common[a]['stationary_fraction']=common[a]['stationary_seconds']/common[a]['observed_seconds']
 c=common['candidate'];b=common['outcome'];gates={'direct_wins_at_least_6_of_8':summary['candidate_vs_outcome_60']['W']>=6,'common_wins_not_worse':c['W']>=b['W'],'common_goal_difference_not_worse':c['GF']-c['GA']>=b['GF']-b['GA'],'stationary_fraction_increase_at_most_1_percentage_point':c['stationary_fraction']<=b['stationary_fraction']+.01,'circling_rate_increase_at_most_25_percent':c['circling_windows']/c['observed_seconds']<=1.25*b['circling_windows']/b['observed_seconds']}
 out={'screen_gates':gates,'qualifies_for_independent_confirmation':all(gates.values()),'paired':paired,'common_opponents':common,'selected_automatically':False,'native_acceptance':'UNTESTED','continuation_only_if_gates_pass':True}
 (OUT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
