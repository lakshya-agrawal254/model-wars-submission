"""Report independent-pair evidence; never treat safety-capped leads as wins."""
import json,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def summarize(path):
 d=json.loads(path.read_text());groups=collections.defaultdict(list);completed=[];censored=[]
 for row in d['records']:
  groups[row['seed']].append(row)
  (censored if row['end']=='safety_cap' or row['end'].startswith('censored') else completed).append(row)
 scores=[sum(1 if r in completed and r['result']=='W' else 0 for r in rows)/len(rows) for rows in groups.values()]
 positive=sum(s>.5 for s in scores);negative=sum(s<.5 for s in scores);n=positive+negative
 p=sum(math.comb(n,i) for i in range(positive,n+1))/2**n if n else 1.
 result={'file':path.name,'completed_wins':sum(r['result']=='W' for r in completed),'completed_losses':sum(r['result']=='L' for r in completed),'completed_draws':sum(r['result']=='D' for r in completed),'unresolved':len(censored),'observed_goals_for_including_capped_play':d['goals_for'],'observed_goals_against_including_capped_play':d['goals_against'],'independent_side_pairs':len(groups),'conservative_pair_scores':scores,'one_sided_exact_pair_sign_p':p,'promotion_pass':p<.05,'evaluation_elapsed_seconds':d['elapsed_seconds'],'raw_WLD_is_not_completed_match_WLD':True}
 return result
if __name__=='__main__':
 results=[summarize(ROOT/'runs'/n) for n in ['screen-A.json','screen-B.json','screen-E.json','confirmation-A.json']]
 target=ROOT/'docs/evidence/evaluation-summary.json';target.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
