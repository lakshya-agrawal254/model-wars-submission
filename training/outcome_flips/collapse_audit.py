"""Same-real-prefix ablation to explain newly rejected held-Jump collapse."""
from world import *
import json,copy
out=ROOT/'runs/outcome-flip-collapse-audit-v1';assert not out.exists();out.mkdir();old=ROOT/'runs/outcome-flip-learning-v1/latest.pt';new=ROOT/'runs/outcome-flip-refinement-v1/latest.pt';scene=dict(kind='kickoff',side=0,index=4,enemy_index=1,seed=9997011)
r,data,trace=trajectory(scene,path=new,record=True);np.savez_compressed(out/'real-prefix.npz',**data);(out/'trace.json').write_text(json.dumps(trace,indent=2));x=torch.tensor(data['x']);a=load(old);b=load(new);swap_head=load(new);swap_memory=load(old)
with torch.no_grad():swap_head.head.load_state_dict(a.head.state_dict());swap_memory.head.load_state_dict(b.head.state_dict())
rows={}
with torch.inference_mode():
 for name,p in [('old128_on_sameprefix',a),('rejected144',b),('newmemory_oldhead',swap_head),('oldmemory_newhead',swap_memory)]:
  l,s,h=p(x);rows[name]=dict(counts=[int((l.argmax(-1)==k).sum()) for k in range(4)],first_mode=int(l[0].argmax()),mean_jump_Chase_logit_difference=float((l[:,2]-l[:,1]).mean()))
channels=torch.load(new,weights_only=True)['phase_channels'];report=dict(actual_neural_trajectory=r,sameprefix_model_ablation=rows,channels=channels,old_Jump_minus_Chase_readout_at_channels=(a.head.weight[2,channels]-a.head.weight[1,channels]).detach().tolist(),new_Jump_minus_Chase_readout_at_channels=(b.head.weight[2,channels]-b.head.weight[1,channels]).detach().tolist(),causal_limit='Hybrid models replay identical recorded observations, not independent closedloop gameplay. Isolates contributions onthisprefix; not proof ofallnative failure causes.')
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='actual_neural_trajectory'},indent=2))
