"""Training-only post-step KL acceptance with exact model/Adam rollback."""
import copy,torch

def bounded_actor_step(actor,optimizer,check,base_lr,mean_limit=.01,trajectory_limit=.025,max_attempts=9):
 weights={k:v.detach().clone() for k,v in actor.state_dict().items()};state=copy.deepcopy(optimizer.state_dict());grads=[None if p.grad is None else p.grad.detach().clone() for p in actor.parameters()];trials=[]
 def restore():
  actor.load_state_dict(weights);optimizer.load_state_dict(copy.deepcopy(state))
  for p,g in zip(actor.parameters(),grads):p.grad=None if g is None else g.clone()
 for attempt in range(max_attempts):
  restore();lr=base_lr*.5**attempt
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.step();measurement=check();accepted=measurement['mean_KL']<=mean_limit and measurement['max_trajectory_KL']<=trajectory_limit
  trials.append(dict(attempt=attempt+1,lr=lr,accepted=accepted,**measurement))
  if accepted:return True,trials
 restore();return False,trials
