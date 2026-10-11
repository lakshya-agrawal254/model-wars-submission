"""Read-only saved-policy, exact behavior distribution and chronological replay checks."""
from train import *
def main():
 torch.set_num_threads(2);saved=torch.load(OUT/'latest.pt',weights_only=True);p=Policy().eval();p.load_state_dict(saved['state_dict'],strict=True);plain=Policy().eval();plain.load_state_dict(torch.load(OUT/'candidate.pt',weights_only=True)['state_dict'],strict=True);ref=Policy().eval();ref.load_state_dict(torch.load(START,weights_only=True)['state_dict'],strict=True);opt=torch.optim.Adam(p.parameters());opt.load_state_dict(saved['actor_optimizer']);critic=Critic();critic.load_state_dict(saved['critic']);co=torch.optim.Adam(critic.parameters());co.load_state_dict(saved['critic_optimizer']);rows=[]
 assert len(opt.state)==8 and all(int(v['step'])==saved['actor_updates'] for v in opt.state.values());assert sum(x.numel() for x in p.parameters())==26181;assert list(saved['state_dict'])==list(torch.load(START,weights_only=True)['state_dict'])
 for snap in OUT.glob('update-*.pt'):
  d=torch.load(snap,weights_only=True);pp=Policy();pp.load_state_dict(d['state_dict'],strict=True);o=torch.optim.Adam(pp.parameters());o.load_state_dict(d['actor_optimizer']);assert len(o.state)==8
 with torch.inference_mode():
  for batch in range(saved['completed_batch']+1):
   path=OUT/f'batch-{batch}.npz';assert sha(path)==saved['batch_data_sha256'][path.name];data=np.load(path);behavior=Policy().eval();behavior.load_state_dict(torch.load(OUT/f'batch-{batch}-behavior.pt',weights_only=True)['state_dict'],strict=True)
   for i in range(8):
    x=torch.tensor(data[f'x_{i}']);l,s,h=p(x);ll,ss,hh=plain(x);torch.testing.assert_close(l,ll,rtol=0,atol=0);torch.testing.assert_close(s,ss,rtol=0,atol=0);torch.testing.assert_close(h,hh,rtol=0,atol=0);ch=None;ls=[]
    for chunk in x.split(128):a,b,ch=p(chunk,ch);ls.append(a)
    torch.testing.assert_close(torch.cat(ls),l,rtol=1e-5,atol=1e-6);torch.testing.assert_close(ch,h,rtol=1e-5,atol=1e-6)
    bl,bs,bh,mem=sequence(behavior,x);acts=torch.tensor(data[f'actions_{i}']);torch.testing.assert_close(mem,torch.tensor(data[f'memory_{i}']),rtol=1e-5,atol=1e-6);torch.testing.assert_close(density(bl,bs,acts[:,0].long(),acts[:,1]),acts[:,2],rtol=1e-4,atol=1e-4)
    rl,rs,_=ref(x);ab=x.clone();ab[:,3:6]=0;ab[:,9:12]=0;al,ass,_=p(ab)
    rows.append(dict(batch=batch,index=i,decisions=len(x),greedy_modes=torch.bincount(l.argmax(-1),minlength=4).tolist(),sampled_modes=np.bincount(acts[:,0].numpy().astype(int),minlength=4).tolist(),forecast_ablation_mode_changes=int((l.argmax(-1)!=al.argmax(-1)).sum()),forecast_ablation_steer_mean=float((s-ass).abs().mean()),mode_changes_vs_initializer=int((l.argmax(-1)!=rl.argmax(-1)).sum()),steer_change_vs_initializer_mean=float((s-rs).abs().mean()),forecast_valid_fraction=float(x[:,9].mean()),forecast_horizon_seconds=[float(x[:,10].min()*2),float(x[:,10].max()*2)],dt_values=np.unique(x[:,12]).tolist(),reward_component_sums=data[f'components_{i}'].sum(axis=0).tolist()))
 preserved=json.loads((OUT/'preservation-before.json').read_text());assert all(sha(ROOT/f)==v for f,v in preserved.items());official=ROOT.parent/'official';files=['src/policy.py','src/observation_contract.py','tools/validate_submission.py','.github/workflows/validate.yml'];assert all((ROOT/f).read_bytes()==(official/f).read_bytes() for f in files)
 report=dict(strict_load=True,all_actor_optimizer_states_restored=True,critic_and_optimizer_restored=True,actor_parameters=26181,tensor_names_preserved=True,plain_full_inference_exact=True,recurrent_chunk_parity=True,each_batch_actual_behavior_and_memory_replay=True,all_prior_checkpoints_unchanged=len(preserved),untouched_official_files=files,rows=rows,forecast_input_column_norms=p.encoder[0].weight[:,[3,4,5,9,10,11]].norm(dim=0).detach().tolist(),forecast_gradient_norms=[x['forecast_gradient_norm'] for x in saved['history']],limits='Ablation uses collected training trajectories, not competitive evidence. Physics forecasts cannot predict future car interventions. No native acceptance claimed.')
 (OUT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
