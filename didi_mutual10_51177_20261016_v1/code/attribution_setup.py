"""Factorial initialization and observational diagnostics; no new loss or RNG."""
import hashlib
import torch

ARMS = {'g0p0':(False,False),'g1p0':(True,False),'g0p1':(False,True),'g1p1':(True,True)}

def tensor_hash(state, keys):
    h=hashlib.sha256()
    for k in sorted(keys):
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def initialize_joint(model, checkpoint, arm):
    from src.model.stgcn_n.fewnode_stgcn import load_transferable_state
    graph_source,predictor_source=ARMS[arm]
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    source=torch.load(checkpoint,map_location='cpu',weights_only=True)
    info=load_transferable_state(model,source)
    source={k.removeprefix('module.'):v for k,v in source.items()}
    graph=[k for k in info['loaded_keys'] if k.startswith('graph_learner.')]
    predictor=[k for k in info['loaded_keys'] if k not in graph]
    assert len(graph)==4 and len(predictor)==38
    chosen=(graph if graph_source else [])+(predictor if predictor_source else [])
    selected={k:v.clone() for k,v in initial.items()}
    for k in chosen:selected[k]=source[k].detach().clone()
    model.load_state_dict(selected,strict=True)
    actual=model.state_dict()
    assert all(torch.equal(actual[k],selected[k]) for k in selected)
    common=[k for k in initial if k not in graph+predictor]
    assert all(torch.equal(actual[k],initial[k]) for k in common)
    anchors={k:actual[k].detach().clone() for k in graph+predictor}
    info.update(attribution_arm=arm,graph_source=graph_source,predictor_source=predictor_source,
                graph_factor_keys=graph,predictor_factor_keys=predictor,source_loaded_keys=chosen,
                actual_source_loaded_tensor_count=len(chosen),anchor_tensor_count=42,anchor_policy='own_initialization_all_arms',
                common_initial_sha256=tensor_hash(initial,common),common_final_sha256=tensor_hash(actual,common),
                graph_initial_sha256=tensor_hash(actual,graph),predictor_initial_sha256=tensor_hash(actual,predictor),
                full_initial_sha256=tensor_hash(actual,actual))
    return anchors,info

def group(name):
    if name.startswith('vae.'):return 'vae'
    if '.residual_' in name:return 'residual'
    if name.startswith('graph_learner.'):return 'graph'
    if name.startswith('output_layer.'):return 'head'
    return 'predictor'

def attach_diagnostics(trainer, log_event):
    initial={k:v.detach().cpu().clone() for k,v in trainer.model.named_parameters()}
    holder={'epoch':None,'before':None}
    def before(opt,args,kwargs):
        epoch=trainer.current_epoch
        if holder['epoch']==epoch:return
        holder['epoch']=epoch;holder['before']={k:p.detach().cpu().clone() for k,p in trainer.model.named_parameters()}
        norms={}
        for k,p in trainer.model.named_parameters():
            g=group(k);norms.setdefault(g,0.)
            if p.grad is not None:norms[g]+=float(p.grad.detach().double().square().sum().cpu())
        holder['grad_norm']={g:x**.5 for g,x in norms.items()}
    def after(opt,args,kwargs):
        if holder['before'] is None:return
        updates={};drift={}
        for k,p in trainer.model.named_parameters():
            v=p.detach().cpu();g=group(k)
            updates[g]=updates.get(g,0.)+float((v.double()-holder['before'][k].double()).square().sum())
            drift[g]=drift.get(g,0.)+float((v.double()-initial[k].double()).square().sum())
        log_event(trainer.configs['save_dir'],'module_update_diagnostics',epoch=trainer.current_epoch,
                  steps=trainer.successful_steps,gradient_l2=holder['grad_norm'],
                  update_l2={g:x**.5 for g,x in updates.items()},drift_l2={g:x**.5 for g,x in drift.items()},
                  scope='first_actual_update_each_epoch_no_extra_forward_backward',
                  attribution_arm=trainer.configs['attribution_arm'])
        holder['before']=None
    trainer.opt.register_step_pre_hook(before)
    trainer.opt.register_step_post_hook(after)
