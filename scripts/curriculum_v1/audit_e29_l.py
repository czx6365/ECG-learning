"""Post-run numerical and source audit without modifying historical artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_e29_l import OUT,Head,e,b,s

def main():
    torch.set_num_threads(4)
    device=torch.device('cuda')
    pools=np.load(OUT/'cache/pooled.npy',mmap_mode='r')
    tokens=np.load(OUT/'cache/layer9.npy',mmap_mode='r')
    ids=np.arange(64)
    z=torch.tensor(np.concatenate((pools[ids].reshape(64,-1),tokens[ids].reshape(64,-1)),1),device=device)
    load=lambda name:torch.tensor(np.load(e.E10_RUN/f'features/train_{name}.npy')[ids],device=device)
    regions,mask=e.e11.masked_region_means(load('lead_ze_mean_max'),load('lead_present').bool())
    args=[z,load('global_ze_mean_max'),regions,mask]
    heads={m:Head(m,0).to(device).eval() for m in ('A','B','C')}
    for m in ('B','C'):
        assert all(torch.equal(v,heads[m].base.state_dict()[k]) for k,v in heads['A'].base.state_dict().items())
    with torch.no_grad():
        delta=(heads['B'](*args)-heads['C'](*args)).abs().max().item()
        torch.testing.assert_close(heads['B'](*args),heads['C'](*args),atol=2e-6,rtol=1e-5)
    model=heads['C']
    model(*args).square().mean().backward()
    gradients=model.adapt.grad.detach().cpu().tolist()
    assert np.isfinite(gradients).all() and np.abs(gradients).sum()>0
    model.gamma.data.fill_(.01)
    altered=z.clone()
    h=altered[:,18432:].reshape(-1,256,769)
    h[...,:768][~h[...,768].bool()]=1000
    with torch.no_grad():
        padding_diff=(model(*args)-model(altered,*args[1:])).abs().max().item()
        assert padding_diff==0
    model_path=e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth'
    backbone=e.load_continuous_checkpoint(model_path,device).eval().requires_grad_(False)
    tok=e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False)
    assert len(backbone.backbone.transformer.layers)==12
    fold=s.OUT/'fold_assignment.csv'
    files=[Path(e.__file__),Path(e.e11.__file__),Path(b.__file__),Path(s.__file__),Path(s.p.__file__),Path(__file__),Path(__file__).with_name('run_e29_l.py'),fold]
    audit=dict(checkpoints={'E12_A':str(model_path),'tokenizer':str(e.TOKENIZER)},
        source_hashes={str(p):e.sha256(p) for p in files},
        transformer_blocks=len(backbone.backbone.transformer.layers),hidden_dimension=768,input_ze_dimension=64,max_tokens=256,pooled_dimension=1536,
        token_validity='tokenizer.valid_mask(x,ch,time); identical mask all layers; CLS excluded from pooling; max invalid=-inf and all-invalid=0; mean valid-count denominator clamp_min(1)',
        initialization_base_exact=True,C_uniform_initial_logits_max_abs_diff=delta,C_fusion_gradient=gradients,padded_token_perturbation_max_abs_diff=padding_diff,
        frozen={'tokenizer':not any(p.requires_grad for p in tok.parameters()),'E12_A_including_MLM':not any(p.requires_grad for p in backbone.parameters()),'eval':not backbone.training and not tok.training},
        rng='E29-S PairedHead saves/restores CPU RNG around Shared initialization; A/B/C modify only zero scalar parameters, consuming no new RNG. Train shuffle generator and E11 initialization paired per seed.',
        gradient_mode='frozen cache uses torch.no_grad, not inference_mode; downstream cache tensors are ordinary tensors; downstream optimizer only sees Head parameters',
        training={'epochs':30,'batch':64,'train_batches':'floor(n_fit/64)','optimizer':'AdamW','lr':5e-4,'weight_decay':.05,'warmup_epochs':5,'clip_norm':3,'loss':'CE4+0.2 patient CE3','selection':'inner select Overall R2, outer never selects epochs'},
        pretraining_overlap=json.loads((s.OUT/'pretraining_overlap.json').read_text()),test_accessed=False,historical_validation_used=False)
    b.dump(OUT/'code_protocol_audit.json',audit)
    print('audit passed')

if __name__=='__main__':main()
