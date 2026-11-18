"""Read-only source and manifest audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_e30_sid as run

e, source = run.e, run.SOURCE
read = lambda p: json.loads(p.read_text())
ck = source/'mlm/scratch/checkpoint-best.pth'
tok = e.e4.load_model(e.TOKENIZER,torch.device('cpu')).eval().requires_grad_(False)
state = torch.load(ck,map_location='cpu',weights_only=False)
files = [ck,e.TOKENIZER,Path(e.__file__)]
frames = {}
for name,folder,splits in [('pretrain',e.PRETRAIN_INDEX,('train','val')),('downstream',e.DOWNSTREAM,('train','val','test'))]:
    for split in splits:
        p = folder/f'{split}_metadata.csv'
        files.append(p)
        frames[name,split] = pd.read_csv(p,low_memory=False)
counts = {f'{n}_{s}':len(f) for (n,s),f in frames.items()}
overlap = {}
for (n,s),f in frames.items():
    for t in ('train','val','test'):
        if n == 'downstream' and s>=t:
            continue
        g = frames['downstream',t]
        overlap[f'{n}_{s}_vs_downstream_{t}'] = {key:len(set(f[key].dropna().astype(str))&set(g[key].dropna().astype(str)))
            for key in ('entity_id','csv_name','member_name') if key in f and key in g}
val = read(source/'downstream/validation_selection.json')['methods']['e12a_region']
test = read(source/'downstream/final_test_summary.json')
fixed = read(source/'fixed_mask_mlm_comparison.json')
audit = dict(file_sha256={str(p):e.sha256(p) for p in files},counts=counts,overlap=overlap,
    config=state['config'],checkpoint_epoch=state['epoch'],trainable_pretrain_parameters=sum(v.numel() for v in state['model'].values()),
    tokenizer_parameter_count=sum(p.numel() for p in tok.parameters()),
    codebook_state_keys={k:list(v.shape) for k,v in tok.state_dict().items() if 'quantize' in k},
    protocol=read(source/'protocol.json'),summary=read(source/'mlm/scratch/summary.json'),
    fixed_mask_historical=fixed,validation_historical=val,test_historical_aggregate=test['aggregate'],
    cache_audit=read(run.CACHE/'audit.json'))
e.dump(run.OUT/'audit.json',audit)
