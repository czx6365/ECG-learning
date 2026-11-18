"""Frozen E25 readout comparison on Train and Validation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
import run_e25 as base

e = base.e
source = e.ROOT / "results/e25_intermediate_layers"
run = e.ROOT / "results/e29_class_conditioned_pooling"
OriginalHead = base.Head

class Head(OriginalHead):
    def __init__(self, method, seed):
        super().__init__("learned_mix", seed)
        self.variant = method
        if method != "learned_mix":
            self.projection = nn.Linear(768, 32, bias=False)
            self.query = nn.Parameter(torch.empty(4 if method in ("class", "capacity") else 1, 32))
            nn.init.normal_(self.query, std=0.1)
            self.readout = nn.Parameter(torch.empty(4,32))
            nn.init.normal_(self.readout, std=0.1)
            self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, z, g, regions, mask, training_outputs=False):
        pooled = z[:, :4608].reshape(-1,3,1536)
        original = super().forward(pooled, g, regions, mask, training_outputs)
        if self.variant == "learned_mix":
            return original
        tokens = z[:,4608:].reshape(-1,256,769)
        valid = tokens[...,768].bool()
        if self.variant == "capacity":
            values = self.projection(pooled[:,1,:768])
            values = values[:,None,:] * self.query[None,:,:]
        else:
            h = self.projection(tokens[...,:768])
            scores = (h @ self.query.T).masked_fill(~valid[...,None], -1e9)
            weights = scores.softmax(1) * valid[...,None]
            values = weights.transpose(1,2) @ h
        delta = self.gamma * (values * self.readout).sum(-1)
        return (original[0]+delta, original[1]) if training_outputs else original+delta

@torch.no_grad()
def extract(device, data):
    tok = e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False)
    model = e.load_continuous_checkpoint(e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",device).eval().requires_grad_(False)
    b = model.backbone
    for split in ("train","val"):
        w,c,t = [np.load(e.DOWNSTREAM/f"{split}_{k}.npy",mmap_mode="r") for k in ("data","data_in_chans","data_in_times")]
        packed = torch.empty((len(w),4608+256*769),device=device)
        packed[:,:4608] = data[split]["z"].flatten(1)
        for i in range(0,len(w),32):
            x,ch,ti = [torch.from_numpy(np.array(a[i:i+32])).to(device) for a in (w,c,t)]
            v = tok.valid_mask(x,ch,ti)
            z = tok.encode_to_quantizer_features(x,ch.long(),ti.long()).float()
            h = b.input_proj(z)*v[...,None]
            h = torch.cat((b.cls_token[None,None].expand(len(x),1,-1),h),1)
            h = b.dropout(h+b.pos_embed)
            fv = torch.cat((torch.ones_like(v[:,:1]),v),1)
            for j,layer in enumerate(b.transformer.layers,1):
                h = e.masked_transformer_forward(SimpleNamespace(layers=[layer]),h,fv)
                if j in (6,9,12):
                    state = b.norm_layer(h)[:,1:]*v[...,None]
                    index = (6,9,12).index(j)
                    torch.testing.assert_close(base.pool(state,v),data[split]["z"][i:i+len(x),index],atol=2e-5,rtol=1e-5)
                    if j == 9:
                        packed[i:i+len(x),4608:] = torch.cat((state,v[...,None]),-1).flatten(1)
        data[split]["z"] = packed

def main():
    run.mkdir(parents=True,exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    frames = [pd.read_csv(e.DOWNSTREAM/f"{s}_metadata.csv") for s in ("train","val")]
    assert not set(frames[0].entity_id) & set(frames[1].entity_id)
    data = base.data(source,device)
    historical = json.loads((source/"validation_summary.json").read_text())
    hashes = {str(p):e.sha256(p) for p in [e.TOKENIZER,e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",*source.glob("downstream/learned_mix/seed*/checkpoint-best.pth")]}
    base.dump(run/"config.json",dict(seeds=[0,1,2],epochs=30,layer=9,projection=32,test_loaded=False))
    base.dump(run/"environment.json",dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    base.status(run,"reproduce_E25")
    baseline = [base.train(run,data,device,"learned_mix",s) for s in range(3)]
    delta = [a["val"]["r2"]-b["val"]["r2"] for a,b in zip(baseline,historical["learned_mix"]["per_seed"])]
    base.dump(run/"baseline_parity.json",delta)
    assert max(map(abs,delta)) < 1e-6
    base.status(run,"extract")
    extract(device,data)
    base.Head = Head
    methods = ("learned_mix","shared","class","capacity")
    base.METHODS = methods
    base.dump(run/"model_parameter_count.json",{m:sum(p.numel() for p in Head(m,0).parameters()) for m in methods})
    d = data["train"]
    args = [d[k][:4] for k in ("z","g","regions","mask")]
    for method in methods[1:]:
        model = Head(method,0).to(device).eval()
        ref = Head("learned_mix",0).to(device).eval()
        torch.testing.assert_close(model(*args),ref(*args),rtol=0,atol=0)
        model(*args).sum().backward()
        assert model.gamma.grad.abs() > 0
        model.gamma.data.fill_(0.01)
        model.zero_grad()
        model(*args).sum().backward()
        assert model.projection.weight.grad.abs().sum() > 0
        changed = args[0].clone()
        token = changed[:,4608:].reshape(-1,256,769)
        token[...,:768][~token[...,768].bool()] = 1000
        torch.testing.assert_close(model(*args),model(changed,*args[1:]))
    rows = baseline.copy()
    for method in methods[1:]:
        for seed in range(3):
            base.status(run,f"{method}_{seed}")
            rows.append(base.train(run,data,device,method,seed))
    base.dump(run/"seed_results.json",rows)
    from report_increment import report
    report(run,rows,data)
    assert all(e.sha256(Path(p))==h for p,h in hashes.items())
    base.status(run,"complete")

if __name__ == "__main__":
    main()
