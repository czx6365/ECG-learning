from types import SimpleNamespace
import numpy as np
import torch
from run_e29_r import e,b as base
@torch.no_grad()
def extract(data):
    device=torch.device("cuda")
    tok = e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False)
    model = e.load_continuous_checkpoint(e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",device).eval().requires_grad_(False)
    b = model.backbone
    for split in ("train",):
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

