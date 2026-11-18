from types import SimpleNamespace
import numpy as np
import torch
import run_e29_pooling as p
from run_e29_r import metrics

base,e=p.base,p.e
@torch.no_grad()
def extract(device, data):
    tok = e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False)
    model = e.load_continuous_checkpoint(e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",device).eval().requires_grad_(False)
    b = model.backbone
    for split in ("test",):
        w,c,t = [np.load(e.DOWNSTREAM/f"{split}_{k}.npy",mmap_mode="r") for k in ("data","data_in_chans","data_in_times")]
        packed = torch.empty((len(w),4608+256*769),device=device)
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
                    packed[i:i+len(x),index*1536:(index+1)*1536] = base.pool(state,v)
                    if j == 9:
                        packed[i:i+len(x),4608:] = torch.cat((state,v[...,None]),-1).flatten(1)
        data[split]["z"] = packed


@torch.no_grad()
def main():
    out=e.ROOT/"results/e25_e29_locked_test"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE").exists():
        return
    device=torch.device("cuda")
    load=lambda k:torch.tensor(np.load(e.E10_RUN/f"features/test_{k}.npy"),device=device)
    regions,mask=e.e11.masked_region_means(load("lead_ze_mean_max"),load("lead_present").bool())
    y=np.load(e.DOWNSTREAM/"test_labels.npy")
    d={"g":load("global_ze_mean_max"),"regions":regions,"mask":mask,"y":torch.tensor(y,device=device)}
    assert len(y)==1366 and len(regions)==len(y) and len(d["g"])==len(y)
    extract(device,{"test":d})
    result={}
    for name,folder,method in (("E25",p.source,"learned_mix"),("Shared",p.run,"shared"),("Class",p.run,"class")):
        predictions,rows,hashes=[],[],[]
        for seed in range(3):
            ck=folder/"downstream"/method/f"seed{seed}/checkpoint-best.pth"
            model=p.Head(method,seed).to(device).eval()
            model.load_state_dict(torch.load(ck,map_location=device,weights_only=False)["model"])
            _,q,_=base.evaluate(model,d)
            predictions.append(q)
            rows.append(metrics(y,q))
            hashes.append(e.sha256(ck))
        np.save(out/f"{name}_probabilities.npy",predictions)
        result[name]=dict(per_seed=rows,mean={k:float(np.mean([x[k] for x in rows])) for k in rows[0]},
            sd={k:float(np.std([x[k] for x in rows],ddof=1)) for k in rows[0]},
            ensemble=metrics(y,np.mean(predictions,0)),checkpoint_sha256=hashes)
    base.dump(out/"test_summary.json",result)
    (out/"COMPLETE").touch()
if __name__=="__main__":
    main()
