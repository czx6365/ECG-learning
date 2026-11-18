"""Cross-check source membership and exposure-adjusted attention mass."""
import numpy as np
import pandas as pd

from run_e29_s import OUT,e,b

def main():
    meta=pd.read_csv(e.DOWNSTREAM/"train_metadata.csv",dtype={"entity_id":str})
    folds=pd.read_csv(OUT/"fold_assignment.csv")
    y=np.load(e.DOWNSTREAM/"train_labels.npy")
    assert np.array_equal(meta.label_id,y.argmax(1))
    outer=folds[folds.role=="outer"]
    assert sorted(outer.row)==list(range(len(meta)))
    sources={s:pd.read_csv(e.PRETRAIN_INDEX/f"{s}_metadata.csv",dtype={"entity_id":str},low_memory=False) for s in ("train","val")}
    results={}
    for fold,records in folds.groupby("fold"):
        roles={role:set(meta.iloc[f.row].entity_id) for role,f in records.groupby("role")}
        assert all(not roles[x]&roles[z] for x,z in (("fit","select"),("fit","outer"),("select","outer")))
        held=meta.iloc[records[records.role=="outer"].row]
        results[int(fold)]={}
        for s,pre in sources.items():
            match=held.csv_name.astype(str).str.replace(r"\.csv$","",regex=True).isin(
                pre.csv_name.astype(str).str.replace(r"\.csv$","",regex=True))
            results[int(fold)][s]={"exact_name_records":int(match.sum()),
                "downstream_entities_with_matched_ecg":int(held[match].entity_id.nunique()),
                "direct_entity_id_overlap":len(set(held.entity_id)&set(pre.entity_id))}
    b.dump(OUT/"pretraining_membership_verified.json",results)
    d=pd.read_csv(OUT/"attention_diagnostics.csv")
    ch=np.load(e.DOWNSTREAM/"val_data_in_chans.npy")[d.row]
    ti=np.load(e.DOWNSTREAM/"val_data_in_times.npy")[d.row]
    waves=np.load(e.DOWNSTREAM/"val_data.npy",mmap_mode="r")
    valid=((np.abs(waves).max(-1)>1e-8)&(np.load(e.DOWNSTREAM/"val_data_in_chans.npy")>0)&(np.load(e.DOWNSTREAM/"val_data_in_times.npy")>0))[d.row]
    assert np.array_equal(valid.sum(1),d.valid_tokens)
    for prefix,index,codes in (("lead",ch,(1,2,3,7,8,9,10,11,12)),("time_bin",ti,range(1,11))):
        for code in codes:
            expected=((index==code)&valid).sum(1)/np.maximum(1,valid.sum(1))
            d[f"{prefix}_{code}_excess"]=d[f"{prefix}_{code}_mass"]-expected
    d.groupby(["seed","label"]).mean(numeric_only=True).to_csv(OUT/"attention_exposure_adjusted.csv")

if __name__=="__main__":
    main()
