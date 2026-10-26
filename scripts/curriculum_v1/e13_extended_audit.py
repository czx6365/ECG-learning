import json
from pathlib import Path
import pandas as pd
import numpy as np
import run_e13_control_audit as e
D=pd.concat([pd.read_csv(e.e4.DATA/f'{s}_metadata.csv').assign(audit_split=s) for s in e.SPLITS]);D=D[D.label_id<3]
sets=D.groupby('entity_id').label_id.apply(lambda x:tuple(sorted(set(map(int,x)))));total=D.groupby('entity_id').size();out={'entity_class_combinations':{'/'.join(e.NAMES[i] for i in key):int((sets==key).sum()) for key in sorted(set(sets))},'all_patient_unique_entities':int(D.entity_id.nunique()),'all_patient_records_per_entity':e.describe(total),'all_patient_records_per_entity_histogram':total.value_counts().sort_index().to_dict(),'by_class':{},'Control_split_year_total_variation':{},'existing_label_mapping':{}}
for k,name in enumerate(e.NAMES):
 ids=set(D.loc[D.label_id==k,'entity_id']);multi=set(sets[sets.apply(len)>1].index)&ids;sub=D[D.entity_id.isin(ids)];out['by_class'][name]={'unique_entities_with_class':len(ids),'multi_label_entities':len(multi),'fraction_multi_label_entities':len(multi)/len(ids),'class_records_from_multi_label_entities':int(((D.label_id==k)&D.entity_id.isin(multi)).sum()),'all_class_records_per_entity_with_this_class':e.describe(total.loc[list(ids)]),'by_split':{}}
 for s in e.SPLITS:
  ds=D[(D.audit_split==s)&(D.label_id==k)];idss=set(ds.entity_id);ms=idss&multi;out['by_class'][name]['by_split'][s]={'class_entities':len(idss),'multi_label_entities':len(ms),'class_records_from_multi_label_entities':int(ds.entity_id.isin(ms).sum()),'all_class_records_per_entity':e.describe(total.loc[list(idss)])}
 for col in ('original_label_name','original_label_id','label_name','标签'):
  if col in D:out['existing_label_mapping'].setdefault(col,{})[name]=e.dist(D.loc[D.label_id==k,col])
for a,b in [('train','val'),('train','test'),('val','test')]:
 freq=[]
 for s in (a,b):
  ds=D[(D.audit_split==s)&(D.label_id==2)];years=ds['心电图时间（日期）'].astype('string').str.extract(r'((?:19|20)\d{2})',expand=False);freq.append(years.value_counts(normalize=True).reindex([str(y) for y in range(2017,2026)],fill_value=0).to_numpy())
 out['Control_split_year_total_variation'][a+'_'+b]=float(np.abs(freq[0]-freq[1]).sum()/2)
out['interpretation_limit']='Multi-label entity means different existing record labels within one entity. This alone does not establish incorrect labels, clinical evolution, or identity collision. Per-class record-per-entity counts differ from all-class record-per-entity counts.'
e.save('extended_entity_audit.json',out)
print(json.dumps(out,ensure_ascii=False))
