#!/usr/bin/env python3
"""E12 frozen representation: simple pair probes, train-only geometry, saved errors."""
import argparse,copy,csv,json,pickle,shutil,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score,precision_score,recall_score,silhouette_score,adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_e4_aggregation as e4
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e12_control_diagnostic_20261012'
E7=ROOT/'results/curriculum_v1/e7_objective_20261011'
REPS=['z_e','z_q','heartlang','heartlang_mean','heartlang_max','heartlang_mean_max','z_e_mean_max','z_q_mean_max']
PAIRS=[('Control_vs_OMI',2,0),('Control_vs_NOMI',2,1),('OMI_vs_NOMI',0,1)]

def save(name,data):e4.dump(RUN/name,data)
def status(s):(RUN/'status.txt').write_text(s);print(s,flush=True)
def arrays(rep):
 return {s:(np.load(e4.FEATURES/f'{s}_heartlang_mean_max.npy')[:,768:] if rep=='heartlang_max' else np.load(e4.FEATURES/f'{s}_{rep}.npy')) for s in ('train','val','test')}
def binary_metrics(y,score):
 pred=score>0
 return {'roc_auc':float(roc_auc_score(y,score)),'pr_auc_average_precision':float(average_precision_score(y,score)),
 'accuracy':float(accuracy_score(y,pred)),'f1_positive':float(f1_score(y,pred,zero_division=0)),
 'precision_positive':float(precision_score(y,pred,zero_division=0)),'recall_positive':float(recall_score(y,pred,zero_division=0)),
 'positive_prevalence':float(y.mean()),'n_records':len(y),'n_positive':int(y.sum())}

def bootstrap_auc(y,score,ids,other=None):
 rng=np.random.default_rng(20261012);unique=np.unique(ids);indices=[np.flatnonzero(ids==v) for v in unique];vals=[]
 for _ in range(300):
  ix=np.concatenate([indices[j] for j in rng.integers(0,len(unique),len(unique))])
  if np.unique(y[ix]).size<2:continue
  z=roc_auc_score(y[ix],score[ix]);vals.append(z if other is None else z-roc_auc_score(y[ix],other[ix]))
 return {'lower95':float(np.percentile(vals,2.5)),'upper95':float(np.percentile(vals,97.5)),'bootstrap_draws':len(vals),'unit':'test patient entity cluster; descriptive CI, no selection'}

def probes(labels,meta):
 results=[];scores={}
 for rep in REPS:
  status('probe_'+rep);x=arrays(rep)
  for name,pos,neg in PAIRS:
   masks={s:(labels[s]==pos)|(labels[s]==neg) for s in labels};ys={s:(labels[s][masks[s]]==pos).astype(int) for s in labels}
   scaler=StandardScaler().fit(x['train'][masks['train']]);xx={s:scaler.transform(x[s][masks[s]]) for s in x};candidates=[];estimators=[]
   for C in (.01,.1,1.):
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always',ConvergenceWarning)
     model=LogisticRegression(C=C,solver='liblinear',max_iter=2000,tol=1e-4,random_state=0).fit(xx['train'],ys['train'])
     assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Logistic failed convergence'
    candidates.append({'C':C,'val':binary_metrics(ys['val'],model.decision_function(xx['val'])),'iterations':model.n_iter_.tolist()});estimators.append(model)
   chosen=max(range(len(candidates)),key=lambda i:candidates[i]['val']['roc_auc']);model=estimators[chosen]
   score=model.decision_function(xx['test']);scores[(rep,name)]=score
   out=RUN/'probes'/rep/name;out.mkdir(parents=True,exist_ok=False);np.save(out/'test_scores.npy',score)
   with (out/'probe.pkl').open('wb') as f:pickle.dump({'scaler':scaler,'model':model},f)
   result={'representation':rep,'pair':name,'positive_class':e4.LABELS[pos],'input_dim':x['train'].shape[1],'selected_C':candidates[chosen]['C'],'val':candidates[chosen]['val'],'test':binary_metrics(ys['test'],score),'candidates':candidates}
   result['test_auc_cluster_bootstrap']=bootstrap_auc(ys['test'],score,meta['test'].loc[masks['test'],'entity_id'].astype(str).to_numpy());results.append(result);save('probe_partial.json',results)
 save('pairwise_probes.json',results)
 differences=[]
 for a,b in [('z_e','z_q'),('z_e_mean_max','z_q_mean_max')]:
  for name,pos,neg in PAIRS:
   mask=(labels['test']==pos)|(labels['test']==neg);y=(labels['test'][mask]==pos).astype(int);s1=scores[(a,name)];s2=scores[(b,name)]
   differences.append({'comparison':b+' minus '+a,'pair':name,'auc_difference':float(roc_auc_score(y,s2)-roc_auc_score(y,s1)),
    'paired_cluster_bootstrap':bootstrap_auc(y,s2,meta['test'].loc[mask,'entity_id'].astype(str).to_numpy(),s1)})
 save('VQ_paired_auc_diagnostic.json',differences)


def describe(values):
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'q25':float(np.percentile(a,25)) if len(a) else None,'q75':float(np.percentile(a,75)) if len(a) else None}

def geometry(labels,meta):
 status('geometry');geometry={};plot_records=[]
 for rep in ['z_e','z_q','heartlang','heartlang_mean','heartlang_max','heartlang_mean_max']:
  x=arrays(rep);tr=labels['train']<3;te=labels['test']<3;scaler=StandardScaler().fit(x['train'][tr]);xt=scaler.transform(x['train'][tr]);xe=scaler.transform(x['test'][te]);yt=labels['train'][tr];ye=labels['test'][te]
  rng=np.random.default_rng(0);ref=np.concatenate([rng.choice(np.flatnonzero(yt==k),512,replace=False) for k in range(3)])
  nn=NearestNeighbors(n_neighbors=5,metric='cosine').fit(normalize(xt[ref]));distance,ids=nn.kneighbors(normalize(xe));neighbor=yt[ref][ids];counts={}
  for k in range(3):
   sel=ye==k;counts[e4.LABELS[k]]={'n':int(sel.sum()),'mean_neighbor_class_fraction':dict(zip(e4.LABELS[:3],[(neighbor[sel]==j).mean().item() for j in range(3)])),
    'nearest_neighbor_label_counts':dict(zip(e4.LABELS[:3],[int((neighbor[sel,0]==j).sum()) for j in range(3)]))}
  control=ye==2;geometry[rep]={'balanced_reference':{'512_per_class':True,'seed':0,'diagnostic_reference_only_not_training_sampler':True},'neighbors':counts,
   'control_nearest_other_class':None}
  refother=ref[yt[ref]!=2];nn2=NearestNeighbors(n_neighbors=1,metric='cosine').fit(normalize(xt[refother]));d2,i2=nn2.kneighbors(normalize(xe[control]));lbl=yt[refother][i2[:,0]]
  geometry[rep]['control_nearest_other_class']=dict(zip(e4.LABELS[:2],[int((lbl==j).sum()) for j in range(2)]))
  if rep=='heartlang_mean_max':
   pca=PCA(n_components=20,random_state=0).fit(xt);ztr=pca.transform(xt);zte=pca.transform(xe)
   controltr=ztr[yt==2];cand=[];models=[]
   for k in (2,3,4,5):
    km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(controltr);cand.append({'k':k,'train_silhouette':float(silhouette_score(controltr,km.labels_))});models.append(km)
   chosen=max(range(len(cand)),key=lambda i:cand[i]['train_silhouette']);km=models[chosen];repeats=[KMeans(n_clusters=km.n_clusters,n_init=10,random_state=s).fit(controltr).labels_ for s in (1,2)]
   ct=km.predict(zte[control]);cv=scaler.transform(x['val'][labels['val']==2]);cv=pca.transform(cv);valclusters=km.predict(cv)
   clusters=[];originaltest=np.flatnonzero(te)[control]
   for cid in range(km.n_clusters):
    group=ct==cid;rows=originaltest[group];predgroups={}
    for seed in (0,1,2):
     pred=np.load(E7/f'CE_seed{seed}/test_raw_outputs.npy').argmax(1);predgroups[str(seed)]=dict(zip(e4.LABELS,[int((pred[rows]==j).sum()) for j in range(4)]))
    clusters.append({'cluster':cid,'train_n':int((km.labels_==cid).sum()),'val_n':int((valclusters==cid).sum()),'test_n':int(group.sum()),'E7_predicted_counts':predgroups,
     'test_mean_neighbor_class_fraction':dict(zip(e4.LABELS[:3],[(neighbor[control][group]==j).mean().item() if group.any() else None for j in range(3)]))})
   geometry[rep]['Control_train_only_PCA_KMeans']={'PCA_dimensions':20,'explained_variance_first2':float(pca.explained_variance_ratio_[:2].sum()),'explained_variance20':float(pca.explained_variance_ratio_.sum()),'candidates':cand,'selected_k':km.n_clusters,'selection':'train Control silhouette only; not clinical subgroup evidence','ARI_seeds1_2_vs0':[float(adjusted_rand_score(km.labels_,q)) for q in repeats],
    'test_silhouette':float(silhouette_score(zte[control],ct)) if np.unique(ct).size>1 else None,'clusters':clusters}
   with (RUN/'geometry_transform.pkl').open('wb') as f:pickle.dump({'scaler':scaler,'pca':pca,'kmeans':km},f)
   np.save(RUN/'test_patient_PCA20.npy',zte);np.save(RUN/'test_patient_parent_indices.npy',np.flatnonzero(te))
   geometry[rep]['control_neighbor_fractions_by_E7_error_group']={}
   for seed in (0,1,2):
    pred=np.load(E7/f'CE_seed{seed}/test_raw_outputs.npy').argmax(1);groups={}
    for k in range(4):
     select=pred[originaltest]==k;groups[e4.LABELS[k]]={'n':int(select.sum()),'neighbor_fractions':dict(zip(e4.LABELS[:3],[(neighbor[control][select]==j).mean().item() if select.any() else None for j in range(3)]))}
    geometry[rep]['control_neighbor_fractions_by_E7_error_group'][str(seed)]=groups
   for i,row in enumerate(np.flatnonzero(te)):
    plot_records.append({'parent_row':int(row),'label':e4.LABELS[int(labels['test'][row])],'pc1':float(zte[i,0]),'pc2':float(zte[i,1])})
 save('geometry.json',geometry);pd.DataFrame(plot_records).to_csv(RUN/'PCA_visualization_points.csv',index=False)


def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config={'representations':REPS,'pairs':PAIRS,'probe':'train-only StandardScaler + unweighted L2 LogisticRegression(liblinear), intercept, C .01/.1/1 selected by pair val ROC-AUC; fixed seed0, maxiter2000/tol1e-4',
 'PR_AUC':'average_precision, positive Control for Control pairs, OMI for OMI/NOMI','probe_prediction':'zero logit threshold, no tuning','test':'selected probe once; stored scores descriptive cluster bootstrap300 at original patient-entity level',
 'geometry':'train-only standardization/PCA20, balanced diagnostic nearest-neighbor reference512/class, cosine K5; ControlKMeans k2..5 selected train silhouette, stability seeds1/2',
 'error_case_baseline':'E7 CE seeds0/1/2 saved predictions; ground truth used only descriptive group assignment','clinical_limit':'no new labels or human clinical interpretation; PCA/clustering descriptive not separability evidence',
 'fixed':{'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'features':str(e4.FEATURES)},'split_unchanged':True}
 save('protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__)]+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ['labels','z_e','z_q','heartlang','heartlang_mean','heartlang_mean_max','z_e_mean_max','z_q_mean_max','sid_frequency','valid_heartbeat_mask']]+[e4.DATA/f'{s}_metadata.csv' for s in ('train','val','test')]
 before={str(p):e4.sha(p) for p in files};save('source_fingerprints.json',before);shutil.copy2(E7/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 labels={s:np.load(e4.FEATURES/f'{s}_labels.npy').argmax(1) for s in ('train','val','test')};meta={s:pd.read_csv(e4.DATA/f'{s}_metadata.csv') for s in labels}
 ids={}
 for s in labels:
  assert len(meta[s])==len(labels[s]);patient=labels[s]<3;entities=meta[s].loc[patient,'entity_id'];assert entities.notna().all();ids[s]=set(entities.astype(str))
 assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
 save('identity_audit.json',{'patient_records':{s:int((y<3).sum()) for s,y in labels.items()},'patient_entities':{s:len(v) for s,v in ids.items()},'entity_disjoint':True})
 with threadpool_limits(limits=4):probes(labels,meta);geometry(labels,meta)
 assert all(e4.sha(p)==before[str(p)] for p in files)
 save('core_validation.json',{'sources_unchanged':True,'24_pairwise_probes':True,'no_test_selection':True,'train_only_scalers_and_geometry':True})
 status('core_complete_heartbeat_pending')
if __name__=='__main__':main()
