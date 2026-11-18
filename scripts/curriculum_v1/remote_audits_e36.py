"""Read-only original model hashes and source Train selection reconciliation. Run remotely."""
import json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_e29_s as s
OUT=s.e.ROOT/'results/e36_episode_label_audit'
def main():
 OUT.mkdir(exist_ok=True)
 paths=[s.e.TOKENIZER,s.e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',*s.OUT.glob('fold*/downstream/shared/seed*/checkpoint-best.pth')]
 current={str(p):s.e.sha256(p) for p in paths};before=OUT/'frozen_before.json'
 if not before.exists():before.write_text(json.dumps(current,indent=2))
 old=json.load(open(before));assert old==current
 m=pd.read_csv(s.e.DOWNSTREAM/'train_metadata.csv',usecols=['original_row_index','patient_sn','csv_name','label_id','selected_by']);m=m[m.label_id<3]
 src=pd.read_csv(s.e.ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3/train_metadata.csv',usecols=['original_row_index','patient_sn','csv_name','label_id'])
 assert set(src.patient_sn)==set(m.patient_sn) and set(src.original_row_index)==set(m.original_row_index)
 first=src.drop_duplicates('original_row_index',keep='first').merge(m,on='original_row_index',suffixes=('_source','_current'));assert len(first)==3320 and first.csv_name_source.eq(first.csv_name_current).all() and first.label_id_source.eq(first.label_id_current).all()
 result=dict(historical_checkpoints_unchanged=True,checkpoint_count=len(paths),sha256=current,source_workbook_sha256=s.e.sha256(s.e.ROOT/'datasets/标签_Label.xlsx'),oof_probability_sha256=s.e.sha256(s.OUT/'oof_shared.npy'),fold_assignment_sha256=s.e.sha256(s.OUT/'fold_assignment.csv'),train_metadata_sha256=s.e.sha256(s.e.DOWNSTREAM/'train_metadata.csv'),train_labels_sha256=s.e.sha256(s.e.DOWNSTREAM/'train_labels.npy'),source_Train_ECGs=len(src),current_Train_patient_ECGs=len(m),first_source_ECG_matches_all=True,selected_by=m.selected_by.value_counts().to_dict(),no_training=True,no_additional_model_inference=True,no_Test_partition_read=True)
 (OUT/'final_readonly_audit.json').write_text(json.dumps(result,indent=2));print('final read-only audit passed')
if __name__=='__main__':main()
