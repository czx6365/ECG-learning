import json,hashlib,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,f1_score
import run_e14_association_audit as e
p=e.RUN;metrics=json.loads((p/'Control_group_E7_metrics.json').read_text());case=pd.read_csv(p/'Control测试组逐例诊断.csv');labels=np.load(e.e4.FEATURES/'test_labels.npy').argmax(1)
for row in metrics:
 raw=np.load(e.E7/f"CE_seed{row['seed']}/test_raw_outputs.npy");pr=raw.argmax(1);z=np.exp(raw-raw.max(1,keepdims=True));s=z/z.sum(1,keepdims=True);indices=case.loc[case.group==row['group'],'parent_test_row'].to_numpy();positive=np.zeros(len(labels),bool);positive[indices]=True;assert positive.sum()==row['positive_records'] and np.isclose((pr[positive]==2).mean(),row['control_recall'])
 for name,neg in [('all_nonControl',labels!=2),('patient_nonControl',(labels<3)&(labels!=2))]:
  mask=positive|neg;assert np.isclose(roc_auc_score(positive[mask],s[mask,2]),row[name]['control_ovr_auc']);assert np.isclose(f1_score(positive[mask],pr[mask]==2),row[name]['control_f1'])
 assert sum(row['predicted_counts'].values())==len(indices)
summary=json.loads((p/'association_summary.json').read_text());assert summary['current_association_mismatch_count']==summary['current_tensor_parent_mismatch_count']==0 and summary['current_rows']==4675
w=json.loads((p/'raw_wave_to_saved_wave_validation.json').read_text());assert all(x['mismatch_count']==0 for x in w['checks']) and sum(x['selected_rows_checked'] for x in w['checks'])==4675 and w['parent_wave_sources_hashes_unchanged']
raw=pd.read_excel(e.RAW);cols=list(raw.columns)
def excelcol(n):
 s=''
 while n:n,r=divmod(n-1,26);s=chr(65+r)+s
 return s
lc=excelcol(cols.index('标签')+1);ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(e.RAW) as z:
 sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));cells=[c for c in sheet.findall('.//s:c',ns) if c.attrib.get('r','').startswith(lc) and ''.join(x for x in c.attrib['r'] if x.isalpha())==lc and int(''.join(x for x in c.attrib['r'] if x.isdigit()))>1];formulas=sum(c.find('s:f',ns) is not None for c in cells)
labelinfo={'sheet':'Sheet1','label_column':lc,'label_cell_range':f'{lc}2:{lc}{len(raw)+1}','label_formula_cell_count':formulas,'source_field_column_ranges':{key:excelcol(cols.index(key)+1)+'2:'+excelcol(cols.index(key)+1)+str(len(raw)+1) for key in ['patient_sn','标签','file_name','心电图时间（日期）','心电图时间（时刻）','冠脉造影时间']},'label_meaning':'Static provided label values, not an Excel formula-derived criterion in this workbook; original clinical assignment rule unavailable'}
e.save('original_workbook_field_locations.json',labelinfo)
v=json.loads((p/'validation.json').read_text());v.update({'subgroup_metrics_independently_recomputed':True,'real_CSV_to_saved_wave_all4675_exact_match':True,'supplemental_source_hashes':{str(Path(__file__).with_name(n)):e.e4.sha(Path(__file__).with_name(n)) for n in ['e14_duplicate_and_wave_trace.py','e14_temporal_audit.py','validate_e14.py']},'raw_original_label_cells_inspected':True});e.save('report_validation.json',v);print('E14 full evidence validation passed',labelinfo)
