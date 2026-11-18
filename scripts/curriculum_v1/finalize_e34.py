"""Decision explanation and final artifact manifest."""
import json
from pathlib import Path
from run_e34 import OUT,e,s

def main():
 assert (OUT/'status.txt').read_text().strip()=='complete'
 d=json.load(open(OUT/'ensemble_metrics.json'));boot=json.load(open(OUT/'paired_bootstrap.json'));v=json.load(open(OUT/'decision.json'))
 a,c,z=d['A'],d['C'],d['D']
 decision='本次未发现任何局部窗口满足预定的稳定独立临床增量标准，保留原E29 Shared Ensemble3，不进入正式多尺度模型开发，不扩大CNN或继续搜索窗口。' if not any(x['stable_clinical_increment'] for x in v.values()) else '存在待独立确认的候选；本次不自动开展后续实验。'
 analysis=f"{decision}\n\nR后代理窗口C的条件OMI/NOMI AUC为{c['omi_nomi_auc']:.6f}，最接近原E29的{a['omi_nomi_auc']:.6f}，但差值{c['omi_nomi_auc']-a['omi_nomi_auc']:+.6f}，并无正向增量。扩展心拍D的Overall差{z['overall']-a['overall']:+.6f}，同时Clinical-3差{z['clinical3']-a['clinical3']:+.6f}、条件AUC差{z['omi_nomi_auc']-a['omi_nomi_auc']:+.6f}。D的Healthy列R²差为{z['Healthy']-a['Healthy']:+.6f}；按Overall=0.75×Clinical-3+0.25×Healthy分解，总体微增来自Healthy，不能视作患者临床增量。\n\nD即使相对额外容量对照有改善，也仍须首先超过原E29；容量对照自身退化不能降低增量判定门槛。本次负结果限于固定100Hz代理窗口和此轻量残差训练，不能证明真实500Hz波形或临床ST/T边界没有信息。"
 report=OUT/'E34_PILOT_REPORT_CN.md';text=report.read_text();marker='本次固定四个新增模块对照均完成三折×三seed×30轮。'
 assert marker in text;text=text.replace(marker,analysis+'\n\n'+marker,1)
 text+='\n最终数值与分组审计见independent_audit.json；6个重建的嵌套fold均通过patient_sn及entity_id双重不重叠检查。原始R峰检测的历史来源与模型代码保留在code/provenance/；历史E32 checkpoint未包含完整训练行清单，其原始脚本、现有manifest路径/大小和重建划分一致，这一来源证据局限记录于独立审计。\n'
 report.write_text(text)
 files=[p for p in OUT.rglob('*') if p.is_file() and 'cache' not in p.name]
 s.b.dump(OUT/'artifact_manifest.json',dict(checkpoints=len(list(OUT.glob('fold*/?/seed*/checkpoint.pth'))),cnn_parameters_each=4532,cnn_seeds=[0,1,2],original_model_untouched=True,report=str(report),raw_cache_policy='continuous/window caches and checkpoints preserved remote; no unnecessary patient IDs exported',file_count=len(files)))
 print('finalized')
if __name__=='__main__':main()
