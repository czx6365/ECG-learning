"""Bounded paired SID/context-teacher continuation pilot; not full JEPA replication."""
import sys,json,copy,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT),str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments')]
import run_e12_continuous_input as e
import run_e25 as e25
R=ROOT/'results/e28_context_teacher_pilot';SOURCE=e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth';STEPS=3;TRAIN_N=8192;VAL_N=1024

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=str))
def status(s):R.mkdir(parents=True,exist_ok=True);(R/'status.txt').write_text(s);print(s,flush=True)
def forward(model,ze,v,mask=None):
 b=model.backbone;h=b.input_proj(ze.float())*v[...,None]
 if mask is not None:h=torch.where(mask[...,None],b.mask_token.expand_as(h),h)
 h=torch.cat((b.cls_token[None,None].expand(len(h),1,-1),h),1);h=b.dropout(h+b.pos_embed);valid=torch.cat((torch.ones(len(h),1,dtype=torch.bool,device=h.device),v),1);selected=[]
 for depth,layer in enumerate(b.transformer.layers,1):
  h=e.masked_transformer_forward(SimpleNamespace(layers=[layer]),h,valid)
  if depth in (6,9,12):selected.append(b.norm_layer(h)[:,1:]*v[...,None])
 return selected
@torch.no_grad()
def teacher_target(teacher,z,v):
 h=forward(teacher,z,v);return F.layer_norm(torch.stack(h).mean(0).float(),(768,)).detach()
def predictor():return nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,768))
def cosine_loss(pred,target):return (2-2*F.cosine_similarity(pred.float(),target.float(),dim=-1)).mean()
def prepare():
 rng=np.random.default_rng(280006);tr=e.ZEDataset(e.DEFAULT_RUN/'pretrain_ze_cache','train');va=e.ZEDataset(e.DEFAULT_RUN/'pretrain_ze_cache','val');train_ids=np.sort(rng.choice(len(tr),min(TRAIN_N,len(tr)),replace=False));val_ids=np.sort(rng.choice(len(va),min(VAL_N,len(va)),replace=False));np.save(R/'train_indices.npy',train_ids);np.save(R/'val_indices.npy',val_ids)
 arrays={}
 for name,ds,ids in (('train',tr,train_ids),('val',va,val_ids)):
  arrays[name]=(np.array(ds.ze[ids]),np.array(ds.sid[ids]));assert arrays[name][0].shape[0]==len(ids)
 return arrays

def smoke(teacher,device,arrays):
 e.e4.set_seed(0);model=e.load_continuous_checkpoint(SOURCE,device);pred=predictor().to(device);model.eval();ze=torch.tensor(arrays['train'][0][:4],device=device,requires_grad=True);sid=torch.tensor(arrays['train'][1][:4],device=device).long();v=(sid>=0)&(sid<1024);gen=torch.Generator(device=device).manual_seed(28);mask=e.random_mask(v,.5,gen);assert mask.any()
 before=e.state_sha(teacher.state_dict());target=teacher_target(teacher,ze.detach(),v);h=forward(model,ze,v,mask)[-1];log=model.lm_head(h[mask]);changed=ze.detach().clone();changed[mask]=torch.randn_like(changed[mask])*1000
 with torch.no_grad():assert torch.equal(log,model.lm_head(forward(model,changed,v,mask)[-1][mask]))
 loss=F.cross_entropy(log.float(),sid[mask])+.1*cosine_loss(pred(h[mask]),target[mask]);loss.backward();assert ze.grad[mask].abs().max()==0;assert pred[0].weight.grad.abs().sum()>0;assert all(p.grad is None for p in teacher.parameters());assert e.state_sha(teacher.state_dict())==before
 # Confirm target is contextual: keep one token input fixed, change visible context.
 changed_context=ze.detach().clone();changed_context[mask]=0;target2=teacher_target(teacher,changed_context,v);visible=v&~mask;diff=float((target2[visible]-target[visible]).abs().mean());assert diff>1e-6
 dump(R/'smoke.json',{'passed':True,'masked_input_student_logits_invariant':True,'masked_input_student_gradient_exact_zero':True,'teacher_gradient_none':True,'teacher_full_state_unchanged':True,'predictor_gradient_nonzero':True,'context_target_change_on_unchanged_tokens':diff,'target_dim_std_mean':float(target[v].std(0).mean()),'test_loaded':False});del model,pred
@torch.no_grad()
def evaluate(model,pred,teacher,arrays,device):
 z,s=arrays;loss=0.;correct=count=0;cos=0.;targetstd=[]
 for start in range(0,len(z),64):
  zz=torch.tensor(z[start:start+64],device=device);ss=torch.tensor(s[start:start+64],device=device).long();v=(ss>=0)&(ss<1024);mask=torch.tensor(e.fixed_masks(s[start:start+64],start),device=device);target=teacher_target(teacher,zz,v);h=forward(model,zz,v,mask)[-1];log=model.lm_head(h[mask]).float();loss+=float(F.cross_entropy(log,ss[mask],reduction='sum'));count+=int(mask.sum());correct+=int((log.argmax(-1)==ss[mask]).sum());cos+=float(cosine_loss(pred(h[mask]),target[mask]))*int(mask.sum());targetstd.append(float(target[v].std(0).mean()))
 return {'sid_ce':loss/max(1,count),'sid_acc':correct/max(1,count),'context_cosine_loss':cos/max(1,count),'teacher_target_std':float(np.mean(targetstd))}
def train(method,teacher,arrays,device):
 out=R/method;out.mkdir(exist_ok=True)
 if (out/'PRETRAIN_COMPLETE').exists():return
 model=e.load_continuous_checkpoint(SOURCE,device);torch.manual_seed(280);pred=predictor().to(device);e.e4.set_seed(280);gen=torch.Generator(device=device).manual_seed(280006);opt=torch.optim.AdamW(list(model.parameters())+list(pred.parameters()),lr=1e-5,weight_decay=.05);rng=np.random.default_rng(280006);z,s=arrays['train'];history=[];best=float('inf');steps=len(z)//64;tb=e.state_sha(teacher.state_dict())
 model.eval();pred.eval();initial=evaluate(model,pred,teacher,arrays['val'],device)
 for epoch in range(1,STEPS+1):
  model.train();pred.train();order=rng.permutation(len(z));ce=[];aux=[]
  for step in range(steps):
   ii=order[step*64:(step+1)*64];zz=torch.tensor(z[ii],device=device);ss=torch.tensor(s[ii],device=device).long();v=(ss>=0)&(ss<1024);mask=e.random_mask(v,.5,gen)
   for g in opt.param_groups:g['lr']=e.lr_at((epoch-1)*steps+step,STEPS*steps,32,1e-5,1e-6,1e-6)
   opt.zero_grad(set_to_none=True)
   with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):target=teacher_target(teacher,zz,v)
   with torch.amp.autocast('cuda',dtype=torch.bfloat16):
    h=forward(model,zz,v,mask)[-1];log=model.lm_head(h[mask]);lc=F.cross_entropy(log.float(),ss[mask]);la=cosine_loss(pred(h[mask]),target[mask]);weight=.1*min(1.,((epoch-1)*steps+step+1)/steps) if method=='context_teacher' else 0.;loss=lc+weight*la
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(pred.parameters()),3.);opt.step();ce.append(float(lc));aux.append(float(la))
  model.eval();pred.eval();val=evaluate(model,pred,teacher,arrays['val'],device);history.append({'epoch':epoch,'train_sid_ce':float(np.mean(ce)),'train_context_loss':float(np.mean(aux)),'val':val});dump(out/'history.json',history)
  if val['sid_ce']<best:
   best=val['sid_ce'];torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':e.model_config(),'epoch':epoch,'best_val_loss':best},out/'checkpoint-best.pth');torch.save(pred.state_dict(),out/'predictor-best.pth')
  print(method,epoch,val,flush=True)
 assert tb==e.state_sha(teacher.state_dict());dump(out/'pretrain_summary.json',{'method':method,'initial':initial,'history':history,'best_sid_ce':best,'selection':'strict pretrain Val SID CE only','training_records':len(z),'epochs':STEPS,'updates':steps*STEPS,'teacher_frozen':True,'pilot_not_full_pretraining':True});(out/'PRETRAIN_COMPLETE').touch()
@torch.inference_mode()
def features(method,device):
 base=R/method;out=base/'features';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 tok=e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False);model=e.load_continuous_checkpoint(base/'checkpoint-best.pth',device).eval().requires_grad_(False);before=(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()))
 for split in ('train','val'):
  w=np.load(e.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');c=np.load(e.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');t=np.load(e.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r');dest=np.lib.format.open_memmap(out/f'{split}.npy',mode='w+',dtype=np.float32,shape=(len(w),3,1536))
  for start in range(0,len(w),32):
   xx=torch.tensor(np.array(w[start:start+32]),device=device).float();ch=torch.tensor(np.array(c[start:start+32]),device=device).long();ti=torch.tensor(np.array(t[start:start+32]),device=device).long();v=tok.valid_mask(xx,ch,ti);ze=tok.encode_to_quantizer_features(xx,ch,ti).float();h=forward(model,ze,v);dest[start:start+len(xx)]=torch.stack([e25.pool(z,v) for z in h],1).cpu().numpy()
  dest.flush()
 assert before==(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()));dump(out/'audit.json',{'frozen_states_unchanged':True,'test_loaded':False});(out/'COMPLETE').touch()
def report():
 rows={}
 for method in ('sid_only','context_teacher'):
  rr=[json.loads((R/method/'downstream/layer12'/f'seed{s}/summary.json').read_text()) for s in (0,1,2)];agg={k:{'mean':float(np.mean([r['val'][k] for r in rr])),'sd':float(np.std([r['val'][k] for r in rr],ddof=1))} for k in ('r2','patient_mean_r2','macro_roc_auc','macro_f1')};agg['per_output_r2']={k:float(np.mean([r['val']['per_output_r2'][k] for r in rr])) for k in rr[0]['val']['per_output_r2']};rows[method]={'aggregate':agg,'per_seed':rr,'pretrain':json.loads((R/method/'pretrain_summary.json').read_text())}
 rows['paired_downstream_r2_delta']=[rows['context_teacher']['per_seed'][s]['val']['r2']-rows['sid_only']['per_seed'][s]['val']['r2'] for s in range(3)];rows['scope']='3 epochs,8192 fixed strict-train ECGs,1 pretraining seed,all downstream Train/Val. Not a full-budget test of JEPA.';dump(R/'validation_summary.json',rows)
def main():
 R.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);device=torch.device('cuda');sources=[SOURCE,e.TOKENIZER,Path(__file__)];before={str(p):e.sha256(p) for p in sources};dump(R/'protocol.json',{'methods':['sid_only','context_teacher'],'initialization':'same E12-A best checkpoint, not random-from-scratch','strict_train_subset':TRAIN_N,'strict_val_subset':VAL_N,'epochs':STEPS,'updates':STEPS*(TRAIN_N//64),'batch':64,'lr_peak':1e-5,'teacher':'fixed E12-A full ECG, eval, stopgrad; mean normalized contextual layers6/9/12','student':'50% masked z_e, all Transformer weights trainable; original SID CE weight1','auxiliary':'0.1*(2-2cos) with first-epoch linear warmup; predictor768->256->768 discarded downstream','selection':'pretrain fixed-mask strict Val SID CE only; downstream Val Overall R2','downstream':'original layer12+E11,30epochs,seed0/1/2','test_loaded':False,'source_sha256':before,'limitation':'bounded continuation pilot, not a full-budget or multi-pretrain-seed JEPA experiment'});arrays=prepare();teacher=e.load_continuous_checkpoint(SOURCE,device).eval().requires_grad_(False);status('smoke');smoke(teacher,device,arrays)
 for method in ('sid_only','context_teacher'):
  status('pretrain_'+method);train(method,teacher,arrays,device)
 del teacher
 for method in ('sid_only','context_teacher'):
  status('features_'+method);features(method,device);data=e25.data(R/method,device)
  for seed in (0,1,2):status(f'downstream_{method}_{seed}');e25.train(R/method,data,device,'layer12',seed)
 status('report');report();assert all(e.sha256(Path(p))==v for p,v in before.items());dump(R/'verification.json',{'source_hashes_unchanged':True,'test_loaded':False});status('complete');(R/'COMPLETE').touch()
if __name__=='__main__':main()
