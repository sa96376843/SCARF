#!/usr/bin/env python3
"""Generate all manuscript figures, tables, and paired bootstrap statistics from raw results."""
from pathlib import Path
import json,warnings,os
import numpy as np,pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
 import pyarrow.parquet as pq
except ImportError:
 pq=None
try:
 from run_experiments import psi_matrix,TRAIN_MONTHS,CAL_MONTH,TEST_MONTHS
except ImportError:
 TRAIN_MONTHS=['2019-08'];CAL_MONTH='2019-09'
 TEST_MONTHS=[f'2019-{m:02d}' for m in range(10,13)]+[f'2020-{m:02d}' for m in range(1,10)]
 psi_matrix=None

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';TAB=ROOT/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
COL={'Static':'#6B7280','Random':'#0072B2','Uncertainty':'#E69F00','RecentOnly':'#CC79A7','SCARF':'#009E73',
     'SCARF-NoEnsemble':'#56B4E9','SCARF-NoSelfTraining':'#D55E00','SCARF-UniformQueryWeight':'#F0E442','SCARF-NoCalibration':'#7A5195','SCARF-NoReplay':'#8C564B'}
MONTH_SHORT=['Oct\n2019','Nov','Dec','Jan\n2020','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.7,'lines.linewidth':1.5,
 'figure.dpi':160,'savefig.dpi':400,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5})
CMAP=LinearSegmentedColormap.from_list('drift',['#F7FBFF','#C6DBEF','#6BAED6','#2171B5','#08306B'])

def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=300);plt.close(fig)
def tex_table(df,name,caption,label,float_format='%.4f'):
 s=df.to_latex(index=False,escape=False,float_format=lambda x:float_format%x,longtable=False)
 s=s.replace('%',r'\%')
 (TAB/(name+'.tex')).write_text(s);df.to_csv(TAB/(name+'.csv'),index=False)
def bootstrap_table(frame):
 out=frame.copy();out.columns=['Comparison','$n$','Mean difference','CI low','CI high','$p$-value']
 out['$p$-value']=out['$p$-value'].map(lambda p:'$<$0.0001' if p<0.0001 else f'{p:.4f}')
 return out

def bootstrap_differences(r,months=None):
 r=r[r.budget==100]
 if months is not None:r=r[r.month.isin(months)]
 rng=np.random.default_rng(20261024);months=sorted(r.month.unique());out=[]
 base=r[r.method=='SCARF'].set_index(['seed','month'])
 for other in ['Static','Random','Uncertainty','RecentOnly']:
  comp=r[r.method==other].set_index(['seed','month'])
  common=base.index.intersection(comp.index)
  frame=pd.DataFrame({'a':base.loc[common,'tpr_at_1fpr'].to_numpy(),'b':comp.loc[common,'tpr_at_1fpr'].to_numpy()},index=common)
  frame['d']=frame.a-frame.b;by_month=[z.d.to_numpy() for _,z in frame.groupby(level='month')]
  d=frame.d.to_numpy();n=len(d);boots=[]
  for _ in range(10000):
   chosen=rng.integers(0,len(by_month),len(by_month));boots.append(np.concatenate([by_month[j] for j in chosen]).mean())
  lo,hi=np.quantile(boots,[.025,.975]);p=2*min(np.mean(np.array(boots)<=0),np.mean(np.array(boots)>=0))
  out.append(dict(comparison='SCARF $-$ '+other,n=n,mean_difference=d.mean(),ci_low=lo,ci_high=hi,p_value=p))
 return pd.DataFrame(out)

def main():
 r=pd.read_csv(RES/'monthly_metrics.csv');s=pd.read_csv(RES/'static_metrics.csv');s['seed']=13;s.loc[s.index,'seed']=13
 # Replicate the deterministic static row for each seed solely for paired calculations.
 static=pd.concat([s.assign(seed=z,budget=100) for z in [13,42,87]],ignore_index=True)
 r=pd.concat([r,static],ignore_index=True);a=pd.read_csv(RES/'ablation_metrics.csv')
 d=pd.read_csv(RES/'drift_metrics.csv');g=pd.read_csv(RES/'group_metrics.csv');q=pd.read_csv(RES/'selected_samples.csv')
 order=['Static','Random','Uncertainty','RecentOnly','SCARF']
 # Dataset audit and monthly counts.
 bodmas_path=os.environ.get('BODMAS_PARQUET','/tmp/bodmas/bodmas-timestamps.parquet')
 cache_path=os.environ.get('SCARF_CACHE','/tmp/scarf_cache')
 if pq is not None and Path(bodmas_path).exists():
  meta=pq.read_table(bodmas_path,columns=['Label','family','timestamp','category']).to_pandas()
  meta['timestamp']=pd.to_datetime(meta.timestamp,utc=True,format='mixed');meta['month']=meta.timestamp.dt.to_period('M').astype(str)
  mc=meta[meta.month.isin(TEST_MONTHS)].groupby('month').Label.agg(['count','sum']).reindex(TEST_MONTHS)
  mc['benign']=mc['count']-mc['sum'];mc['malware_rate']=mc['sum']/mc['count']
  ds=pd.DataFrame({'Month':TEST_MONTHS,'Total':mc['count'].values,'Malware':mc['sum'].astype(int).values,'Benign':mc.benign.astype(int).values,'Malware rate':mc.malware_rate.values})
  tex_table(ds,'dataset_months','Monthly BODMAS evaluation population','tab:data',float_format='%.3f')
 else:
  # Deterministic regeneration from the saved audit table when the optional Parquet reader is unavailable.
  ds=pd.read_csv(TAB/'dataset_months.csv').set_index('Month').reindex(TEST_MONTHS)
  mc=pd.DataFrame({'count':ds.Total,'sum':ds.Malware,'benign':ds.Benign,'malware_rate':ds['Malware rate']},index=TEST_MONTHS)
  tex_table(ds.reset_index(),'dataset_months','Monthly BODMAS evaluation population','tab:data',float_format='%.3f')
  meta=None
 # Fig. 2: aligned monthly evidence matrix (raw values annotated; color normalized by row).
 dd=d[(d.method=='SCARF')&(d.seed==42)&(d.budget==100)].set_index('month').reindex(TEST_MONTHS)
 x=np.arange(12)
 raw=np.vstack([mc['count'].to_numpy(),dd.mean_psi.to_numpy(),dd.risk_q90.to_numpy(),mc.malware_rate.to_numpy()])
 norm=(raw-raw.min(axis=1,keepdims=True))/(raw.max(axis=1,keepdims=True)-raw.min(axis=1,keepdims=True)+1e-12)
 fig,ax=plt.subplots(figsize=(7.1,2.25));im=ax.imshow(norm,aspect='auto',cmap='YlGnBu',vmin=0,vmax=1)
 ax.set_xticks(x,MONTH_SHORT);ax.set_yticks(range(4),['Sample count','Mean feature PSI','Margin uncertainty (q90)','Malware prevalence'])
 for i in range(4):
  for j in range(12):
   text=f'{raw[i,j]:,.0f}' if i==0 else (f'{100*raw[i,j]:.1f}%' if i==3 else f'{raw[i,j]:.3f}')
   ax.text(j,i,text,ha='center',va='center',fontsize=6,color='white' if norm[i,j]>.58 else '#17202A')
 ax.set_xlabel('Evaluation month');ax.grid(False)
 cb=fig.colorbar(im,ax=ax,pad=.015);cb.set_label('Within-row relative level (min--max)')
 save(fig,'fig02_temporal_drift')
 # Per-feature drift matrix (real unlabeled month statistics).
 cache=ROOT/'cache_local'
 # Use the scratch memmap when available; a small manifest is saved for portability.
 feature_cache=Path(cache_path)
 if (feature_cache/'X384.npy').exists() and (feature_cache/'base_features.json').exists():
  X=np.load(feature_cache/'X384.npy',mmap_mode='r');base=json.loads((feature_cache/'base_features.json').read_text())
  if meta is not None:
   refidx=np.flatnonzero(meta.month==CAL_MONTH);ref=np.asarray(X[refidx]);fm=[]
   for m in TEST_MONTHS:
    idx=np.flatnonzero(meta.month==m);fm.append(psi_matrix(ref,np.asarray(X[idx])))
   fm=np.asarray(fm);np.savetxt(RES/'feature_drift_matrix.csv',fm,delimiter=',')
  else:
   fm=np.loadtxt(RES/'feature_drift_matrix.csv',delimiter=',')
  top=np.argsort(fm.mean(0))[-20:];labels=[f'F{base[j]+1}' for j in top]
  fig,ax=plt.subplots(figsize=(7.1,4.1));im=ax.imshow(fm[:,top].T,aspect='auto',cmap=CMAP,norm=mpl.colors.PowerNorm(.55))
  ax.set_xticks(x,MONTH_SHORT);ax.set_yticks(np.arange(20),labels);ax.set_ylabel('Model-relevant PE feature');ax.set_xlabel('Evaluation month')
  cb=fig.colorbar(im,ax=ax,pad=.015);cb.set_label('Population stability index');save(fig,'fig03_feature_drift_heatmap')
 # Main table, seed 42 B=100.
 main=r[(r.seed==42)&(r.budget==100)&r.method.isin(order)].groupby('method')[['auc','tpr_at_1fpr','f1','tpr','fpr','brier','ece']].mean().reindex(order)
 mt=main.reset_index().rename(columns={'method':'Method','auc':'AUC','tpr_at_1fpr':'TPR@1%FPR','f1':'F1','tpr':'TPR','fpr':'FPR','brier':'Brier','ece':'ECE'})
 tex_table(mt,'main_results','Monthly macro-averaged performance (seed 42, 100 labels/month)','tab:main')
 # Fig. 4: month-by-method performance matrices replace overlapping line traces.
 principal=r[(r.seed==42)&(r.budget==100)&r.method.isin(order)]
 fig,ax=plt.subplots(1,2,figsize=(7.1,2.65))
 for k,(axx,metric,title,cmap) in enumerate(zip(ax,['tpr_at_1fpr','fpr'],['TPR at ROC-interpolated 1% FPR','Realized threshold FPR'],['YlGnBu','magma'])):
  mat=principal.pivot_table(index='method',columns='month',values=metric).reindex(order)[TEST_MONTHS]
  im=axx.imshow(mat,aspect='auto',cmap=cmap)
  if k==0:axx.set_yticks(np.arange(len(order)),order)
  else:axx.set_yticks([])
  axx.set_xticks(x,['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'],rotation=45,ha='right')
  axx.set_title(title,fontsize=8);axx.grid(False);fig.colorbar(im,ax=axx,pad=.035,shrink=.86)
 ax[0].set_ylabel('Maintenance policy');ax[0].set_xlabel('Evaluation month');ax[1].set_xlabel('Evaluation month')
 fig.subplots_adjust(wspace=.22)
 save(fig,'fig04_temporal_performance')
 # Multi-seed summary.
 ms=r[(r.budget==100)&r.method.isin(order)].groupby(['seed','method'])[['auc','tpr_at_1fpr','f1','fpr','brier']].mean().groupby('method').agg(['mean','std']).reindex(order)
 flat=pd.DataFrame({'Method':order})
 for metric,col in [('auc','AUC'),('tpr_at_1fpr','TPR@1%FPR'),('f1','F1'),('fpr','FPR'),('brier','Brier')]:
  flat[col]=[f'{ms.loc[m,(metric,"mean")]:.4f} $\\pm$ {ms.loc[m,(metric,"std")]:.4f}' for m in order]
 tex_table(flat,'multiseed_results','Mean and standard deviation across seeds and months (100 labels/month)','tab:seeds')
 # Locked late-month check, separated from the six algorithm-development months.
 late=r[(r.budget==100)&r.month.isin(TEST_MONTHS[6:])&r.method.isin(order)]
 latemean=late.groupby('method')[['auc','tpr_at_1fpr','f1','fpr','brier']].mean().reindex(order)
 latetab=latemean.reset_index().rename(columns={'method':'Method','auc':'AUC','tpr_at_1fpr':'TPR@1%FPR','f1':'F1','fpr':'FPR','brier':'Brier'})
 tex_table(latetab,'locked_results','Locked late-month check across seeds (100 labels/month)','tab:locked')
 latebs=bootstrap_differences(r,TEST_MONTHS[6:]);latebst=bootstrap_table(latebs)
 tex_table(latebst,'locked_bootstrap','Month-clustered paired bootstrap on the locked late-month subset','tab:lockedbootstrap')
 (RES/'locked_bootstrap_tests.json').write_text(latebs.to_json(orient='records',indent=2))
 # Fig. 5 drift-response phase portrait.
 fig,ax=plt.subplots(figsize=(3.55,3.1))
 for method in ['Random','SCARF']:
  z=r[(r.seed==42)&(r.budget==100)&(r.method==method)].set_index('month').reindex(TEST_MONTHS)
  delta=z.tpr_at_1fpr-r[(r.seed==42)&(r.method=='Static')].set_index('month').reindex(TEST_MONTHS).tpr_at_1fpr
  ax.plot(dd.mean_psi,delta,color=COL[method],alpha=.35);sc=ax.scatter(dd.mean_psi,delta,c=x,cmap='viridis',s=30,label=method,edgecolor='white',linewidth=.4)
 ax.axhline(0,color='black',lw=.8,ls=':');ax.set_xlabel('Mean feature PSI');ax.set_ylabel('$\\Delta$ TPR@1%FPR vs. static');ax.legend(frameon=False)
 cb=fig.colorbar(sc,ax=ax);cb.set_label('Month index');save(fig,'fig05_drift_response_phase')
 # Budget table and efficiency frontier.
 bu=r[r.method.isin(order)].groupby(['method','budget'])[['tpr_at_1fpr','auc','f1','fpr','brier']].mean().reset_index()
 bt=bu.rename(columns={'method':'Method','budget':'Labels/month','tpr_at_1fpr':'TPR@1%FPR','auc':'AUC','f1':'F1','fpr':'FPR','brier':'Brier'})
 tex_table(bt,'budget_results','Label-budget sensitivity (monthly macro averages)','tab:budget')
 fig,ax=plt.subplots(figsize=(3.55,3.0))
 for method in ['Random','Uncertainty','RecentOnly','SCARF']:
  z=bu[bu.method==method].sort_values('budget')
  ax.scatter(z.budget*12,z.tpr_at_1fpr,s=18+900*z.fpr,color=COL[method],alpha=.72,label=method,edgecolor='white',linewidth=.5)
 ax.set_xlabel('Cumulative analyst labels (12 months)');ax.set_ylabel('Mean TPR@1%FPR');ax.legend(frameon=False,ncol=2)
 save(fig,'fig06_label_efficiency_frontier')
 # Family preservation-adaptation plane.
 gg=g[(g.budget==100)&g.group.isin(['Known family','Novel family'])].groupby(['method','group']).tpr.mean().unstack()
 gt=gg.reset_index().rename(columns={'method':'Method','Known family':'Known-family TPR','Novel family':'Novel-family TPR'})
 tex_table(gt,'family_results','Known-family retention and novel-family adaptation','tab:family')
 fig,ax=plt.subplots(figsize=(3.55,3.0))
 for method,z in gg.iterrows():
  if method not in COL:continue
  ax.scatter(z['Known family'],z['Novel family'],s=65,color=COL[method],label=method,edgecolor='white',linewidth=.7)
  ax.annotate(method,(z['Known family'],z['Novel family']),xytext=(4,4),textcoords='offset points',fontsize=6)
 ax.set_xlabel('Known-family TPR (retention)');ax.set_ylabel('Novel-family TPR (adaptation)');ax.grid(alpha=.25);save(fig,'fig07_family_adaptation_plane')
 # Calibration mosaic.
 cm=r[(r.seed==42)&(r.budget==100)&r.method.isin(order)]
 fig,ax=plt.subplots(1,2,figsize=(7.1,2.55))
 month_labels=['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']
 for axx,metric,cmap in zip(ax,['ece','brier'],['viridis','magma']):
  mat=cm.pivot_table(index='method',columns='month',values=metric).reindex(order)[TEST_MONTHS]
  im=axx.imshow(mat,aspect='auto',cmap=cmap);axx.set_yticks(np.arange(len(order)),order)
  axx.set_xticks(x,month_labels,rotation=45,ha='right');axx.set_title(metric.upper());axx.grid(False)
  axx.set_xlabel('Evaluation month (October 2019–September 2020)')
  fig.colorbar(im,ax=axx,pad=.015)
 save(fig,'fig08_calibration_mosaic')
 # Ablation table; the consolidated results table carries this information without a redundant parallel-coordinate plot.
 ab=a.groupby('method')[['auc','tpr_at_1fpr','f1','tpr','fpr','brier','ece']].mean()
 ab.loc['SCARF']=main.loc['SCARF'];ab=ab.sort_values('tpr_at_1fpr',ascending=False)
 at=ab.reset_index().rename(columns={'method':'Variant','auc':'AUC','tpr_at_1fpr':'TPR@1%FPR','f1':'F1','fpr':'FPR','brier':'Brier','ece':'ECE'})
 tex_table(at[['Variant','AUC','TPR@1%FPR','F1','FPR','Brier','ECE']],'ablation_results','Component-wise ablation at 100 labels/month','tab:ablation')
 # Bootstrap table and raincloud.
 bs=bootstrap_differences(r);bst=bootstrap_table(bs)
 tex_table(bst,'bootstrap_tests','Paired bootstrap comparisons of monthly TPR at 1% FPR','tab:bootstrap')
 (RES/'bootstrap_tests.json').write_text(bs.to_json(orient='records',indent=2))
 fig,ax=plt.subplots(figsize=(7.1,2.8));rng=np.random.default_rng(4)
 r100=r[r.budget==100];scarf=r100[r100.method=='SCARF'].set_index(['seed','month'])
 for k,other in enumerate(['Static','Random','Uncertainty','RecentOnly']):
  comp=r100[r100.method==other].set_index(['seed','month']);idx=scarf.index.intersection(comp.index)
  delta=scarf.loc[idx,'tpr_at_1fpr'].to_numpy()-comp.loc[idx,'tpr_at_1fpr'].to_numpy()
  parts=ax.violinplot([delta],positions=[k],widths=.65,showextrema=False)
  for body in parts['bodies']:body.set_facecolor(COL[other]);body.set_alpha(.45)
  ax.scatter(np.full(len(delta),k)+rng.normal(0,.035,len(delta)),delta,s=9,color=COL[other],alpha=.45)
  ax.scatter(k,delta.mean(),marker='D',s=35,color='black',zorder=5)
 ax.axhline(0,color='black',ls=':',lw=1);ax.set_xticks(range(4),['Static','Random','Uncertainty','RecentOnly']);ax.set_ylabel('Paired monthly $\\Delta$ TPR@1%FPR');save(fig,'fig10_bootstrap_raincloud')
 # Query composition, computed from actual analyst selections.
 qq=q.groupby(['method','month','budget','seed']).label.mean().reset_index()
 z=qq[(qq.seed==42)&(qq.budget==100)&qq.method.isin(['Random','Uncertainty','RecentOnly','SCARF'])]
 qorder=['Random','Uncertainty','RecentOnly','SCARF'];mat=z.pivot_table(index='method',columns='month',values='label').reindex(qorder)[TEST_MONTHS]
 fig,ax=plt.subplots(figsize=(3.55,2.2));im=ax.imshow(mat,aspect='auto',cmap='PuRd',vmin=0,vmax=1)
 ax.set_xticks(x,['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'],rotation=45,ha='right')
 ax.set_yticks(range(4),qorder);ax.set_xlabel('Evaluation month');ax.grid(False)
 cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label('Malicious query fraction');save(fig,'fig11_query_composition')
 # Numeric macros keep the abstract and discussion synchronized with raw results.
 if meta is not None:
  counts={'TrainN':int(meta.month.isin(TRAIN_MONTHS).sum()),'CalN':int((meta.month==CAL_MONTH).sum()),
   'TestN':int(meta.month.isin(TEST_MONTHS).sum()),'TotalN':len(meta)}
 else:
  import re
  previous=(ROOT/'result_macros.tex').read_text()
  counts={k:int(float(re.search(r'\\newcommand{\\'+k+r'}\{([^}]+)\}',previous).group(1))) for k in ['TrainN','CalN','TestN','TotalN']}
 macro={
  'ScarfTPR':main.loc['SCARF','tpr_at_1fpr'],'StaticTPR':main.loc['Static','tpr_at_1fpr'],
  'RandomTPR':main.loc['Random','tpr_at_1fpr'],'UncertaintyTPR':main.loc['Uncertainty','tpr_at_1fpr'],
  'ScarfAUC':main.loc['SCARF','auc'],'UncertaintyAUC':main.loc['Uncertainty','auc'],
  'StaticAUC':main.loc['Static','auc'],'ScarfBrier':main.loc['SCARF','brier'],'StaticBrier':main.loc['Static','brier'],
  'ScarfNovel':gg.loc['SCARF','Novel family'],'ScarfKnown':gg.loc['SCARF','Known family'],
  'LateScarfTPR':latemean.loc['SCARF','tpr_at_1fpr'],'LateUncertaintyTPR':latemean.loc['Uncertainty','tpr_at_1fpr'],
  'LateRandomTPR':latemean.loc['Random','tpr_at_1fpr'],'LateScarfAUC':latemean.loc['SCARF','auc'],
  'LateUncertaintyAUC':latemean.loc['Uncertainty','auc'],
  **counts}
 macro['DeltaStaticPP']=100*(macro['ScarfTPR']-macro['StaticTPR']);macro['DeltaRandomPP']=100*(macro['ScarfTPR']-macro['RandomTPR'])
 macro['DeltaUncertaintyPP']=100*(macro['ScarfTPR']-macro['UncertaintyTPR'])
 macro['LateDeltaUncertaintyPP']=100*(macro['LateScarfTPR']-macro['LateUncertaintyTPR'])
 macro['LateDeltaRandomPP']=100*(macro['LateScarfTPR']-macro['LateRandomTPR'])
 allmean=r[(r.budget==100)&r.method.isin(order)].groupby('method')[['auc','tpr_at_1fpr','f1','fpr','brier']].mean()
 macro['AllScarfTPR']=allmean.loc['SCARF','tpr_at_1fpr'];macro['AllUncertaintyTPR']=allmean.loc['Uncertainty','tpr_at_1fpr']
 macro['AllDeltaUncertaintyPP']=100*(macro['AllScarfTPR']-macro['AllUncertaintyTPR'])
 for other,short in [('Static','Static'),('Random','Random'),('Uncertainty','Uncertainty'),('RecentOnly','Recent')]:
  z=bs[bs.comparison=='SCARF $-$ '+other].iloc[0]
  macro['CI'+short+'Low']=z.ci_low;macro['CI'+short+'High']=z.ci_high;macro['P'+short]=z.p_value
 lines=[]
 for k,v in macro.items():
  val=f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
  lines.append('\\newcommand{\\'+k+'}{'+val+'}\n')
 (ROOT/'result_macros.tex').write_text(''.join(lines))
 # Protocol table and concise result manifest.
 prot=pd.DataFrame([
  ['Initial training','August 2019 only'],['Calibration','September 2019 only'],['Evaluation','October 2019--September 2020'],
  ['Base / retained features','384 / 384'],['Monthly label budgets','25, 50, 100, 200'],['Seeds','13, 42, 87'],
  ['Operating target','1% calibration FPR'],['Update rule','Post-evaluation only'],['Base learner','Baselines: 120 trees; SCARF: 3 x 240 trees']],columns=['Protocol element','Value'])
 tex_table(prot,'protocol','Frozen experimental protocol','tab:protocol')
 print('Generated',len(list(FIG.glob('*.pdf'))),'PDF figures and',len(list(TAB.glob('*.tex'))),'LaTeX tables')
if __name__=='__main__':
 warnings.filterwarnings('ignore');main()
