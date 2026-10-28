#!/usr/bin/env python3
"""Generate extended comparison, sensitivity, efficiency, and significance artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results'; FIG = ROOT/'figures'; TAB = ROOT/'tables'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
ORDER = ['Static','LogisticRegression','RandomForest','ExtraTrees','Random','Uncertainty',
         'RecentOnly','ReservoirReplay','ExpDecayReplay','EnsembleMargin','QBC','SCARF']
DISPLAY = {'Static':'Static','LogisticRegression':'Logistic','RandomForest':'RandomForest',
    'ExtraTrees':'ExtraTrees','Random':'Random replay','Uncertainty':'Uncertainty',
    'RecentOnly':'RecentOnly','ReservoirReplay':'Reservoir','ExpDecayReplay':'ExpDecay',
    'EnsembleMargin':'Ens-Margin','QBC':'QBC','SCARF':'SCARF'}
METRICS = [('auc','AUC',True),('tpr_at_1fpr','TPR@1\\%FPR',True),('f1','F1',True),
           ('fpr','FPR',False),('brier','Brier',False)]
COL = {'Static':'#6B7280','LogisticRegression':'#7A5195','RandomForest':'#8C564B',
    'ExtraTrees':'#BCBD22','Random':'#0072B2','Uncertainty':'#E69F00','RecentOnly':'#CC79A7',
    'ReservoirReplay':'#56B4E9','ExpDecayReplay':'#D55E00','EnsembleMargin':'#F0E442',
    'QBC':'#9467BD','SCARF':'#009E73'}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.linewidth':.7,'lines.linewidth':1.5,'figure.dpi':160,'savefig.dpi':400,
    'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,
    'grid.alpha':.18,'grid.linewidth':.5})


def save(fig, name):
    fig.savefig(FIG/(name+'.pdf')); fig.savefig(FIG/(name+'.png'), dpi=300); plt.close(fig)


def paired_bootstrap(r, other, replicates=10000):
    r = r[r.budget == 100]
    a = r[r.method == 'SCARF'].set_index(['seed','month'])
    b = r[r.method == other].set_index(['seed','month'])
    idx = a.index.intersection(b.index)
    d = (a.loc[idx,'tpr_at_1fpr'].to_numpy()-b.loc[idx,'tpr_at_1fpr'].to_numpy())
    frame = pd.DataFrame({'d':d}, index=idx)
    by_month = [z.d.to_numpy() for _,z in frame.groupby(level='month')]
    rng = np.random.default_rng(20261024)
    boots = []
    for _ in range(replicates):
        chosen = rng.integers(0,len(by_month),len(by_month))
        boots.append(np.concatenate([by_month[j] for j in chosen]).mean())
    lo,hi = np.quantile(boots,[.025,.975])
    p = 2*min(np.mean(np.asarray(boots)<=0),np.mean(np.asarray(boots)>=0))
    return dict(comparison='SCARF $-$ '+DISPLAY[other],n=len(d),mean_difference=d.mean(),
                ci_low=lo,ci_high=hi,p_value=p)


def write_comparison_table(seed_means):
    means = seed_means.groupby('method')[list(dict((m,m) for m,_,_ in METRICS))].mean().reindex(ORDER)
    # Recompute explicitly to avoid relying on dictionary ordering tricks.
    means = seed_means.groupby('method')[[m for m,_,_ in METRICS]].mean().reindex(ORDER)
    stds = seed_means.groupby('method')[[m for m,_,_ in METRICS]].std().reindex(ORDER).fillna(0)
    best = {}; second = {}
    for metric,_,higher in METRICS:
        ranked = means[metric].sort_values(ascending=not higher)
        best[metric] = ranked.index[0]; second[metric] = ranked.index[1]
    lines = [r'\begin{tabular}{lccccc}',r'\toprule',
             r'Method & AUC & TPR@1\%FPR & F1 & FPR $\downarrow$ & Brier $\downarrow$ \\',r'\midrule']
    for method in ORDER:
        cells = [DISPLAY[method]]
        for metric,_,_ in METRICS:
            value = f'{means.loc[method,metric]:.4f} $\\pm$ {stds.loc[method,metric]:.4f}'
            if method == best[metric]: value = r'\textbf{'+value+'}'
            elif method == second[metric]: value = r'\underline{'+value+'}'
            cells.append(value)
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TAB/'extended_comparison.tex').write_text('\n'.join(lines)+'\n')
    means.join(stds,lsuffix='_mean',rsuffix='_std').to_csv(TAB/'extended_comparison.csv')
    return means


def main():
    original = pd.read_csv(RES/'monthly_metrics.csv')
    static = pd.read_csv(RES/'static_metrics.csv')
    static = pd.concat([static.assign(seed=s,budget=100) for s in [13,42,87]],ignore_index=True)
    new = pd.read_csv(ROOT/'results_extended_comparisons/monthly_metrics.csv')
    allr = pd.concat([original,new,static],ignore_index=True)
    allr = allr[allr.method.isin(ORDER) & ((allr.budget==100)|(allr.method=='Static'))].copy()
    allr.loc[allr.method=='Static','budget'] = 100
    seed_means = allr.groupby(['method','seed'])[[m for m,_,_ in METRICS]].mean().reset_index()
    means = write_comparison_table(seed_means)
    tests = pd.DataFrame([paired_bootstrap(allr,m) for m in ORDER if m != 'SCARF'])
    tests.to_csv(RES/'extended_pairwise_tests.csv',index=False)
    tab = tests.copy(); tab.columns = ['Comparison','$n$','Mean difference','CI low','CI high','$p$-value']
    tab['$p$-value'] = tab['$p$-value'].map(lambda p:'$<$0.0001' if p<0.0001 else f'{p:.4f}')
    tab.to_latex(TAB/'extended_pairwise_tests.tex',index=False,escape=False,float_format=lambda x:f'{x:.4f}')
    # Rank matrix: 1 is best within each metric; lower FPR/Brier is better.
    ranks = pd.DataFrame(index=ORDER)
    for metric,label,higher in METRICS:
        ranks[label] = means[metric].rank(ascending=not higher,method='min').astype(int)
    fig,ax = plt.subplots(figsize=(7.1,4.0))
    im = ax.imshow(ranks.values,aspect='auto',cmap='RdYlGn_r',vmin=1,vmax=len(ORDER))
    ax.set_xticks(np.arange(ranks.shape[1]),ranks.columns)
    ax.set_yticks(np.arange(len(ORDER)),[DISPLAY[m] for m in ORDER]); ax.grid(False)
    for i in range(len(ORDER)):
        for j in range(ranks.shape[1]):
            ax.text(j,i,str(ranks.iloc[i,j]),ha='center',va='center',fontsize=7,
                    color='white' if ranks.iloc[i,j]>8 else 'black')
    cb = fig.colorbar(im,ax=ax,pad=.015); cb.set_label('Rank (1 = best)')
    ax.set_xlabel('Evaluation metric'); ax.set_ylabel('Maintenance policy or detector')
    save(fig,'fig12_extended_rank_heatmap')

    # Sensitivity: reuse existing ablations and add only genuinely new configurations.
    ab = pd.read_csv(RES/'ablation_metrics.csv')
    sens = pd.read_csv(ROOT/'results_extended_sensitivity/monthly_metrics.csv')
    scarf = original[(original.seed==42)&(original.budget==100)&(original.method=='SCARF')]
    specs = [
        ('Real-query weight',1,'SCARF-UniformQueryWeight',ab),('Real-query weight',2,'SCARF',scarf),
        ('Real-query weight',4,'SCARF-QW4',sens),
        ('Pseudo-label weight',0,'SCARF-NoSelfTraining',ab),('Pseudo-label weight',.05,'SCARF-PW05',sens),
        ('Pseudo-label weight',.1,'SCARF',scarf),('Pseudo-label weight',.2,'SCARF-PW20',sens),
        ('Ensemble members',1,'SCARF-NoEnsemble',ab),('Ensemble members',3,'SCARF',scarf),
        ('Ensemble members',5,'SCARF-M5',sens),
        ('Pseudo cap/class',250,'SCARF-Cap250',sens),('Pseudo cap/class',500,'SCARF',scarf),
        ('Pseudo cap/class',1000,'SCARF-Cap1000',sens)]
    sr = []
    for parameter,value,method,source in specs:
        z = source[source.method==method]
        row = {'Parameter':parameter,'Value':value,'Source method':method}
        for metric,_,_ in METRICS: row[metric] = z[metric].mean()
        sr.append(row)
    sr = pd.DataFrame(sr); sr.to_csv(RES/'sensitivity_metrics.csv',index=False)
    st = sr.rename(columns={'auc':'AUC','tpr_at_1fpr':'TPR@1%FPR','f1':'F1','fpr':'FPR','brier':'Brier'})
    st = st[['Parameter','Value','AUC','TPR@1%FPR','F1','FPR','Brier']]
    st['Value'] = st['Value'].map(lambda v:f'{v:g}')
    latex = st.to_latex(index=False,escape=False,float_format=lambda x:f'{x:.4f}')
    latex = latex.replace('%',r'\%')
    (TAB/'sensitivity_results.tex').write_text(latex); st.to_csv(TAB/'sensitivity_results.csv',index=False)
    fig,axes = plt.subplots(2,2,figsize=(7.1,4.4))
    for ax,(parameter,z) in zip(axes.ravel(),sr.groupby('Parameter',sort=False)):
        z = z.sort_values('Value'); xx = np.arange(len(z))
        ax.scatter(xx,z.tpr_at_1fpr,color='#009E73',marker='o',s=32,label='TPR@1%FPR',zorder=3)
        ax.set_xticks(xx,[f'{v:g}' for v in z.Value]); ax.set_xlabel(parameter)
        ax.set_ylabel('TPR@1%FPR',color='#009E73'); ax.tick_params(axis='y',labelcolor='#009E73')
        ax.set_ylim(max(.94,z.tpr_at_1fpr.min()-.002),min(1.0,z.tpr_at_1fpr.max()+.002))
        ax2 = ax.twinx(); ax2.scatter(xx,100*z.fpr,color='#D55E00',marker='s',s=28,label='FPR',zorder=3)
        ax2.set_ylabel('Realized FPR (%)',color='#D55E00'); ax2.tick_params(axis='y',labelcolor='#D55E00'); ax2.grid(False)
    fig.tight_layout(); save(fig,'fig13_hyperparameter_sensitivity')

    # Efficiency combines timed legacy/SCARF runs with timed new comparisons.
    old_eff = pd.read_csv(ROOT/'results_extended_efficiency/efficiency.csv')
    new_eff = pd.read_csv(ROOT/'results_extended_comparisons/efficiency.csv')
    eff = pd.concat([old_eff,new_eff],ignore_index=True)
    eff = eff.sort_values(['method','seed']).drop_duplicates(['method','seed'],keep='last')
    perf = allr.groupby('method').tpr_at_1fpr.mean()
    eff['tpr_at_1fpr'] = eff.method.map(perf)
    eff.to_csv(RES/'extended_efficiency.csv',index=False)
    et = eff.groupby('method')[['update_fit_seconds_per_month','scoring_ms_per_1000',
        'analyst_labels','tpr_at_1fpr']].mean().reindex(ORDER)
    et['Relative update cost'] = et.update_fit_seconds_per_month/et.loc['Uncertainty','update_fit_seconds_per_month']
    out = pd.DataFrame({'Method':pd.Series(et.index,index=et.index).map(DISPLAY),
        'Update (s/month)':et.update_fit_seconds_per_month,
        'Scoring (ms/1k)':et.scoring_ms_per_1000,'Relative update cost':et['Relative update cost'],
        'TPR@1%FPR':et.tpr_at_1fpr})
    latex = out.to_latex(index=False,escape=False,float_format=lambda x:f'{x:.3f}')
    latex = latex.replace('%',r'\%').replace('Static & 0.000',r'Static & 0.000')
    (TAB/'extended_efficiency.tex').write_text(latex); out.to_csv(TAB/'extended_efficiency.csv',index=False)

    baseline_means = means.drop(index='SCARF').tpr_at_1fpr
    strongest = baseline_means.idxmax(); delta = 100*(means.loc['SCARF','tpr_at_1fpr']-baseline_means.max())
    macros = {
        'ExtStrongestBaseline':DISPLAY[strongest],
        'ExtStrongestTPR':baseline_means.max(),
        'ExtScarfMeanTPR':means.loc['SCARF','tpr_at_1fpr'],
        'ExtDeltaStrongestPP':delta,
        'ExtScarfTPRRank':int(ranks.loc['SCARF','TPR@1\\%FPR']),
        'ExtScarfAUCRank':int(ranks.loc['SCARF','AUC']),
        'ExtScarfFoneRank':int(ranks.loc['SCARF','F1']),
        'ExtScarfFPRRank':int(ranks.loc['SCARF','FPR']),
        'ExtScarfBrierRank':int(ranks.loc['SCARF','Brier']),
        'ExtComparatorCount':len(ORDER)-1,
        'SensMinTPR':sr.tpr_at_1fpr.min(),'SensMaxTPR':sr.tpr_at_1fpr.max(),
        'EffScarfUpdate':float(et.loc['SCARF','update_fit_seconds_per_month']),
        'EffUncertaintyUpdate':float(et.loc['Uncertainty','update_fit_seconds_per_month']),
    }
    macros['EffUpdateRatio'] = macros['EffScarfUpdate']/macros['EffUncertaintyUpdate']
    with (ROOT/'extended_result_macros.tex').open('w') as f:
        for key,value in macros.items():
            rendered = str(value) if isinstance(value,str) else (f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
            f.write('\\newcommand{\\'+key+'}{'+rendered+'}\n')
    print('Extended analysis complete:',len(ORDER)-1,'comparators,',len(sr),'sensitivity rows')


if __name__ == '__main__':
    main()
