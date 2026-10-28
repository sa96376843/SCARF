#!/usr/bin/env python3
"""Component ablations for the locked SCARF-v3 configuration (seed 42, B=100)."""
from pathlib import Path
import argparse,warnings
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from run_experiments import load_data,metrics,weighted_quantile,TRAIN_MONTHS,CAL_MONTH,TEST_MONTHS
from run_scarf_v3 import fit_ensemble,predict,pseudo_batch,acquisition,QUERY_WEIGHT,PSEUDO_WEIGHT,RELIABILITY_AUC,RELIABILITY_SCALE

def main(data,out,cache):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);meta,y,X,base=load_data(data,cache)
    tr=np.flatnonzero(meta.month.isin(TRAIN_MONTHS));cal=np.flatnonzero(meta.month==CAL_MONTH);tests={m:np.flatnonzero(meta.month==m) for m in TEST_MONTHS}
    rows=[]
    for method in ['SCARF-NoEnsemble','SCARF-NoSelfTraining','SCARF-UniformQueryWeight','SCARF-NoCalibration','SCARF-NoReplay']:
        print(method,flush=True);seed=42;budget=100;rng=np.random.default_rng(142);recent=[]
        members=1 if method=='SCARF-NoEnsemble' else 3;qw=1. if method=='SCARF-UniformQueryWeight' else QUERY_WEIGHT
        init=tr[::4] if method=='SCARF-NoReplay' else tr
        clfs=fit_ensemble(X,y,init,seed,len(init),members=members)
        pc=predict(clfs,np.asarray(X[cal]));thr=float(np.quantile(pc[y[cal]==0],.99));fixed=thr
        cal_auc=float(roc_auc_score(y[cal],pc));explore=float(np.clip((RELIABILITY_AUC-cal_auc)/RELIABILITY_SCALE,0,1))
        for month,idx in tests.items():
            Xt=np.asarray(X[idx]);yt=y[idx];p=predict(clfs,Xt);z=metrics(yt,p,thr)
            z.update(seed=seed,budget=budget,method=method,month=month,calibration_auc=cal_auc,explore_fraction=explore);rows.append(z)
            q,u,flags=acquisition(p,budget,explore,rng);recent.extend(idx[q].tolist())
            pX,py=pseudo_batch(Xt,p);pw=0 if method=='SCARF-NoSelfTraining' else PSEUDO_WEIGHT*(1-explore)
            replay=np.concatenate([init,np.asarray(recent[-3*budget:] if method=='SCARF-NoReplay' else recent,dtype=int)])
            clfs=fit_ensemble(X,y,replay,seed,len(init),query_weight=qw,members=members,pseudo_X=pX,pseudo_y=py,pseudo_weight=pw)
            if method!='SCARF-NoCalibration':
                pool=np.concatenate([cal,np.asarray(recent,dtype=int)]);cp=predict(clfs,np.asarray(X[pool]));cy=y[pool]
                cw=np.concatenate([np.ones(len(cal)),np.full(len(recent),4.)]);thr=weighted_quantile(cp[cy==0],.99,cw[cy==0])
            else:thr=fixed
        pd.DataFrame(rows).to_csv(out/'ablation_metrics.csv',index=False)
if __name__=='__main__':
    warnings.filterwarnings('ignore',category=UserWarning)
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',required=True);a=p.parse_args();main(a.data,a.out,a.cache)
