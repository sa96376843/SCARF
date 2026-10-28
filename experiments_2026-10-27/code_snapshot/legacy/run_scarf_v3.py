#!/usr/bin/env python3
"""Final SCARF-v3: reliability-gated querying, replay, self-training, and ensembles."""
from pathlib import Path
import argparse,json,warnings
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from run_experiments import (load_data,model,metrics,weighted_quantile,psi_matrix,
    TRAIN_MONTHS,CAL_MONTH,TEST_MONTHS)

ENSEMBLE_MEMBERS=3
ENSEMBLE_TREES=240
ENSEMBLE_LEARNING_RATE=0.05
QUERY_WEIGHT=2.0
PSEUDO_WEIGHT=0.1
PSEUDO_CAP_PER_CLASS=500
RELIABILITY_AUC=0.995
RELIABILITY_SCALE=0.05


def fit_ensemble(X,y,replay,seed,initial_n,query_weight=QUERY_WEIGHT,members=ENSEMBLE_MEMBERS,
                 pseudo_X=None,pseudo_y=None,pseudo_weight=PSEUDO_WEIGHT,weight_mode='uniform'):
    replay=np.asarray(replay,dtype=int);trainX=np.asarray(X[replay]);trainy=y[replay]
    weights=np.ones(len(replay),np.float32)
    if weight_mode=='hard_benign':weights[initial_n:]=np.where(trainy[initial_n:]==0,query_weight,1.)
    else:weights[initial_n:]=query_weight
    if pseudo_X is not None and len(pseudo_X) and pseudo_weight>0:
        trainX=np.concatenate([trainX,pseudo_X]);trainy=np.concatenate([trainy,pseudo_y])
        weights=np.concatenate([weights,np.full(len(pseudo_y),pseudo_weight,np.float32)])
    clfs=[]
    for k in range(members):
        clf=model(seed+1009*k,ENSEMBLE_TREES);clf.set_params(learning_rate=ENSEMBLE_LEARNING_RATE)
        clf.fit(trainX,trainy,sample_weight=weights);clfs.append(clf)
    return clfs

def predict(clfs,X):
    return np.mean([c.predict_proba(X)[:,1] for c in clfs],axis=0)

def pseudo_batch(X,p,cap=PSEUDO_CAP_PER_CLASS):
    xs=[];ys=[]
    for c in [0,1]:
        ids=np.flatnonzero((p<=.005) if c==0 else (p>=.995))
        if len(ids)>cap:ids=ids[np.argsort(np.abs(p[ids]-c))[:cap]]
        xs.append(X[ids]);ys.append(np.full(len(ids),c,np.int8))
    return np.concatenate(xs),np.concatenate(ys)

def acquisition(p,budget,explore,rng):
    u=1-np.abs(2*p-1);nr=int(round(budget*explore));nu=budget-nr
    chosen=[]
    if nu:chosen.extend(np.argsort(u)[-nu:].tolist())
    remaining=np.setdiff1d(np.arange(len(p)),np.asarray(chosen,dtype=int))
    if nr:chosen.extend(rng.choice(remaining,size=nr,replace=False).tolist())
    q=np.asarray(chosen,dtype=int)
    flags=np.zeros(len(q),np.int8);flags[nu:]=1
    return q,u,flags

def run(data,out,cache,months=None,weight_mode='uniform'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);months=months or TEST_MONTHS
    meta,y,X,base=load_data(data,cache);tr=np.flatnonzero(meta.month.isin(TRAIN_MONTHS));cal=np.flatnonzero(meta.month==CAL_MONTH)
    tests={m:np.flatnonzero(meta.month==m) for m in months};known=set(meta.iloc[tr].family.dropna())
    protocol=dict(version='SCARF-v3',development_months=TEST_MONTHS[:6],locked_months=TEST_MONTHS[6:],
        ensemble_members=ENSEMBLE_MEMBERS,ensemble_trees=ENSEMBLE_TREES,ensemble_learning_rate=ENSEMBLE_LEARNING_RATE,
        query_weight=QUERY_WEIGHT,pseudo_weight=PSEUDO_WEIGHT,
        pseudo_cap_per_class=PSEUDO_CAP_PER_CLASS,reliability_auc=RELIABILITY_AUC,reliability_scale=RELIABILITY_SCALE,
        weight_mode=weight_mode,prequential=True)
    (out/'protocol_v3.json').write_text(json.dumps(protocol,indent=2))
    rows=[];drift=[];groups=[];ledger=[]
    configs=[(13,100),(42,25),(42,50),(42,100),(42,200),(87,100)]
    for seed,budget in configs:
        print('SCARF-v3',seed,budget,flush=True);rng=np.random.default_rng(seed+budget);recent=[]
        clfs=fit_ensemble(X,y,tr,seed,len(tr),weight_mode=weight_mode)
        pc=predict(clfs,np.asarray(X[cal]));thr=float(np.quantile(pc[y[cal]==0],.99));cal_auc=float(roc_auc_score(y[cal],pc))
        explore=float(np.clip((RELIABILITY_AUC-cal_auc)/RELIABILITY_SCALE,0,1));reference=np.asarray(X[cal])
        for month,idx in tests.items():
            Xt=np.asarray(X[idx]);yt=y[idx];p=predict(clfs,Xt);met=metrics(yt,p,thr)
            met.update(seed=seed,budget=budget,method='SCARF',month=month,calibration_auc=cal_auc,explore_fraction=explore);rows.append(met)
            u=1-np.abs(2*p-1);psi=psi_matrix(reference,Xt)
            pX,py=pseudo_batch(Xt,p);pw=PSEUDO_WEIGHT*(1-explore)
            drift.append(dict(seed=seed,budget=budget,method='SCARF',month=month,mean_psi=float(psi.mean()),max_psi=float(psi.max()),
                risk_q90=float(np.quantile(u,.9)),mean_uncertainty=float(u.mean()),mean_novelty=np.nan,
                pseudo_n=int(len(py) if pw>0 else 0),explore_fraction=explore))
            fam=meta.iloc[idx].family.to_numpy();known_mask=np.array([f in known for f in fam])
            for group,mask in [('Known family',known_mask&(yt==1)),('Novel family',(~known_mask)&(yt==1)),('Benign',yt==0)]:
                if mask.sum():groups.append(dict(seed=seed,budget=budget,method='SCARF',month=month,group=group,n=int(mask.sum()),
                    tpr=float((p[mask]>=thr).mean()) if group!='Benign' else np.nan,fpr=float((p[mask]>=thr).mean()) if group=='Benign' else np.nan,
                    mean_score=float(p[mask].mean())))
            q,u,flags=acquisition(p,min(budget,len(idx)),explore,rng);qidx=idx[q];recent.extend(qidx.tolist())
            ledger.extend(dict(seed=seed,budget=budget,method='SCARF',month=month,index=int(i),label=int(y[i]),risk=float(u[q[k]]),
                random_exploration=int(flags[k])) for k,i in enumerate(qidx))
            replay=np.concatenate([tr,np.asarray(recent,dtype=int)])
            clfs=fit_ensemble(X,y,replay,seed,len(tr),pseudo_X=pX,pseudo_y=py,pseudo_weight=pw,weight_mode=weight_mode)
            pool=np.concatenate([cal,np.asarray(recent,dtype=int)]);cp=predict(clfs,np.asarray(X[pool]));cy=y[pool]
            cw=np.concatenate([np.ones(len(cal)),np.full(len(recent),4.)]);thr=weighted_quantile(cp[cy==0],.99,cw[cy==0])
        pd.DataFrame(rows).to_csv(out/'monthly_metrics.csv',index=False);pd.DataFrame(drift).to_csv(out/'drift_metrics.csv',index=False)
        pd.DataFrame(groups).to_csv(out/'group_metrics.csv',index=False);pd.DataFrame(ledger).to_csv(out/'selected_samples.csv',index=False)
    print('Completed',len(rows),'SCARF-v3 monthly evaluations',flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=UserWarning)
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--development-only',action='store_true');p.add_argument('--weight-mode',choices=['uniform','hard_benign'],default='uniform');a=p.parse_args()
    months=TEST_MONTHS[:6] if a.development_only else None
    run(a.data,a.out,a.cache,months,a.weight_mode)
