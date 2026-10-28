#!/usr/bin/env python3
"""SCARF: fully prequential experiments on the real BODMAS feature archive.

The script never uses labels from month t until after month t has been scored.
It writes every monthly metric and prediction summary needed by the manuscript.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, roc_curve
import lightgbm as lgb

SEED = 42
N_BASE_FEATURES = 384
N_STABLE_FEATURES = 380
MONTHS = [f'2019-{m:02d}' for m in range(8,13)] + [f'2020-{m:02d}' for m in range(1,10)]
TRAIN_MONTHS = MONTHS[:1]
CAL_MONTH = MONTHS[1]
TEST_MONTHS = MONTHS[2:]


def model(seed=SEED, n_estimators=120):
    return lgb.LGBMClassifier(objective='binary', n_estimators=n_estimators,
        learning_rate=.08, num_leaves=31, max_depth=-1, min_child_samples=40,
        subsample=.9, subsample_freq=1, colsample_bytree=.85,
        reg_alpha=.1, reg_lambda=1.0, random_state=seed, n_jobs=4,
        verbosity=-1, deterministic=True, force_col_wise=True)


def ece(y, p, bins=10):
    edges=np.linspace(0,1,bins+1); ans=0.
    for i in range(bins):
        m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else 1.000001))
        if m.any(): ans += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(ans)


def metrics(y,p,threshold):
    z=p>=threshold; tp=int(((z==1)&(y==1)).sum()); fp=int(((z==1)&(y==0)).sum())
    tn=int(((z==0)&(y==0)).sum()); fn=int(((z==0)&(y==1)).sum())
    rf,rt,_=roc_curve(y,p); tpr1=float(np.interp(.01,rf,rt))
    return dict(n=len(y), malware=int(y.sum()), auc=float(roc_auc_score(y,p)), tpr_at_1fpr=tpr1,
        f1=float(f1_score(y,z)), tpr=tp/max(tp+fn,1), fpr=fp/max(fp+tn,1),
        precision=tp/max(tp+fp,1), accuracy=(tp+tn)/len(y),
        brier=float(brier_score_loss(y,p)), ece=ece(y,p), threshold=float(threshold),
        tp=tp,fp=fp,tn=tn,fn=fn)


def weighted_quantile(values, quantile, weights):
    order=np.argsort(values);v=values[order];w=weights[order]
    return float(v[np.searchsorted(np.cumsum(w),quantile*w.sum(),side='left')])


def psi_matrix(reference, current, bins=8):
    """Quantile-binned population stability index, one value per feature."""
    out=np.zeros(reference.shape[1],dtype=np.float32)
    for j in range(reference.shape[1]):
        r=reference[:,j]; q=np.unique(np.quantile(r,np.linspace(0,1,bins+1)))
        if len(q)<3: out[j]=0.; continue
        q[0],q[-1]=-np.inf,np.inf
        a=np.histogram(r,q)[0]/len(r); b=np.histogram(current[:,j],q)[0]/len(current)
        a=np.clip(a,1e-4,None);b=np.clip(b,1e-4,None);a/=a.sum();b/=b.sum()
        out[j]=np.sum((a-b)*np.log(a/b))
    return out


def diverse_select(X, scores, budget, rng):
    """Farthest-first selection among the highest-risk 4B candidates."""
    if budget>=len(X): return np.arange(len(X))
    k=min(len(X),4*budget)
    cand=np.argpartition(scores,-k)[-k:]
    # PCA is fitted on unlabeled candidates only; no label information is used.
    dim=min(12,len(cand)-1,X.shape[1])
    Z=PCA(n_components=dim,random_state=SEED).fit_transform(X[cand])
    Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    selected=[int(np.argmax(scores[cand]))]
    nearest=cdist(Z,Z[selected]).ravel()
    for _ in range(1,budget):
        nxt=int(np.argmax(nearest + 1e-3*(scores[cand]-scores[cand].min())))
        selected.append(nxt);nearest=np.minimum(nearest,cdist(Z,Z[[nxt]]).ravel())
    return cand[np.array(selected)]


def load_data(parquet_path, cache_dir):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    meta=pq.read_table(parquet_path,columns=['Label','sha256','family','timestamp','category']).to_pandas()
    meta['timestamp']=pd.to_datetime(meta['timestamp'],utc=True,format='mixed')
    meta['month']=meta.timestamp.dt.to_period('M').astype(str)
    y=meta.Label.to_numpy(np.int8)
    tr=meta.month.isin(TRAIN_MONTHS).to_numpy()
    feature_names=[f'F{i}' for i in range(1,2382)]
    selected_path=cache_dir/'base_features.json'
    reduced_path=cache_dir/'X384.npy'
    if selected_path.exists() and reduced_path.exists():
        selected=json.loads(selected_path.read_text())
        X=np.load(reduced_path,mmap_mode='r')
        return meta,y,X,selected
    print('Pass 1: fitting initial full-feature selector',flush=True)
    # Build the training matrix with boolean extraction from batches (memory bounded).
    chunks=[];pos=0
    for b in pq.ParquetFile(parquet_path).iter_batches(batch_size=10000,columns=feature_names):
        a=b.to_pandas().to_numpy(np.float32);m=tr[pos:pos+len(a)]
        if m.any():chunks.append(a[m])
        pos+=len(a)
    trX=np.concatenate(chunks);del chunks
    clf=model(SEED,160);clf.fit(trX,y[tr]);gain=clf.booster_.feature_importance('gain')
    selected=np.argsort(gain)[-N_BASE_FEATURES:];selected=np.sort(selected).tolist()
    del trX,clf;gc.collect()
    print('Pass 2: materializing selected features',flush=True)
    cols=[feature_names[i] for i in selected]
    X=np.lib.format.open_memmap(reduced_path,mode='w+',dtype=np.float32,shape=(len(meta),N_BASE_FEATURES))
    pos=0
    for b in pq.ParquetFile(parquet_path).iter_batches(batch_size=10000,columns=cols):
        a=b.to_pandas().to_numpy(np.float32);X[pos:pos+len(a)]=a;pos+=len(a)
    X.flush();selected_path.write_text(json.dumps(selected))
    return meta,y,np.load(reduced_path,mmap_mode='r'),selected


def run(parquet_path, outdir, cache_dir, quick=False, methods=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    meta,y,X,base_cols=load_data(parquet_path,cache_dir)
    train_idx=np.flatnonzero(meta.month.isin(TRAIN_MONTHS));cal_idx=np.flatnonzero(meta.month==CAL_MONTH)
    test_indices={m:np.flatnonzero(meta.month==m) for m in TEST_MONTHS}
    known_families=set(meta.iloc[train_idx].family.dropna())
    # Preserve an auditable protocol before running any comparison.
    protocol=dict(dataset='BODMAS timestamps parquet mirror',rows=len(meta),features=int(X.shape[1]),
        train_months=TRAIN_MONTHS,calibration_month=CAL_MONTH,test_months=TEST_MONTHS,
        train_n=len(train_idx),calibration_n=len(cal_idx),test_n=sum(map(len,test_indices.values())),
        base_feature_indices=base_cols,retained_features=384,fpr_target=.01,
        seeds=[13,42,87],budgets=[25,50,100,200],prequential=True,
        implementation='single-model-baselines')
    (outdir/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];pred_rows=[];drift_rows=[];selection_rows=[]
    configs=[]
    seeds=[42] if quick else [13,42,87]
    budgets=[100] if quick else [25,50,100,200]
    for seed in seeds:
        for budget in budgets:
            for method in ['Random','Uncertainty','RecentOnly']:
                if seed!=42 and budget!=100:continue
                configs.append((seed,budget,method))
    if quick: configs=[c for c in configs if c[2] in ['Random','SCARF']]
    if methods: configs=[c for c in configs if c[2] in methods]
    if not configs: raise ValueError('No configurations selected; check --methods.')
    # Static model and all adaptive runs share the same initial training data and model class.
    for run_no,(seed,budget,method) in enumerate(configs):
        print(f'Run {run_no+1}/{len(configs)}: {method} seed={seed} B={budget}',flush=True)
        rng=np.random.default_rng(seed+budget)
        replay=list(train_idx);recent_labels=[]
        stable_cols=np.arange(X.shape[1]);clf=model(seed);clf.fit(np.asarray(X[replay]),y[replay])
        pcal=clf.predict_proba(np.asarray(X[cal_idx]))[:,1]
        threshold=float(np.quantile(pcal[y[cal_idx]==0],.99))
        reference=np.asarray(X[cal_idx][:,stable_cols])
        # Class-conditional novelty reference in a small PCA space (calibration data only).
        pca=PCA(n_components=16,random_state=seed).fit(reference[::4])
        Zcal=pca.transform(reference[::4]);ycal=y[cal_idx][::4]
        centers=np.array([Zcal[ycal==c].mean(0) for c in [0,1]])
        scales=np.array([Zcal[ycal==c].std(0)+1e-4 for c in [0,1]])
        for month in TEST_MONTHS:
            idx=test_indices[month];Xt=np.asarray(X[idx][:,stable_cols]);yt=y[idx]
            p=clf.predict_proba(Xt)[:,1]
            met=metrics(yt,p,threshold);met.update(seed=seed,budget=budget,method=method,month=month)
            rows.append(met)
            uncertainty=1-np.abs(2*p-1)
            Z=pca.transform(Xt)
            novelty=np.min(np.stack([np.mean(((Z-centers[c])/scales[c])**2,axis=1) for c in [0,1]]),axis=0)
            def rank01(a):
                o=np.argsort(np.argsort(a));return o/max(len(a)-1,1)
            risk=.9*rank01(uncertainty)+.1*rank01(novelty)
            psi=psi_matrix(reference,Xt)
            drift_rows.append(dict(seed=seed,budget=budget,method=method,month=month,
                mean_psi=float(np.mean(psi)),max_psi=float(np.max(psi)),
                risk_q90=float(np.quantile(risk,.9)),mean_uncertainty=float(np.mean(uncertainty)),
                mean_novelty=float(np.mean(novelty))))
            # Family-aware summaries; labels are used only for post-hoc evaluation.
            fam=meta.iloc[idx].family.to_numpy();known=np.array([(f in known_families) for f in fam])
            for group,mask in [('Known family',known&(yt==1)),('Novel family',(~known)&(yt==1)),('Benign',yt==0)]:
                if mask.sum():
                    pred_rows.append(dict(seed=seed,budget=budget,method=method,month=month,group=group,
                        n=int(mask.sum()),tpr=float((p[mask]>=threshold).mean()) if group!='Benign' else np.nan,
                        fpr=float((p[mask]>=threshold).mean()) if group=='Benign' else np.nan,
                        mean_score=float(p[mask].mean())))
            # Labels from month t become available only now, after all month-t metrics are stored.
            if method=='Random':q=rng.choice(len(idx),size=min(budget,len(idx)),replace=False)
            elif method in ['Uncertainty','RecentOnly']:q=np.argsort(uncertainty)[-min(budget,len(idx)):]
            else:q=diverse_select(Xt,risk,min(budget,len(idx)),rng)
            qidx=idx[q]
            selection_rows.extend(dict(seed=seed,budget=budget,method=method,month=month,
                index=int(i),label=int(y[i]),risk=float(risk[q[k]])) for k,i in enumerate(qidx))
            recent_labels.extend(qidx.tolist())
            if method=='RecentOnly':replay=list(train_idx[::4])+recent_labels[-3*budget:]
            else:replay=list(train_idx)+recent_labels
            # SCARF uses model-sensitive feature gains regularized by unlabeled temporal PSI.
            if method=='SCARF':
                gains=clf.feature_importances_;stability=1/(1+np.maximum(psi,0))
                stable_cols=np.argsort(np.log1p(gains)*(.25+.75*stability))[-N_STABLE_FEATURES:]
                stable_cols=np.sort(stable_cols)
                reference=np.asarray(X[cal_idx][:,stable_cols])
                pca=PCA(n_components=16,random_state=seed).fit(reference[::4])
                Zcal=pca.transform(reference[::4]);ycal=y[cal_idx][::4]
                centers=np.array([Zcal[ycal==c].mean(0) for c in [0,1]])
                scales=np.array([Zcal[ycal==c].std(0)+1e-4 for c in [0,1]])
            clf=model(seed);clf.fit(np.asarray(X[replay][:,stable_cols]),y[replay])
            # Weighted conformal operating-point update: old calibration scores have unit
            # weight and newly labeled benign scores have weight four. Only past labels are used.
            pool=np.concatenate([cal_idx,np.asarray(recent_labels,dtype=int)])
            cp=clf.predict_proba(np.asarray(X[pool][:,stable_cols]))[:,1]
            cy=y[pool];cw=np.concatenate([np.ones(len(cal_idx)),np.full(len(recent_labels),4.)])
            if (cy==0).any():threshold=weighted_quantile(cp[cy==0],.99,cw[cy==0])
        # flush after every configuration
        pd.DataFrame(rows).to_csv(outdir/'monthly_metrics.csv',index=False)
        pd.DataFrame(drift_rows).to_csv(outdir/'drift_metrics.csv',index=False)
        pd.DataFrame(pred_rows).to_csv(outdir/'group_metrics.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(outdir/'selected_samples.csv',index=False)
    # Static reference is run once; zero additional labels.
    clf=model(42);clf.fit(np.asarray(X[train_idx]),y[train_idx])
    pcal=clf.predict_proba(np.asarray(X[cal_idx]))[:,1];threshold=float(np.quantile(pcal[y[cal_idx]==0],.99))
    static=[]
    for month,idx in test_indices.items():
        p=clf.predict_proba(np.asarray(X[idx]))[:,1];met=metrics(y[idx],p,threshold)
        met.update(seed=42,budget=0,method='Static',month=month);static.append(met)
    pd.DataFrame(static).to_csv(outdir/'static_metrics.csv',index=False)
    print('Completed',len(rows),'adaptive monthly evaluations',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--cache',required=True);ap.add_argument('--quick',action='store_true')
    ap.add_argument('--methods',nargs='+',choices=['Random','Uncertainty','RecentOnly']);a=ap.parse_args()
    warnings.filterwarnings('ignore',category=UserWarning)
    run(a.data,a.out,a.cache,a.quick,a.methods)
