#!/usr/bin/env python3
"""Extended prequential model, memory, acquisition, sensitivity, and efficiency comparisons.

Every month is scored before its labels are revealed. Each configuration is saved
separately so interrupted runs can resume without mixing partial monthly outputs.
Only pre-extracted BODMAS feature vectors and labels are used.
"""
from pathlib import Path
import argparse, json, re, resource, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from run_experiments import (load_data, model, metrics, weighted_quantile,
    TRAIN_MONTHS, CAL_MONTH, TEST_MONTHS)
from run_scarf_v3 import pseudo_batch

COMPARISON_METHODS = ['LogisticRegression', 'RandomForest', 'ExtraTrees',
    'ReservoirReplay', 'ExpDecayReplay', 'QBC', 'EnsembleMargin']
EFFICIENCY_METHODS = ['Static', 'Random', 'Uncertainty', 'RecentOnly', 'SCARF']
SENSITIVITY = [
    ('SCARF-QW4', {'query_weight': 4.0}),
    ('SCARF-PW05', {'pseudo_weight': 0.05}),
    ('SCARF-PW20', {'pseudo_weight': 0.20}),
    ('SCARF-M5', {'members': 5}),
    ('SCARF-Cap250', {'pseudo_cap': 250}),
    ('SCARF-Cap1000', {'pseudo_cap': 1000}),
]
DEFAULTS = dict(members=3, trees=240, learning_rate=0.05,
    query_weight=2.0, pseudo_weight=0.1, pseudo_cap=500)


def is_ensemble(method):
    return method in {'QBC', 'EnsembleMargin', 'SCARF'} or method.startswith('SCARF-')


def fit_state(method, X, y, replay, seed, initial_n, params, scaler=None,
              sample_weights=None, pseudo_X=None, pseudo_y=None, pseudo_weight=0.0):
    replay = np.asarray(replay, dtype=int)
    trainX = np.asarray(X[replay]); trainy = y[replay]
    if sample_weights is None:
        weights = np.ones(len(replay), np.float32)
    else:
        weights = np.asarray(sample_weights, np.float32)
    if method == 'SCARF' or method.startswith('SCARF-'):
        weights[initial_n:] = params['query_weight']
    if pseudo_X is not None and len(pseudo_X) and pseudo_weight > 0:
        trainX = np.concatenate([trainX, pseudo_X]); trainy = np.concatenate([trainy, pseudo_y])
        weights = np.concatenate([weights, np.full(len(pseudo_y), pseudo_weight, np.float32)])
    if is_ensemble(method):
        clfs = []
        for k in range(params['members']):
            clf = model(seed + 1009*k, params['trees'])
            clf.set_params(learning_rate=params['learning_rate'])
            clf.fit(trainX, trainy, sample_weight=weights); clfs.append(clf)
        return {'kind': 'ensemble', 'clfs': clfs}
    if method == 'LogisticRegression':
        clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=2000,
                                 random_state=seed)
        clf.fit(scaler.transform(trainX), trainy, sample_weight=weights)
        return {'kind': 'linear', 'clfs': [clf], 'scaler': scaler}
    if method == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=240, max_features='sqrt',
            min_samples_leaf=2, random_state=seed, n_jobs=4)
    elif method == 'ExtraTrees':
        clf = ExtraTreesClassifier(n_estimators=240, max_features='sqrt',
            min_samples_leaf=2, random_state=seed, n_jobs=4)
    else:
        clf = model(seed, 120)
    clf.fit(trainX, trainy, sample_weight=weights)
    return {'kind': 'single', 'clfs': [clf]}


def member_probabilities(state, X):
    if state['kind'] == 'linear':
        return np.asarray([state['clfs'][0].predict_proba(state['scaler'].transform(X))[:, 1]])
    return np.asarray([c.predict_proba(X)[:, 1] for c in state['clfs']])


def predict_state(state, X):
    return member_probabilities(state, X).mean(axis=0)


def acquire(method, member_p, budget, explore, rng):
    p = member_p.mean(axis=0)
    if method in {'Random'}:
        return rng.choice(len(p), size=min(budget, len(p)), replace=False)
    if method == 'QBC':
        votes = (member_p >= 0.5).mean(axis=0)
        v = np.clip(votes, 1e-12, 1-1e-12)
        risk = -(v*np.log(v) + (1-v)*np.log(1-v))
    else:
        risk = 1-np.abs(2*p-1)
    if method == 'SCARF' or method.startswith('SCARF-'):
        nr = int(round(budget*explore)); nu = budget-nr
        chosen = np.argsort(risk)[-nu:].tolist() if nu else []
        remaining = np.setdiff1d(np.arange(len(p)), np.asarray(chosen, dtype=int))
        if nr:
            chosen.extend(rng.choice(remaining, size=nr, replace=False).tolist())
        return np.asarray(chosen, dtype=int)
    return np.argsort(risk)[-min(budget, len(p)):]


def safe_name(method):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', method)


def run_one(method, seed, budget, X, y, meta, tr, cal, tests, params):
    rng = np.random.default_rng(seed+budget)
    reservoir_rng = np.random.default_rng(seed+100000+budget)
    scaler = None
    if method == 'LogisticRegression':
        scaler = StandardScaler().fit(np.asarray(X[tr]))
    recent = []; reservoir = []; seen = 0; acquisition_month = []
    fit_seconds = 0.0; score_seconds = 0.0; query_seconds = 0.0
    start = time.perf_counter()
    state = fit_state(method, X, y, tr, seed, len(tr), params, scaler)
    fit_seconds += time.perf_counter()-start; initial_fit_seconds = fit_seconds
    start = time.perf_counter(); member_cal = member_probabilities(state, np.asarray(X[cal]))
    pc = member_cal.mean(axis=0); score_seconds += time.perf_counter()-start
    thr = float(np.quantile(pc[y[cal] == 0], .99))
    explore = 0.0
    if method == 'SCARF' or method.startswith('SCARF-'):
        from sklearn.metrics import roc_auc_score
        cal_auc = float(roc_auc_score(y[cal], pc))
        explore = float(np.clip((0.995-cal_auc)/0.05, 0, 1))
    rows = []; ledger = []
    for month_no, (month, idx) in enumerate(tests.items()):
        Xt = np.asarray(X[idx]); yt = y[idx]
        start = time.perf_counter(); member_p = member_probabilities(state, Xt)
        p = member_p.mean(axis=0); score_seconds += time.perf_counter()-start
        met = metrics(yt, p, thr)
        met.update(seed=seed, budget=budget, method=method, month=month,
                   calibration_explore=explore)
        rows.append(met)
        if method == 'Static':
            continue
        start = time.perf_counter()
        q = acquire(method, member_p, min(budget, len(idx)), explore, rng)
        qidx = idx[q]; recent.extend(qidx.tolist()); acquisition_month.extend([month_no]*len(qidx))
        if method == 'ReservoirReplay':
            for i in qidx:
                seen += 1
                if len(reservoir) < 600:
                    reservoir.append(int(i))
                else:
                    j = int(reservoir_rng.integers(0, seen))
                    if j < 600: reservoir[j] = int(i)
        pseudo_X = pseudo_y = None; pw = 0.0
        if method == 'SCARF' or method.startswith('SCARF-'):
            pseudo_X, pseudo_y = pseudo_batch(Xt, p, cap=params['pseudo_cap'])
            pw = params['pseudo_weight']*(1-explore)
        query_seconds += time.perf_counter()-start
        if method == 'RecentOnly':
            replay = np.concatenate([tr[::4], np.asarray(recent[-3*budget:], dtype=int)])
        elif method == 'ReservoirReplay':
            replay = np.concatenate([tr[::4], np.asarray(reservoir, dtype=int)])
        else:
            replay = np.concatenate([tr, np.asarray(recent, dtype=int)])
        weights = np.ones(len(replay), np.float32)
        if method == 'ExpDecayReplay' and recent:
            ages = month_no-np.asarray(acquisition_month, dtype=int)
            weights[len(tr):] = np.exp(-ages/3.0)
        start = time.perf_counter()
        state = fit_state(method, X, y, replay, seed, len(tr), params, scaler,
                          weights, pseudo_X, pseudo_y, pw)
        fit_seconds += time.perf_counter()-start
        threshold_recent = reservoir if method == 'ReservoirReplay' else recent
        pool = np.concatenate([cal, np.asarray(threshold_recent, dtype=int)])
        start = time.perf_counter(); cp = predict_state(state, np.asarray(X[pool]))
        score_seconds += time.perf_counter()-start
        cy = y[pool]
        cw = np.concatenate([np.ones(len(cal)), np.full(len(threshold_recent), 4.)])
        if method == 'ExpDecayReplay' and threshold_recent:
            cw[len(cal):] *= np.exp(-(month_no-np.asarray(acquisition_month))/3.0)
        if (cy == 0).any():
            thr = weighted_quantile(cp[cy == 0], .99, cw[cy == 0])
        ledger.extend(dict(seed=seed, budget=budget, method=method, month=month,
                           index=int(i), label=int(y[i])) for i in qidx)
    scored_n = sum(len(v) for v in tests.values())
    efficiency = dict(method=method, seed=seed, budget=budget,
        initial_fit_seconds=initial_fit_seconds,
        update_fit_seconds=fit_seconds-initial_fit_seconds,
        update_fit_seconds_per_month=(fit_seconds-initial_fit_seconds)/len(tests),
        score_seconds=score_seconds, scoring_ms_per_1000=1000*1000*score_seconds/scored_n,
        query_seconds=query_seconds, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scored_objects=scored_n, analyst_labels=budget*len(tests) if method != 'Static' else 0,
        explore_fraction=explore)
    return rows, ledger, efficiency


def main(data, out, cache, suite):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    meta, y, X, base = load_data(data, cache)
    tr = np.flatnonzero(meta.month.isin(TRAIN_MONTHS)); cal = np.flatnonzero(meta.month == CAL_MONTH)
    tests = {m: np.flatnonzero(meta.month == m) for m in TEST_MONTHS}
    configs = []
    if suite == 'comparisons':
        configs = [(m, s, 100, DEFAULTS) for m in COMPARISON_METHODS for s in [13, 42, 87]]
    elif suite == 'sensitivity':
        configs = [(m, 42, 100, dict(DEFAULTS, **p)) for m, p in SENSITIVITY]
    else:
        configs = [(m, 42, 100 if m != 'Static' else 0, DEFAULTS) for m in EFFICIENCY_METHODS]
    for run_no, (method, seed, budget, params) in enumerate(configs, 1):
        stem = f'{safe_name(method)}_seed{seed}_B{budget}'
        metric_path = out/f'{stem}_metrics.csv'
        if metric_path.exists():
            print(f'Skip completed {run_no}/{len(configs)}: {stem}', flush=True); continue
        print(f'Run {run_no}/{len(configs)}: {method} seed={seed} B={budget}', flush=True)
        rows, ledger, efficiency = run_one(method, seed, budget, X, y, meta, tr, cal, tests, params)
        pd.DataFrame(rows).to_csv(metric_path, index=False)
        if not metric_path.exists() or len(pd.read_csv(metric_path)) != len(rows):
            raise IOError(f'Metric write verification failed: {metric_path}')
        pd.DataFrame(ledger).to_csv(out/f'{stem}_queries.csv', index=False)
        (out/f'{stem}_efficiency.json').write_text(json.dumps(efficiency, indent=2))
        # Read only per-configuration outputs; never re-ingest the aggregate files.
        previous = [p for p in sorted(out.glob('*_seed*_B*_metrics.csv')) if p != metric_path]
        frames = [pd.read_csv(p) for p in previous]
        aggregate_path = out/'monthly_metrics.csv'
        if aggregate_path.exists():
            frames.append(pd.read_csv(aggregate_path))
        frames.append(pd.DataFrame(rows))
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            ['method','seed','budget','month'], keep='last')
        all_eff = [json.loads(p.read_text()) for p in sorted(out.glob('*_seed*_B*_efficiency.json'))]
        combined.to_csv(aggregate_path, index=False)
        pd.DataFrame(all_eff).to_csv(out/'efficiency.csv', index=False)
    protocol = dict(suite=suite, prequential=True, seeds=[13, 42, 87], budget=100,
        reservoir_capacity=600, exponential_decay_months=3.0,
        comparison_methods=COMPARISON_METHODS, sensitivity=SENSITIVITY,
        defaults=DEFAULTS, base_feature_count=len(base))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('Suite complete:', suite, flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--suite', choices=['comparisons', 'sensitivity', 'efficiency'], required=True)
    a = ap.parse_args(); main(a.data, a.out, a.cache, a.suite)
