"""Isolated SCARF dynamic-gate runner, with budget-matched causal ablations."""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
import numpy as np
from gate import GateConfig, ReliabilityGate, acquire, pseudo_indices, epsilon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'legacy'))
MONTHS = [f'2019-{m:02d}' for m in range(10, 13)] + [f'2020-{m:02d}' for m in range(1, 10)]
METHODS = ['LegacyV3', 'EnsembleMargin', 'FrozenGate', 'DynamicGate', 'QueryOnly',
           'PseudoOnly', 'NoPseudo', 'RecentReplay', 'FrozenThreshold']


@dataclass(frozen=True)
class Variant:
    name: str
    legacy: bool = False
    query_feedback: bool = False
    pseudo_feedback: bool = False
    query_weight: float = 2.0
    pseudo_weight: float = .1
    recent_replay: bool = False
    frozen_threshold: bool = False


def variant(name):
    if name not in METHODS:
        raise ValueError(name)
    if name == 'LegacyV3':
        return Variant(name, legacy=True)
    if name == 'EnsembleMargin':
        return Variant(name, legacy=True, query_weight=1, pseudo_weight=0)
    return Variant(name, query_feedback=name not in ['FrozenGate', 'PseudoOnly'],
                   pseudo_feedback=name not in ['FrozenGate', 'QueryOnly', 'NoPseudo'],
                   pseudo_weight=0 if name == 'NoPseudo' else .1,
                   recent_replay=name == 'RecentReplay', frozen_threshold=name == 'FrozenThreshold')


def configured_variant(name, protocol):
    setting = variant(name)
    # Honor the protocol instead of leaving apparently editable fields unused.
    return replace(setting,
                   query_weight=1.0 if name == 'EnsembleMargin' else protocol['query_weight'],
                   pseudo_weight=0.0 if name in ['EnsembleMargin', 'NoPseudo'] else protocol['pseudo_weight'])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class LightGBMBackend:
    def __init__(self, seed, config):
        self.seed, self.config = seed, config
        self.clfs = []

    def fit(self, X, y, weights):
        from run_experiments import model
        self.clfs = []
        for k in range(self.config['members']):
            clf = model(self.seed+1009*k, self.config['trees'])
            clf.set_params(learning_rate=self.config['learning_rate'], n_jobs=self.config['threads'])
            clf.fit(X, y, sample_weight=weights)
            self.clfs.append(clf)

    def predict(self, X):
        return np.mean([c.predict_proba(X)[:, 1] for c in self.clfs], axis=0)

    def save(self, folder, label):
        for i, clf in enumerate(self.clfs):
            clf.booster_.save_model(str(folder / f'{label}_member{i}.txt'))


def run_stream(X, y, tr, cal, tests, setting, seed, budget, gate_config,
               backend, output, pseudo_cap=500):
    """All access to queried labels occurs after frozen predictions/metrics.

    Evaluation labels are used ONLY for reporting. Gate.observe receives solely
    uniform audit labels and their frozen PRE-update predictions.
    """
    from sklearn.metrics import roc_auc_score
    from run_experiments import metrics, weighted_quantile
    from run_scarf_v3 import acquisition as legacy_acquire, pseudo_batch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len(np.unique(y[tr])) != 2 or len(np.unique(y[cal])) != 2:
        raise ValueError('Training and calibration must contain both classes')
    indices = [np.asarray(tr), np.asarray(cal), *[np.asarray(ids) for _, ids in tests]]
    combined = np.concatenate(indices)
    if len(np.unique(combined)) != len(combined):
        raise ValueError('Temporal partitions overlap')
    if [m for m, _ in tests] != sorted(set(m for m, _ in tests)):
        raise ValueError('Months must be unique and chronological')
    initial_start = time.perf_counter()
    backend.fit(np.asarray(X[tr]), y[tr], np.ones(len(tr), dtype=np.float32))
    initial_seconds = time.perf_counter()-initial_start
    pc = backend.predict(np.asarray(X[cal]))
    threshold = float(np.quantile(pc[y[cal] == 0], .99))
    fixed_threshold = threshold
    initial_auc = float(roc_auc_score(y[cal], pc))
    gate = ReliabilityGate(initial_auc, gate_config)
    initial_epsilon = gate.explore
    legacy_rng = np.random.default_rng(seed+budget)
    selection_rng = np.random.default_rng(seed+budget)
    audit_rng = np.random.default_rng(seed+budget+200000)
    recent = []
    rows, feedback, queries, pseudo_rows = [], [], [], []
    train_seconds = score_seconds = 0.0
    for step, (month, idx) in enumerate(tests):
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0 or len(np.unique(y[idx])) != 2:
            raise ValueError(f'{month}: empty or single-class evaluation batch')
        if setting.legacy:
            used_epsilon = initial_epsilon if setting.name == 'LegacyV3' else 0.0
        else:
            used_epsilon = gate.explore if setting.query_feedback else initial_epsilon
        t0 = time.perf_counter()
        p = backend.predict(np.asarray(X[idx]))
        score_seconds += time.perf_counter()-t0
        # Save raw predictions before acquiring any labels; scores are immutable.
        np.savez_compressed(output / f'predictions_{month}.npz', index=idx, label=y[idx],
                            probability=p, threshold=threshold)
        row = metrics(y[idx], p, threshold)
        row.update(method=setting.name, seed=seed, budget=budget, month=month,
                   reliability_auc_before=gate.rho, exploration_used=used_epsilon)
        rows.append(row)
        write_csv(output / 'monthly_metrics.csv', rows)
        if setting.legacy:
            if setting.name == 'EnsembleMargin':
                q = np.argsort(1-np.abs(2*p-1))[-min(budget, len(idx)):]
                roles = np.full(len(q), 'uncertainty')
            else:
                q, _, flags = legacy_acquire(p, min(budget, len(idx)), used_epsilon, legacy_rng)
                roles = np.where(flags == 1, 'exploration', 'uncertainty')
            state = dict(reliability_auc=initial_auc, measured_audit_auc=None,
                         feedback_valid=False, feedback_status='legacy_fixed',
                         epsilon_before=initial_epsilon, epsilon_after=initial_epsilon,
                         audit_positive=0, audit_negative=0, audit_pairs=0,
                         current_audit_fpr=None, history_steps=[])
        else:
            q, roles = acquire(p, budget, used_epsilon, gate_config, audit_rng, selection_rng)
            audit_q = q[roles == 'audit']
            state = gate.observe(step, p[audit_q], y[idx[audit_q]], threshold)
        qidx = idx[q]
        recent.extend(qidx.tolist())
        for local, global_id, role in zip(q, qidx, roles):
            queries.append(dict(method=setting.name, seed=seed, budget=budget, month=month,
                                index=int(global_id), label=int(y[global_id]), role=str(role),
                                pre_update_probability=float(p[local])))
        weight_epsilon = gate.explore if setting.pseudo_feedback else initial_epsilon
        pw = setting.pseudo_weight * (1-weight_epsilon)
        if setting.legacy:
            px, py = pseudo_batch(np.asarray(X[idx]), p, cap=pseudo_cap)
            pids = None  # Legacy selection intentionally retains original semantics.
        else:
            pids, py = pseudo_indices(p, q, pseudo_cap)
            px = np.asarray(X[idx[pids]])
            # This label access is an evaluation diagnostic ONLY, after selection.
            for local, label in zip(pids, py):
                pseudo_rows.append(dict(method=setting.name, seed=seed, budget=budget, month=month,
                                        index=int(idx[local]), pseudo_label=int(label),
                                        true_label_posthoc=int(y[idx[local]]),
                                        probability=float(p[local]), training_weight=pw))
        replay_queries = recent[-3*budget:] if setting.recent_replay else recent
        # Clean replay ablation: ALWAYS retain the exact full initial training set.
        replay = np.concatenate([tr, np.asarray(replay_queries, dtype=int)])
        weights = np.ones(len(replay), dtype=np.float32)
        weights[len(tr):] = setting.query_weight
        tx, ty = np.asarray(X[replay]), y[replay]
        if len(py) and pw > 0:
            tx, ty = np.concatenate([tx, px]), np.concatenate([ty, py])
            weights = np.concatenate([weights, np.full(len(py), pw, np.float32)])
        t0 = time.perf_counter()
        backend.fit(tx, ty, weights)
        elapsed = time.perf_counter()-t0
        train_seconds += elapsed
        # Same historical threshold rule as v3, except in FrozenThreshold.
        if setting.frozen_threshold:
            threshold = fixed_threshold
        else:
            pool = np.concatenate([cal, np.asarray(recent, dtype=int)])
            cp = backend.predict(np.asarray(X[pool]))
            cw = np.concatenate([np.ones(len(cal)), np.full(len(recent), 4.)])
            threshold = weighted_quantile(cp[y[pool] == 0], .99, cw[y[pool] == 0])
        state.pop('config', None)
        state['history_steps'] = json.dumps(state['history_steps'])
        state.update(method=setting.name, seed=seed, budget=budget, month=month,
                     query_epsilon_used=used_epsilon,
                     query_epsilon_next=gate.explore if setting.query_feedback else initial_epsilon,
                     audit_count=int((roles == 'audit').sum()),
                     extra_exploration_count=int((roles == 'exploration').sum()),
                     uncertainty_count=int((roles == 'uncertainty').sum()),
                     pseudo_selected=len(py), pseudo_used=len(py) if pw > 0 else 0,
                     pseudo_weight=pw, replay_initial_n=len(tr), replay_queried_n=len(replay_queries),
                     update_seconds=elapsed, threshold_next=threshold)
        feedback.append(state)
        write_csv(output / 'feedback.csv', feedback)
        write_csv(output / 'queries.csv', queries)
        write_csv(output / 'pseudo_labels.csv', pseudo_rows)
        backend.save(output, f'after_{month}')
        print(f'{setting.name} seed={seed} {month}: eps={used_epsilon:.3f} -> '
              f'{state["query_epsilon_next"]:.3f}, {state["feedback_status"]}', flush=True)
    return dict(months=len(rows), analyst_labels=len(queries), initial_fit_seconds=initial_seconds,
                update_seconds=train_seconds, score_seconds=score_seconds,
                audit_n=sum(f['audit_count'] for f in feedback),
                observed_gate_positive_months=sum(f['epsilon_after'] > 0 for f in feedback),
                positive_acquisition_months=sum(f['query_epsilon_used'] > 0 for f in feedback),
                valid_feedback_months=sum(f['feedback_valid'] for f in feedback),
                gate_efficacy_established=False)


def guarded_load(data, cache, expected):
    from run_experiments import load_data
    actual = digest(data)
    if actual != expected:
        raise ValueError(f'Dataset SHA-256 mismatch: expected {expected}, got {actual}')
    cache = Path(cache) / actual[:16]
    cache.mkdir(parents=True, exist_ok=True)
    signature = {'data_sha256': actual, 'loader_sha256': digest(ROOT / 'legacy/run_experiments.py')}
    manifest = cache / 'cache_manifest.json'
    if any(cache.iterdir()):
        if not manifest.exists():
            raise ValueError('Incomplete/unverified cache; use a new cache directory')
        saved = json.loads(manifest.read_text())
        if saved['signature'] != signature:
            raise ValueError('Cache signature mismatch')
        for name, sha in saved['files'].items():
            if digest(cache / name) != sha:
                raise ValueError(f'Cache content changed: {name}')
    meta, y, X, base = load_data(data, cache)
    if X.shape != (len(meta), 384) or len(base) != 384 or len(set(base)) != 384:
        raise ValueError('Invalid reduced feature cache')
    write_json(manifest, dict(signature=signature, files={name: digest(cache / name)
                                                        for name in ['X384.npy', 'base_features.json']}))
    return meta, y, X, base, actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs')
    ap.add_argument('--cache', type=Path, default=ROOT / 'cache')
    ap.add_argument('--protocol', type=Path, default=ROOT / 'protocol.json')
    ap.add_argument('--methods', nargs='+', choices=METHODS)
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    methods = args.methods or protocol['methods']
    seeds = args.seeds or protocol['seeds']
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        ap.error('Duplicate methods or seeds')
    gc = GateConfig(**protocol['gate'])
    if protocol['budget'] < 1:
        ap.error('Budget must be positive')
    print(json.dumps(dict(methods=methods, seeds=seeds, budget=protocol['budget'],
                          runs=len(methods)*len(seeds), months=MONTHS, gate=asdict(gc)), indent=2))
    if args.plan_only:
        return
    if args.data is None or not args.data.is_file():
        ap.error('--data must point to the original timestamped BODMAS Parquet')
    old = ROOT.parent / 'SCARF_TDSC'
    for path in [args.out, args.cache]:
        if path.resolve().is_relative_to(old.resolve()):
            ap.error('Output/cache must not be inside the historical SCARF_TDSC artifact')
    import importlib.metadata as md
    packages = {n: md.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'pyarrow']}
    meta, y, X, base, sha = guarded_load(args.data, args.cache, protocol['dataset_sha256'])
    tr = np.flatnonzero(meta.month == '2019-08')
    cal = np.flatnonzero(meta.month == '2019-09')
    tests = [(month, np.flatnonzero(meta.month == month)) for month in MONTHS]
    if any(len(ids) == 0 for ids in [tr, cal, *[ids for _, ids in tests]]):
        raise ValueError('Missing temporal partition')
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               [ROOT/'gate.py', ROOT/'run_dynamic.py', *sorted((ROOT/'legacy').glob('*.py'))]}
    for name in methods:
        for seed in seeds:
            setting = configured_variant(name, protocol)
            folder = args.out / f'{name}_seed{seed}_B{protocol["budget"]}'
            # Never silently skip partial runs or overwrite completed evidence.
            folder.mkdir(parents=True, exist_ok=False)
            run_info = dict(protocol=protocol, variant=asdict(setting), seed=seed,
                            source_sha256=sources, dataset_sha256=sha, base_features=base,
                            python=sys.version, platform=platform.platform(), packages=packages)
            write_json(folder/'run_manifest.json', run_info)
            write_json(folder/'status.json', {'status': 'RUNNING'})
            try:
                result = run_stream(X, y, tr, cal, tests, setting, seed, protocol['budget'], gc,
                                    LightGBMBackend(seed, protocol['model']), folder, protocol['pseudo_cap'])
                write_json(folder/'status.json', {'status': 'COMPLETE', **result})
            except BaseException as exc:
                write_json(folder/'status.json', {'status': 'FAILED', 'error': repr(exc),
                                                  'traceback': traceback.format_exc()})
                raise


if __name__ == '__main__':
    main()
