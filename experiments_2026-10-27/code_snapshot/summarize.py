"""Aggregate complete dynamic-gate runs, retaining F1/FPR/ranking trade-offs."""
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import argparse
import csv
import json
import sys
import numpy as np
from run_dynamic import ROOT, MONTHS, write_csv, write_json, digest

METRICS = ['f1', 'fpr', 'tpr_at_1fpr', 'auc', 'brier', 'ece']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(runs, output, protocol):
    runs, output = Path(runs), Path(output)
    if output.exists():
        raise FileExistsError('Use a new summary directory to retain earlier analysis')
    rows, statuses, ledger_checks = [], [], []
    hashes = {}
    fingerprints = set()
    for method in protocol['methods']:
        for seed in protocol['seeds']:
            folder = runs / f'{method}_seed{seed}_B{protocol["budget"]}'
            status = json.loads((folder/'status.json').read_text())
            manifest = json.loads((folder/'run_manifest.json').read_text())
            if status['status'] != 'COMPLETE':
                raise ValueError(f'Incomplete run: {folder.name}')
            if manifest['protocol'] != protocol or manifest['seed'] != seed or manifest['variant']['name'] != method:
                raise ValueError(f'Protocol/run identity mismatch: {folder.name}')
            fingerprints.add(json.dumps({k: manifest[k] for k in
                                         ['source_sha256', 'dataset_sha256', 'packages']}, sort_keys=True))
            monthly = read_csv(folder/'monthly_metrics.csv')
            feedback = read_csv(folder/'feedback.csv')
            queries = read_csv(folder/'queries.csv')
            if sorted(r['month'] for r in monthly) != MONTHS or sorted(r['month'] for r in feedback) != MONTHS:
                raise ValueError(f'Missing or duplicate months: {folder.name}')
            keys = [(r['month'], int(r['index'])) for r in queries]
            if len(keys) != len(set(keys)):
                raise ValueError(f'Duplicate queried samples: {folder.name}')
            for row in monthly:
                if row['method'] != method or int(row['seed']) != seed or int(row['budget']) != protocol['budget']:
                    raise ValueError('Metric identity mismatch')
                expected = min(protocol['budget'], int(row['n']))
                if sum(q['month'] == row['month'] for q in queries) != expected:
                    raise ValueError('Analyst budget mismatch')
                values = {metric: float(row[metric]) for metric in METRICS}
                if not np.isfinite(list(values.values())).all():
                    raise ValueError('Nonfinite metrics')
                rows.append(dict(method=method, seed=seed, month=row['month'], **values))
            statuses.append(dict(method=method, seed=seed,
                                 valid_feedback_months=status['valid_feedback_months'],
                                 observed_positive_months=status['observed_gate_positive_months'],
                                 used_positive_months=status['positive_acquisition_months']))
            ledger_checks.append(dict(run=folder.name, unique_queries=True, budget_valid=True))
            for name in ['status.json', 'run_manifest.json', 'monthly_metrics.csv', 'feedback.csv', 'queries.csv']:
                hashes[str((folder/name).relative_to(runs))] = digest(folder/name)
    if len(fingerprints) != 1:
        raise ValueError('Mixed source/data/package versions: report them separately')
    per_seed = defaultdict(list)
    for row in rows:
        per_seed[(row['method'], row['seed'])].append(row)
    seed_rows = [dict(method=method, seed=seed,
                      **{m: mean(r[m] for r in group) for m in METRICS})
                 for (method, seed), group in sorted(per_seed.items())]
    summary = []
    for method in protocol['methods']:
        group = [r for r in seed_rows if r['method'] == method]
        summary.append(dict(method=method, seeds=len(group),
                            **{m+'_mean': mean(r[m] for r in group) for m in METRICS},
                            **{m+'_std': stdev(r[m] for r in group) if len(group)>1 else None for m in METRICS}))
    pairs = []
    if 'DynamicGate' in protocol['methods']:
        for comparator in [m for m in protocol['methods'] if m != 'DynamicGate']:
            for metric in METRICS:
                lookup = {(r['method'], r['seed'], r['month']): r[metric] for r in rows}
                d = np.array([[lookup['DynamicGate', s, m]-lookup[comparator, s, m]
                               for s in protocol['seeds']] for m in MONTHS])
                # Resample whole months, preserving each seed's paired difference.
                rng = np.random.default_rng(20261026)
                bootstrap = d.mean(axis=1)[rng.integers(0, len(MONTHS), (10000, len(MONTHS)))].mean(axis=1)
                lo, hi = np.quantile(bootstrap, [.025, .975])
                base = mean(lookup[comparator, s, m] for s in protocol['seeds'] for m in MONTHS)
                pairs.append(dict(comparison='DynamicGate - '+comparator, metric=metric,
                                  lower_is_better=metric in ['fpr', 'brier', 'ece'],
                                  mean_difference=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                                  relative_change_percent=float(d.mean()/base*100) if base != 0 else None))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'comparison.csv', summary)
    write_csv(output/'seed_means.csv', seed_rows)
    write_csv(output/'paired_intervals.csv', pairs)
    write_csv(output/'gate_activation.csv', statuses)
    notes = ['# 补充实验汇总', '', '全部预定方法与种子通过完整性检查。',
             'F1、实际阈值 FPR 与 ROC 插值 TPR@1%FPR 分别报告，不计算事后加权综合分数。',
             '月份 bootstrap 区间是本语料的描述性不确定性分析；未作多重比较校正，也未保持相邻月份自相关。',
             '门控触发仅证明机制被执行，效果仍须看 DynamicGate 对 FrozenGate 的配对比较。', '']
    dynamic = [s for s in statuses if s['method'] == 'DynamicGate']
    if dynamic and not any(s['used_positive_months'] for s in dynamic):
        notes.append('**动态查询门控未实际启用：不得将当前结果写成探索门控有效的证据。**')
    notes += ['', '| 方法 | F1 | 实际 FPR | TPR@1%FPR |', '|---|---:|---:|---:|']
    notes += [f'| {r["method"]} | {r["f1_mean"]:.6f} | {r["fpr_mean"]:.6f} | {r["tpr_at_1fpr_mean"]:.6f} |' for r in summary]
    (output/'结果摘要.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    write_json(output/'analysis_manifest.json', dict(protocol=protocol, source_sha256=hashes,
               analysis_sha256=digest(Path(__file__)), checks=ledger_checks,
               python=sys.version, numpy=np.__version__))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'runs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'protocol.json')
    args = parser.parse_args()
    result = summarize(args.runs, args.out, json.loads(args.protocol.read_text(encoding='utf-8')))
    print(f'Validated and summarized {len(result)} methods into {args.out}')
