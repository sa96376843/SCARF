"""Check saved predictions, label budgets, feedback timing, and historical differences."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code_snapshot'))
from gate import GateConfig, ReliabilityGate


def main():
    protocol = json.loads((ROOT / 'code_snapshot/protocol.json').read_text())
    meta = pq.read_table(ROOT.parent / 'data/bodmas-timestamps.parquet',
                         columns=['Label', 'timestamp']).to_pandas()
    months = pd.to_datetime(meta.timestamp, utc=True, format='mixed').dt.strftime('%Y-%m')
    labels = meta.Label.to_numpy(dtype=np.int8)
    historical = {
        'LegacyV3': pd.read_csv(ROOT.parent / 'SCARF_TDSC/results_final_scarf/monthly_metrics.csv'),
        'EnsembleMargin': pd.read_csv(ROOT.parent / 'SCARF_TDSC/results_extended_comparisons/monthly_metrics.csv'),
    }
    expected_months = [f'2019-{m:02d}' for m in range(10, 13)] + [f'2020-{m:02d}' for m in range(1, 10)]
    audits = {}
    records, differences = [], []
    prediction_max_error = 0.0
    feedback_max_error = 0.0
    for method in protocol['methods']:
        for seed in protocol['seeds']:
            run = ROOT / 'runs' / f'{method}_seed{seed}_B100'
            status = json.loads((run / 'status.json').read_text())
            assert status['status'] == 'COMPLETE', run
            metric = pd.read_csv(run / 'monthly_metrics.csv')
            feedback = pd.read_csv(run / 'feedback.csv')
            queries = pd.read_csv(run / 'queries.csv')
            pseudo = pd.read_csv(run / 'pseudo_labels.csv') if (run / 'pseudo_labels.csv').exists() else None
            assert metric.month.tolist() == expected_months
            assert feedback.month.tolist() == expected_months
            assert (feedback.replay_initial_n == 4051).all()
            initial_auc = float(metric.iloc[0].reliability_auc_before)
            gate = ReliabilityGate(initial_auc, GateConfig(**protocol['gate']))
            for step, month in enumerate(expected_months):
                row, fb = metric.iloc[step], feedback.iloc[step]
                with np.load(run / f'predictions_{month}.npz') as saved:
                    ids, y, p = saved['index'], saved['label'], saved['probability']
                    threshold = float(saved['threshold'])
                assert np.array_equal(ids, np.flatnonzero(months == month))
                assert np.array_equal(y, labels[ids])
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                if method == 'FrozenThreshold':
                    with np.load(ROOT / 'runs' / f'DynamicGate_seed{seed}_B100' / f'predictions_{month}.npz') as matched:
                        assert np.array_equal(p, matched['probability']), 'Threshold ablation changed ranking scores'
                assert abs(threshold - row.threshold) < 1e-12
                decision = p >= threshold
                counts = {'tp': int((decision & (y == 1)).sum()),
                          'fp': int((decision & (y == 0)).sum()),
                          'tn': int((~decision & (y == 0)).sum()),
                          'fn': int((~decision & (y == 1)).sum())}
                assert all(int(row[k]) == v for k, v in counts.items())
                fpr, tpr, _ = roc_curve(y, p)
                recalculated = {'f1': f1_score(y, decision),
                                'fpr': counts['fp'] / (counts['fp'] + counts['tn']),
                                'auc': roc_auc_score(y, p),
                                'tpr_at_1fpr': np.interp(.01, fpr, tpr),
                                'brier': brier_score_loss(y, p)}
                error = max(abs(float(row[k]) - v) for k, v in recalculated.items())
                prediction_max_error = max(prediction_max_error, error)
                assert error < 1e-12
                q = queries[queries.month == month]
                assert len(q) == 100 and q['index'].nunique() == 100
                positions = pd.Index(ids).get_indexer(q['index'])
                assert (positions >= 0).all()
                assert np.array_equal(q.label.to_numpy(), y[positions])
                assert np.allclose(q.pre_update_probability, p[positions], atol=1e-14, rtol=0)
                assert int(fb.audit_count + fb.extra_exploration_count + fb.uncertainty_count) == 100
                if step:
                    assert abs(float(fb.query_epsilon_used) - float(feedback.iloc[step-1].query_epsilon_next)) < 1e-12
                    assert abs(threshold - float(feedback.iloc[step-1].threshold_next)) < 1e-12
                if method not in ['LegacyV3', 'EnsembleMargin']:
                    audit = q[q.role == 'audit']
                    assert len(audit) == 20
                    key = (seed, month)
                    selected = tuple(audit['index'])
                    if key in audits: assert audits[key] == selected
                    audits[key] = selected
                    audit_pos = pd.Index(ids).get_indexer(audit['index'])
                    recomputed = gate.observe(step, p[audit_pos], y[audit_pos], threshold)
                    err = max(abs(recomputed['reliability_auc'] - fb.reliability_auc),
                              abs(recomputed['epsilon_after'] - fb.epsilon_after))
                    feedback_max_error = max(feedback_max_error, err)
                    assert err < 1e-12
                    assert recomputed['feedback_status'] == fb.feedback_status
                    if pseudo is not None:
                        batch = pseudo[pseudo.month == month]
                        assert set(batch['index']).isdisjoint(set(q['index']))
                        assert batch['index'].nunique() == len(batch)
                expected_replay = min((step+1)*100, 300) if method == 'RecentReplay' else (step+1)*100
                assert int(fb.replay_queried_n) == expected_replay
            if method == 'FrozenThreshold': assert metric.threshold.nunique() == 1
            if method in historical:
                old_name = 'SCARF' if method == 'LegacyV3' else method
                old = historical[method]
                old = old[(old.method == old_name) & (old.seed == seed) & (old.budget == 100)]
                merged = metric.merge(old, on='month', suffixes=('_current', '_historical'), validate='one_to_one')
                assert len(merged) == 12
                for _, row in merged.iterrows():
                    for name in ['f1', 'fpr', 'tpr_at_1fpr', 'auc', 'brier', 'ece']:
                        differences.append({'method': method, 'seed': seed, 'month': row.month,
                                            'metric': name, 'current': row[name+'_current'],
                                            'historical': row[name+'_historical'],
                                            'difference': row[name+'_current'] - row[name+'_historical']})
            records.append({'method': method, 'seed': seed, 'months': 12,
                            'analyst_labels': len(queries), 'audits': int(feedback.audit_count.sum()),
                            'initial_fit_seconds': status['initial_fit_seconds'],
                            'update_seconds': status['update_seconds'], 'score_seconds': status['score_seconds']})
    manifest = json.loads((ROOT.parent / 'dynamic_gate_2026-10-26/original_artifact_sha256.json').read_text())
    changed = []
    for name, expected in manifest.items():
        path = ROOT.parent / 'SCARF_TDSC' / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected: changed.append(name)
    assert not changed, changed
    output = ROOT / 'analysis'
    pd.DataFrame(differences).to_csv(output / 'historical_reproduction_differences.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'cost_and_budget.csv', index=False)
    report = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
              'runs_checked': len(records), 'monthly_predictions_checked': 12*len(records),
              'prediction_metric_max_abs_error': float(prediction_max_error),
              'recomputed_feedback_max_abs_error': float(feedback_max_error),
              'audit_ids_matched_across_new_variants': True, 'analyst_budget_valid': True,
              'initial_training_set_retained': True, 'new_pseudo_and_query_sets_disjoint': True,
              'feedback_applied_to_next_month_only': True,
              'threshold_ablation_preserves_probability_scores': True,
              'historical_files_checked': len(manifest), 'historical_files_changed': changed,
              'historical_result_note': 'Current-environment reruns are compared to each other; differences from historical CSVs are retained separately.'}
    (output / 'postrun_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
