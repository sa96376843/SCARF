"""Mechanism and integration tests; synthetic fixtures are NOT paper results."""
from pathlib import Path
import csv
import json
import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate import GateConfig, ReliabilityGate, acquire, pseudo_indices, epsilon
from run_dynamic import run_stream, variant, configured_variant, LightGBMBackend, MONTHS, write_json


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


class FixedScoreBackend:
    """Transparent test double; column zero is an externally set score."""
    def __init__(self):
        self.fits = []

    def fit(self, X, y, weights):
        self.fits.append((X.copy(), y.copy(), weights.copy()))

    def predict(self, X):
        return X[:, 0].copy()

    def save(self, folder, label):
        pass


def fixture(month_count=5):
    rng = np.random.default_rng(44)
    X = rng.random((80 + month_count*100, 3))
    X[:, 1] = np.arange(len(X))  # Stable IDs used to inspect replay.
    y = (X[:, 0] > .5).astype(np.int8)
    # Perfect source calibration, reversed ranking in the first stream month.
    y[80:180] = 1-y[80:180]
    tests = [(f'2020-{i+1:02d}', np.arange(80+i*100, 180+i*100)) for i in range(month_count)]
    return X, y, np.arange(40), np.arange(40, 80), tests


class GateTests(unittest.TestCase):
    def test_mapping_and_invalid_config(self):
        gc = GateConfig()
        self.assertEqual(epsilon(.998, gc), 0)
        self.assertAlmostEqual(epsilon(.990, gc), .1)
        self.assertAlmostEqual(epsilon(.970, gc), .5)
        with self.assertRaises(ValueError):
            GateConfig(audit_fraction=0)
        with self.assertRaises(ValueError):
            epsilon(float('nan'), gc)

    def test_degrade_and_recover_without_forcing(self):
        gate = ReliabilityGate(.998, GateConfig(window_months=2, min_per_class=2))
        labels = np.array([0, 0, 1, 1])
        good = np.array([.1, .2, .8, .9])
        self.assertEqual(gate.observe(0, good, labels, .5)['epsilon_after'], 0)
        self.assertEqual(gate.observe(1, 1-good, labels, .5)['epsilon_after'], 1)
        gate.observe(2, good, labels, .5)
        self.assertEqual(gate.observe(3, good, labels, .5)['epsilon_after'], 0)
        with self.assertRaises(ValueError):
            gate.observe(3, good, labels, .5)

    def test_single_class_holds_without_fabricating_auc(self):
        gate = ReliabilityGate(.97, GateConfig())
        state = gate.observe(0, [.1]*10, [0]*10, .5)
        self.assertFalse(state['feedback_valid'])
        self.assertIsNone(state['measured_audit_auc'])
        self.assertAlmostEqual(state['epsilon_after'], .5)

    def test_auc_compares_only_within_month(self):
        gate = ReliabilityGate(.998, GateConfig(min_per_class=2))
        gate.observe(0, [.8, .9], [0, 1], .5)
        state = gate.observe(1, [.1, .2], [0, 1], .5)
        self.assertTrue(state['feedback_valid'])
        self.assertEqual(state['measured_audit_auc'], 1)

    def test_query_budget_roles_and_matched_audit(self):
        scores = np.linspace(0, 1, 300)
        audits = []
        for eps in [0, .1, 1]:
            q, roles = acquire(scores, 100, eps, GateConfig(),
                               np.random.default_rng(1), np.random.default_rng(2))
            self.assertEqual(len(q), 100)
            self.assertEqual(len(set(q)), 100)
            self.assertEqual(int((roles == 'audit').sum()), 20)
            self.assertEqual(int((roles == 'exploration').sum()), round(80*eps))
            audits.append(q[roles == 'audit'])
        np.testing.assert_array_equal(audits[0], audits[1])
        np.testing.assert_array_equal(audits[0], audits[2])
        q, _ = acquire(scores[:3], 100, 1, GateConfig(),
                       np.random.default_rng(1), np.random.default_rng(2))
        self.assertEqual(set(q), {0, 1, 2})

    def test_pseudo_never_duplicates_genuine_labels(self):
        q, y = pseudo_indices([.001, .002, .999, .998, .5], [0, 2], cap=1)
        np.testing.assert_array_equal(q, [1, 3])
        np.testing.assert_array_equal(y, [0, 1])

    def test_protocol_weights_reach_variants(self):
        protocol = dict(query_weight=3., pseudo_weight=.2)
        self.assertEqual(configured_variant('DynamicGate', protocol).query_weight, 3.)
        self.assertEqual(configured_variant('FrozenGate', protocol).pseudo_weight, .2)
        self.assertEqual(configured_variant('EnsembleMargin', protocol).query_weight, 1.)
        self.assertEqual(configured_variant('NoPseudo', protocol).pseudo_weight, 0.)


class StreamTests(unittest.TestCase):
    def run_fixture(self, folder, name='DynamicGate', data=None, budget=40):
        data = data or fixture()
        backend = FixedScoreBackend()
        result = run_stream(*data, variant(name), 42, budget,
                            GateConfig(min_per_class=1), backend, folder)
        return backend, result

    def test_feedback_is_lagged_and_budgeted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            _, result = self.run_fixture(tmp)
            feedback = read(Path(tmp)/'feedback.csv')
            self.assertEqual(float(feedback[0]['query_epsilon_used']), 0)
            self.assertEqual(float(feedback[0]['epsilon_after']), 1)
            self.assertEqual(float(feedback[1]['query_epsilon_used']), 1)
            self.assertEqual(float(feedback[0]['pseudo_weight']), 0)
            self.assertEqual(result['analyst_labels'], 5*40)
            self.assertEqual(result['audit_n'], 5*8)
            self.assertFalse(result['gate_efficacy_established'])

    def test_unqueried_labels_cannot_change_policy_or_training(self):
        data = fixture(3)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            a, _ = self.run_fixture(Path(tmp)/'first', data=data)
            ledger = read(Path(tmp)/'first/queries.csv')
            queried = {int(r['index']) for r in ledger}
            X, y, tr, cal, tests = data
            altered = y.copy()
            for _, idx in tests:
                untouched = [i for i in idx if i not in queried]
                altered[untouched] = 1-altered[untouched]
            b, _ = self.run_fixture(Path(tmp)/'altered', data=(X, altered, tr, cal, tests))
            self.assertEqual(ledger, read(Path(tmp)/'altered/queries.csv'))
            for af, bf in zip(a.fits, b.fits):
                for av, bv in zip(af, bf):
                    np.testing.assert_array_equal(av, bv)
            fa, fb = read(Path(tmp)/'first/feedback.csv'), read(Path(tmp)/'altered/feedback.csv')
            for ar, br in zip(fa, fb):
                for key in ['reliability_auc', 'epsilon_after', 'pseudo_weight', 'query_epsilon_used']:
                    self.assertEqual(ar[key], br[key])

    def test_clean_replay_ablation_preserves_initial_data(self):
        data = fixture(5)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            backend, _ = self.run_fixture(tmp, 'RecentReplay', data, budget=10)
            for X, _, _ in backend.fits:
                np.testing.assert_array_equal(X[:40], data[0][:40])
            feedback = read(Path(tmp)/'feedback.csv')
            self.assertEqual(int(feedback[-1]['replay_queried_n']), 30)
            self.assertTrue(all(int(r['replay_initial_n']) == 40 for r in feedback))

    def test_frozen_gate_and_threshold_controls(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            self.run_fixture(Path(tmp)/'gate', 'FrozenGate')
            feedback = read(Path(tmp)/'gate/feedback.csv')
            self.assertTrue(any(float(r['epsilon_after']) > 0 for r in feedback))
            self.assertTrue(all(float(r['query_epsilon_used']) == 0 for r in feedback))
            self.assertTrue(all(float(r['pseudo_weight']) == .1 for r in feedback))
            self.run_fixture(Path(tmp)/'threshold', 'FrozenThreshold')
            rows = read(Path(tmp)/'threshold/monthly_metrics.csv')
            self.assertEqual(len({r['threshold'] for r in rows}), 1)

    def test_rejects_overlapping_partitions(self):
        X, y, tr, cal, tests = fixture(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaises(ValueError):
                self.run_fixture(tmp, data=(X, y, tr, tr, tests))


class LightGBMIntegrationTests(unittest.TestCase):
    def test_legacy_runner_matches_original_algorithm(self):
        # This tiny synthetic fixture checks compatibility, not research performance.
        import run_scarf_v3 as old
        from run_experiments import metrics, weighted_quantile
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(7)
        X = rng.normal(size=(1200, 6)).astype(np.float32)
        y = (X[:, 0]+.2*X[:, 1] > 0).astype(np.int8)
        tr, cal = np.arange(400), np.arange(400, 600)
        tests = [('2020-01', np.arange(600, 800)), ('2020-02', np.arange(800, 1000)),
                 ('2020-03', np.arange(1000, 1200))]
        cfg = dict(members=3, trees=12, learning_rate=.05, threads=1)
        expected, expected_queries = [], []
        with patch.object(old, 'ENSEMBLE_TREES', 12):
            clfs = old.fit_ensemble(X, y, tr, 42, len(tr))
            pc = old.predict(clfs, X[cal])
            thr = float(np.quantile(pc[y[cal] == 0], .99))
            eps = float(np.clip((.995-roc_auc_score(y[cal], pc))/.05, 0, 1))
            recent = []
            qrng = np.random.default_rng(142)
            for month, idx in tests:
                p = old.predict(clfs, X[idx])
                expected.append(metrics(y[idx], p, thr))
                px, py = old.pseudo_batch(X[idx], p)
                q, _, _ = old.acquisition(p, 100, eps, qrng)
                expected_queries.extend(idx[q].tolist())
                recent.extend(idx[q].tolist())
                replay = np.concatenate([tr, recent])
                clfs = old.fit_ensemble(X, y, replay, 42, len(tr), pseudo_X=px,
                                        pseudo_y=py, pseudo_weight=.1*(1-eps))
                pool = np.concatenate([cal, recent])
                cp = old.predict(clfs, X[pool])
                cw = np.concatenate([np.ones(len(cal)), np.full(len(recent), 4.)])
                thr = weighted_quantile(cp[y[pool] == 0], .99, cw[y[pool] == 0])
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            run_stream(X, y, tr, cal, tests, variant('LegacyV3'), 42, 100,
                       GateConfig(), LightGBMBackend(42, cfg), tmp)
            actual = read(Path(tmp)/'monthly_metrics.csv')
            for a, e in zip(actual, expected):
                for key in ['auc', 'tpr_at_1fpr', 'f1', 'fpr', 'brier', 'threshold']:
                    self.assertAlmostEqual(float(a[key]), e[key], places=12)
            self.assertEqual([int(r['index']) for r in read(Path(tmp)/'queries.csv')], expected_queries)
            self.assertEqual(len(list(Path(tmp).glob('after_*_member*.txt'))), 9)
            dynamic_dir = Path(tmp)/'dynamic'
            result = run_stream(X, y, tr, cal, tests, variant('DynamicGate'), 42, 100,
                                GateConfig(), LightGBMBackend(42, cfg), dynamic_dir)
            self.assertEqual(result['analyst_labels'], 300)
            self.assertEqual(result['audit_n'], 60)
            ledger = read(dynamic_dir/'queries.csv')
            pseudo = read(dynamic_dir/'pseudo_labels.csv') if (dynamic_dir/'pseudo_labels.csv').exists() else []
            qkeys = {(r['month'], r['index']) for r in ledger}
            self.assertFalse(qkeys & {(r['month'], r['index']) for r in pseudo})


class SummaryTests(unittest.TestCase):
    def test_complete_matrix_and_missing_run_rejection(self):
        from summarize import summarize
        protocol = json.loads((ROOT/'protocol.json').read_text())
        protocol['methods'] = ['FrozenGate', 'DynamicGate']
        protocol['seeds'] = [13, 42]
        X, y, tr, cal, batches = fixture(12)
        tests = [(month, ids) for month, (_, ids) in zip(MONTHS, batches)]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            for method in protocol['methods']:
                for seed in protocol['seeds']:
                    folder = tmp/'runs'/f'{method}_seed{seed}_B100'
                    result = run_stream(X, y, tr, cal, tests, variant(method), seed, 100,
                                        GateConfig(), FixedScoreBackend(), folder)
                    write_json(folder/'status.json', {'status': 'COMPLETE', **result})
                    write_json(folder/'run_manifest.json', dict(protocol=protocol, seed=seed,
                               variant={'name': method}, source_sha256={'fixture':'synthetic'},
                               dataset_sha256='synthetic-test-only', packages={'test': 'only'}))
            summary = summarize(tmp/'runs', tmp/'summary', protocol)
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(read(tmp/'summary/paired_intervals.csv')), 6)
            with self.assertRaises(FileExistsError):
                summarize(tmp/'runs', tmp/'summary', protocol)
            missing = dict(protocol, seeds=[13, 42, 87])
            # Different protocol is rejected even before a missing run is reached.
            with self.assertRaises(ValueError):
                summarize(tmp/'runs', tmp/'missing', missing)
            self.assertFalse((tmp/'missing').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
