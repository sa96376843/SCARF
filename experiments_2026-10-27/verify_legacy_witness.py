"""Compare one original-v3 update against saved real-data predictions in this environment."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code_snapshot/legacy'))
from run_scarf_v3 import fit_ensemble, predict, acquisition, pseudo_batch
from run_experiments import weighted_quantile


def main():
    cache = next((ROOT / 'cache').glob('*/X384.npy'))
    X = np.load(cache, mmap_mode='r')
    meta = pq.read_table(ROOT.parent / 'data/bodmas-timestamps.parquet',
                         columns=['Label', 'timestamp']).to_pandas()
    month = pd.to_datetime(meta.timestamp, utc=True, format='mixed').dt.strftime('%Y-%m')
    y = meta.Label.to_numpy(np.int8)
    tr = np.flatnonzero(month == '2019-08')
    cal = np.flatnonzero(month == '2019-09')
    seed, budget = 13, 100
    rng = np.random.default_rng(seed + budget)
    clfs = fit_ensemble(X, y, tr, seed, len(tr))
    pc = predict(clfs, np.asarray(X[cal]))
    threshold = float(np.quantile(pc[y[cal] == 0], .99))
    run = ROOT / 'runs/LegacyV3_seed13_B100'
    result = []
    for name in ['2019-10', '2019-11']:
        ids = np.flatnonzero(month == name)
        p = predict(clfs, np.asarray(X[ids]))
        with np.load(run / f'predictions_{name}.npz') as saved:
            error = float(np.max(np.abs(p - saved['probability'])))
            threshold_error = abs(threshold - float(saved['threshold']))
        assert error < 1e-12 and threshold_error < 1e-12
        result.append({'month': name, 'prediction_max_abs_error': error,
                       'threshold_abs_error': threshold_error})
        if name == '2019-10':
            q, _, _ = acquisition(p, budget, 0.0, rng)
            qids = ids[q]
            actual_queries = pd.read_csv(run / 'queries.csv')
            assert np.array_equal(qids, actual_queries.loc[actual_queries.month == name, 'index'].to_numpy())
            px, py = pseudo_batch(np.asarray(X[ids]), p)
            replay = np.concatenate([tr, qids])
            clfs = fit_ensemble(X, y, replay, seed, len(tr), pseudo_X=px, pseudo_y=py, pseudo_weight=.1)
            pool = np.concatenate([cal, qids])
            cp = predict(clfs, np.asarray(X[pool]))
            cw = np.concatenate([np.ones(len(cal)), np.full(len(qids), 4.)])
            threshold = weighted_quantile(cp[y[pool] == 0], .99, cw[y[pool] == 0])
    report = {'status': 'PASS', 'seed': seed, 'budget': budget,
              'scope': 'Original v3 functions, initial fit and first update, exact same current data/cache/environment.',
              'member_fits': 6, 'checks': result,
              'historical_difference_note': 'This verifies the new LegacyV3 path against original functions; it does not identify the cause of differences from archived historical runs.'}
    (ROOT / 'analysis/legacy_function_witness.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
