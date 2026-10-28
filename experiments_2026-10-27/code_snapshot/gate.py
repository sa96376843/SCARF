"""Causal reliability feedback from uniformly sampled, budgeted audit labels."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class GateConfig:
    target_auc: float = 0.995
    scale: float = 0.05
    audit_fraction: float = 0.20
    window_months: int = 3
    min_per_class: int = 5

    def __post_init__(self):
        if not 0 < self.target_auc <= 1 or not 0 < self.scale <= 1:
            raise ValueError('Invalid AUC target or scale')
        if not 0 < self.audit_fraction < 1:
            raise ValueError('audit_fraction must be strictly between zero and one')
        if self.window_months < 1 or self.min_per_class < 1:
            raise ValueError('Positive window and class counts required')


def validate_scores(scores):
    p = np.asarray(scores, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Scores must be a finite probability vector')
    return p


def epsilon(auc, config):
    if not np.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError('Invalid reliability AUC')
    return float(np.clip((config.target_auc - auc) / config.scale, 0, 1))


class ReliabilityGate:
    """Stores PRE-update scores only, never rescores training data for feedback.

    Pair-weighted average of within-month audit AUCs avoids comparing score
    scales across different models. There is one-batch delay for acquisition.
    Small/single-class windows retain the previous state with an explicit flag.
    """
    def __init__(self, initial_auc, config=GateConfig()):
        self.config = config
        self.rho = float(initial_auc)
        self.explore = epsilon(self.rho, config)
        self.history = []
        self.last_step = -1

    def observe(self, step, scores, labels, threshold):
        from sklearn.metrics import roc_auc_score
        if step <= self.last_step:
            raise ValueError('Audit batches must arrive strictly chronologically')
        p = validate_scores(scores)
        y = np.asarray(labels)
        if y.shape != p.shape or not np.isin(y, [0, 1]).all():
            raise ValueError('Audit labels must be binary and match the scores')
        if not np.isfinite(threshold):
            raise ValueError('Invalid prediction threshold')
        self.last_step = step
        self.history.append((step, p.copy(), y.copy(), float(threshold)))
        self.history = [b for b in self.history if b[0] > step - self.config.window_months]
        positives = negatives = pairs = 0
        weighted_auc = 0.0
        for _, bp, by, _ in self.history:
            n1, n0 = int((by == 1).sum()), int((by == 0).sum())
            if n1 and n0:
                positives += n1
                negatives += n0
                pairs += n1 * n0
                weighted_auc += roc_auc_score(by, bp) * n1 * n0
        sufficient = (positives >= self.config.min_per_class and
                      negatives >= self.config.min_per_class and pairs > 0)
        measured = float(weighted_auc / pairs) if pairs else None
        before = self.explore
        if sufficient:
            self.rho = measured
            self.explore = epsilon(self.rho, self.config)
        benign = y == 0
        return dict(reliability_auc=self.rho, measured_audit_auc=measured,
                    audit_positive=positives, audit_negative=negatives,
                    audit_pairs=pairs, feedback_valid=sufficient,
                    feedback_status='updated' if sufficient else 'hold_insufficient_audit_classes',
                    epsilon_before=before, epsilon_after=self.explore,
                    current_audit_fpr=float((p[benign] >= threshold).mean()) if benign.any() else None,
                    history_steps=[b[0] for b in self.history], config=asdict(self.config))


def acquire(scores, budget, explore, config, audit_rng, selection_rng):
    """Select a uniform audit BEFORE margin acquisition, using no labels.

    Audit count is INCLUDED in B. Extra exploration is epsilon of the remainder.
    Separate RNGs keep audit IDs matched across ablations despite other choices.
    """
    p = validate_scores(scores)
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError('Budget must be a positive integer')
    if not np.isfinite(explore) or not 0 <= explore <= 1:
        raise ValueError('Exploration must be in [0, 1]')
    b = min(budget, len(p))
    if b == 0:
        raise ValueError('Cannot query an empty batch')
    na = min(b, max(1, int(round(b * config.audit_fraction))))
    audit = audit_rng.choice(len(p), na, replace=False)
    available = np.setdiff1d(np.arange(len(p)), audit)
    nr = int(round((b - na) * explore))
    nu = b - na - nr
    risk = 1 - np.abs(2*p - 1)
    uncertain = available[np.argsort(risk[available], kind='stable')[-nu:]] if nu else np.array([], dtype=int)
    remaining = np.setdiff1d(available, uncertain)
    random = selection_rng.choice(remaining, nr, replace=False) if nr else np.array([], dtype=int)
    ids = np.concatenate([audit, uncertain, random]).astype(int)
    roles = np.array(['audit']*na + ['uncertainty']*nu + ['exploration']*nr)
    assert len(ids) == b and len(np.unique(ids)) == b
    return ids, roles


def pseudo_indices(scores, queried, cap=500):
    """Genuine queried labels take precedence; no duplicate pseudo-label rows."""
    p = validate_scores(scores)
    if cap < 0:
        raise ValueError('Negative pseudo-label cap')
    allowed = np.ones(len(p), dtype=bool)
    allowed[np.asarray(queried, dtype=int)] = False
    ids, labels = [], []
    for cls in [0, 1]:
        candidates = np.flatnonzero(allowed & ((p <= .005) if cls == 0 else (p >= .995)))
        if len(candidates) > cap:
            candidates = candidates[np.argsort(np.abs(p[candidates]-cls), kind='stable')[:cap]]
        ids.extend(candidates.tolist())
        labels.extend([cls]*len(candidates))
    return np.asarray(ids, dtype=int), np.asarray(labels, dtype=np.int8)
