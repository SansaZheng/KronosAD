"""Frozen unsupervised thresholds and ordinary, unadjusted test metrics."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, auc)

from anomaly.pot import POTThreshold


def calibrate(scores, risk=.01, quantile=.85):
    # No labels accepted. Failure is explicit: no test-F1 search or silent fallback.
    return POTThreshold.fit(scores, initial_quantile=quantile, risk=risk, min_peaks=20)


def evaluate(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError('Need matching one-dimensional labels and scores')
    if not np.isin(labels, [-1, 0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid labels or scores')
    known = labels != -1
    y, s = labels[known], scores[known]
    if not len(y):
        raise ValueError('No labelled test examples')
    tn, fp, fn, tp = map(int, confusion_matrix(y, s >= threshold, labels=[0, 1]).ravel())
    p, r = tp / (tp + fp) if tp+fp else 0., tp / (tp+fn) if tp+fn else 0.
    result = dict(evaluated=len(y), unknown_excluded=int((~known).sum()),
                  positives=int(y.sum()), prevalence=float(y.mean()),
                  tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r,
                  f1=2*p*r/(p+r) if p+r else 0., fpr=fp/(fp+tn) if fp+tn else None,
                  alarm_fraction=float((s >= threshold).mean()))
    if len(np.unique(y)) == 2:
        precision, recall, _ = precision_recall_curve(y, s)
        result.update(average_precision=float(average_precision_score(y, s)),
                      pr_auc_trapezoid=float(auc(recall, precision)),
                      roc_auc=float(roc_auc_score(y, s)))
    else:
        result.update(average_precision=None, pr_auc_trapezoid=None, roc_auc=None)
    return result
