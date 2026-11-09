"""The B1 reference recipe (`b1_reference_v1`): the frozen real-only baseline.

Contract: experiments/baseline_reference/PROTOCOL.md. In short:
- full-band route: channel mean, native -> 16 kHz directly, centred window;
- two representations, always reported separately: MFCC26 and the BEATs
  layer mean (mean of the 12 layer means);
- every training recording, balanced class weights;
- train-only StandardScaler + lbfgs logistic regression, fixed argmax rule;
- C chosen from CS by mean inner leave-one-group-out AUROC over the training
  groups; ties go to the smaller C.

Changing any of this makes a new recipe version, not an edit of this one.
"""
from __future__ import annotations

import numpy as np

from abvid.learning.grouped_search import fit, metrics

RECIPE_ID = "b1_reference_v1"
REPRESENTATIONS = ("mfcc26", "beats_avg")
CS = (0.001, 0.01, 0.1, 1.0)
INTERMEDIATE_RATE_HZ = 16000


def preprocessing(h1_preprocessing):
    """The full-band route: the H1 observation settings without the 8 kHz step."""
    return dict(h1_preprocessing, intermediate_rate_hz=INTERMEDIATE_RATE_HZ)


def representation(name, beats_layers=None, mfcc=None):
    """Feature matrix for one recipe representation from cached encoder outputs."""
    if name == "beats_avg":
        return np.asarray(beats_layers).mean(axis=1)
    if name == "mfcc26":
        return np.asarray(mfcc)
    raise ValueError(f"Not a reference representation: {name}")


def inner_scores(x, y, groups, cs=CS):
    """Per C, the metrics of each inner leave-one-group-out fit, groups in sorted order."""
    groups = np.asarray(groups)
    out = []
    for c in cs:
        scores = []
        for group in sorted(set(groups.tolist())):
            train, test = groups != group, groups == group
            model = fit(x[train], y[train], c, True)
            scores.append(metrics(y[test], model.predict_proba(x[test])[:, 1]))
        out.append(scores)
    return out


def select_c(scores, cs=CS):
    """The C with the highest mean inner AUROC; ties go to the smaller C."""
    means = [float(np.mean([m["auroc"] for m in per_group])) for per_group in scores]
    return cs[max(range(len(cs)), key=lambda i: (means[i], -i))], means


def fit_reference(x, y, groups, cs=CS):
    """Choose C on the training groups only, then fit on all training recordings."""
    scores = inner_scores(x, y, groups, cs)
    c, means = select_c(scores, cs)
    return fit(x, y, c, True), {"c": c, "inner_auroc_by_c": dict(zip(map(str, cs), means)), "inner": scores}
