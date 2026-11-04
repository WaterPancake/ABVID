"""Background-shortcut probe: Alex writes this module.

Before writing it, answer the open decisions in experiments/background_shortcut/PROTOCOL.md
and freeze the protocol. The data side (roles, site folds, separation) is
`abvid.audit.background_shortcut`; the run record is `abvid.audit.run_record`.

Use the H1 head unchanged: `abvid.learning.linear.fixed_fit` with the `classifier` block of
configs/reference/h1.json (StandardScaler + L2 logistic regression, fitted on training rows
only). Import scikit-learn inside functions, never at module level: the cast environment
collects tests without it.
"""


def fit_head(X_train, y_train, classifier):
    """One H1 head for one fold's training rows."""
    raise NotImplementedError("write with abvid.learning.linear.fixed_fit")


def p_truck(model, X):
    """P(truck) for each row."""
    raise NotImplementedError


def primary_statistic(fold_scores, summary):
    """The single number the protocol names as the answer. Decide its form in PROTOCOL.md."""
    raise NotImplementedError


def control(fold_scores, summary, seed):
    """The run that should come out flat if the probe measures what it claims."""
    raise NotImplementedError
