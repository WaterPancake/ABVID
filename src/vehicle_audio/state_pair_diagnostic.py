"""Explicitly authorized within-session diagnostic, never a benchmark splitter."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def validate_split(train, test, *, allow_within_session=False):
    """Reject shared sessions by default; never permit overlapping source samples."""
    for rows in (train, test):
        if {r["pair_label"] for r in rows} != {0, 1}:
            raise ValueError("both pair members are required in each partition")
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate sample IDs")
        for row in rows:
            if not 0 <= row["window_start_seconds"] < row["window_end_seconds"]:
                raise ValueError("invalid window bounds")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in test}:
        raise ValueError("train/test sample overlap")
    shared = sorted(
        {r["recording_session"] for r in train} & {r["recording_session"] for r in test}
    )
    if shared and not allow_within_session:
        raise ValueError("shared sessions require diagnostic-only opt-in")
    for a in train:
        for b in test:
            if a["source_id"] == b["source_id"] and (
                a["window_start_seconds"] < b["window_end_seconds"]
                and b["window_start_seconds"] < a["window_end_seconds"]
            ):
                raise ValueError("train/test source audio overlaps")
    return shared


def make_tasks(records, inventory, config, *, allow_within_session=False):
    identities = {}
    for interval in inventory["intervals"]:
        for sample_id in interval["sample_ids"]:
            if sample_id in identities:
                raise ValueError("duplicate identity assignment")
            identities[sample_id] = interval["reviewed_interval_model"]
    tasks = []
    for pair in config["pairs"]:
        selected = []
        for label, member in enumerate(pair["members"]):
            for index, record in enumerate(records):
                if (
                    record["source_id"] == member["source_id"]
                    and identities[record["sample_id"]] == member["model"]
                    and record["operating_condition"] in {"idle", "steady_speed"}
                ):
                    selected.append(
                        {
                            **record,
                            "parent_index": index,
                            "pair_label": label,
                            "reviewed_interval_model": member["model"],
                        }
                    )
        for train_state, test_state in config["directions"]:
            if {train_state, test_state} != {"idle", "steady_speed"}:
                raise ValueError(
                    "only idle/steady-speed cross-state directions allowed"
                )
            train = sorted(
                [r for r in selected if r["operating_condition"] == train_state],
                key=lambda r: r["parent_index"],
            )
            test = sorted(
                [r for r in selected if r["operating_condition"] == test_state],
                key=lambda r: r["parent_index"],
            )
            shared = validate_split(
                train, test, allow_within_session=allow_within_session
            )
            tasks.append(
                {
                    "id": f"{pair['id']}__{train_state}_to_{test_state}",
                    "pair": pair,
                    "train_state": train_state,
                    "test_state": test_state,
                    "shared_sessions": shared,
                    "benchmark_eligible": False,
                    "domain": config["domain"],
                    "train": train,
                    "test": test,
                }
            )
    return tasks


def fit_head(features, labels, config):
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("invalid training features")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("training requires both pair labels")
    weights = np.array([len(y) / (2 * np.count_nonzero(y == k)) for k in y])
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        random_state=config["seed"],
    ).fit(scaler.transform(x), y, sample_weight=weights)
    if np.max(model.n_iter_) >= config["max_iter"]:
        raise RuntimeError("logistic regression did not converge")
    state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "classes": model.classes_.tolist(),
    }
    return state, scaler, model


def replay_head(features, state):
    x = np.asarray(features, dtype=np.float64)
    return expit(
        ((x - np.asarray(state["mean"])) / np.asarray(state["scale"]))
        @ np.asarray(state["coefficient"])
        + state["intercept"]
    )


def metrics(labels, probability, threshold=0.5):
    y, probability = np.asarray(labels), np.asarray(probability)
    if set(y.tolist()) != {0, 1} or y.shape != probability.shape:
        raise ValueError("invalid scoring inputs")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("invalid probability")
    predicted = (probability >= threshold).astype(int)
    confusion = np.array(
        [
            [np.count_nonzero((y == actual) & (predicted == guess)) for guess in (0, 1)]
            for actual in (0, 1)
        ]
    )
    recalls = confusion.diagonal() / confusion.sum(axis=1)
    return {
        "balanced_accuracy": float(recalls.mean()),
        "accuracy": float((y == predicted).mean()),
        "recall_A": float(recalls[0]),
        "recall_B": float(recalls[1]),
        "n_A": int(np.count_nonzero(y == 0)),
        "n_B": int(np.count_nonzero(y == 1)),
        "confusion_true_rows_predicted_columns_A_B": confusion.tolist(),
    }
