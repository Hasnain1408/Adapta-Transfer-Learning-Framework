"""Classifier factory and fit-on-source/score-on-target evaluation.

``evaluate_classifier`` is shared by ``RunTransferLearningUseCase.execute``
for both the adapted run and its no-adaptation baseline, so the two stay
strictly comparable: same classifier, same averaging fallback, same AUC
handling.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from domain.models import EvaluationMetrics


def build_classifier(name: str):
    """Return a fitted-ready sklearn classifier for the given *name*."""
    name = name.strip()
    if name == "SVM":
        # `probability` deliberately left unset: its default already behaves
        # as False (no predict_proba, _roc_auc below falls back to
        # decision_function), and passing `probability=False` explicitly
        # triggers a FutureWarning under sklearn >=1.9 -- the parameter is
        # being removed in 1.11, but only the *explicit* form is deprecated.
        # Do not "fix" that warning by passing `probability=True`: it forces
        # libsvm's internal 5-fold CV calibration, which is far slower to fit
        # for no benefit this app uses (AUC only needs decision_function).
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    # Default / "k-NN"
    return KNeighborsClassifier(n_neighbors=5)


def evaluate_classifier(
    classifier_name: str, train_x, train_y, test_x, test_y
) -> tuple[EvaluationMetrics, np.ndarray]:
    """Fit *classifier_name* on the source half and score it on the target."""
    clf = build_classifier(classifier_name)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    total_samples = len(test_y)
    correct_predictions = int(np.sum(predictions == test_y))

    accuracy = float(accuracy_score(test_y, predictions))
    try:
        precision = float(precision_score(test_y, predictions, average="binary", zero_division=0))
        recall = float(recall_score(test_y, predictions, average="binary", zero_division=0))
        f1 = float(f1_score(test_y, predictions, average="binary", zero_division=0))
    except ValueError:
        precision = float(precision_score(test_y, predictions, average="macro", zero_division=0))
        recall = float(recall_score(test_y, predictions, average="macro", zero_division=0))
        f1 = float(f1_score(test_y, predictions, average="macro", zero_division=0))

    return (
        EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            wrong_predictions=total_samples - correct_predictions,
            auc=_roc_auc(clf, test_x, test_y),
        ),
        predictions,
    )


def _roc_auc(clf, test_x, test_y) -> float | None:
    """ROC AUC from whichever scoring interface the classifier offers.

    Returns ``None`` rather than raising or substituting a number: AUC is
    genuinely undefined for a single-class target, and several of the
    selectable classifiers (SVC without ``probability=True``) expose only
    a decision function, whose shape differs between the binary and
    multi-class cases.
    """
    try:
        classes = np.unique(test_y)
        if classes.size < 2:
            return None
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(test_x)
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(test_x)
        else:
            return None

        scores = np.asarray(scores)
        if classes.size == 2:
            # Binary AUC wants the positive class's score alone.
            if scores.ndim == 2:
                scores = scores[:, -1]
            return float(roc_auc_score(test_y, scores))
        if scores.ndim != 2:
            return None
        return float(
            roc_auc_score(test_y, scores, multi_class="ovr", average="macro")
        )
    except Exception:  # noqa: BLE001 - an unscoreable run reports no AUC
        return None


__all__ = ["build_classifier", "evaluate_classifier"]
