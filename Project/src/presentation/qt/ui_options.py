"""The option tables backing the configuration dropdowns.

Each entry is (display label, value) and the combo boxes send
currentData() — the value, never the visible label — so a dropdown can
be reworded without silently disabling the transform behind it. The values
are what application.preprocessing dispatches on, so adding an option
here means adding the matching branch there.
"""

from __future__ import annotations

#: (display label, value). The value is what the use case dispatches on, so
#: labels can be reworded without silently disabling a transform.
NORMALIZATION_OPTIONS: list[tuple[str, str]] = [
    ("None", "None"),
    ("Min-Max Normalization", "Min-Max Normalization"),
    ("Z-score Standardization", "Z-score Standardization"),
    ("Robust Standardization", "Robust Standardization"),
    ("L1 normalization", "L1 normalization"),
    ("L2 (Unit Vector) Normalization", "L2 (Unit Vector) Normalization"),
    ("Mean Centering", "Mean Centering"),
    ("Column-wise unit norm", "Column-wise unit norm"),
    ("Row-wise unit norm", "Row-wise unit norm"),
    ("Frobenius normalization", "Frobenius normalization"),
    ("Spectral normalization", "Spectral normalization"),
]

STANDARDIZATION_OPTIONS: list[tuple[str, str]] = [
    ("None", "None"),
    ("Z-score Standardization", "Z-score Standardization"),
    ("Robust Standardization", "Robust Standardization"),
    ("Mahalanobis Standardization", "Mahalanobis Standardization"),
    ("Variance Scaling", "Variance Scaling"),
    ("Column mean centering", "Column mean centering"),
    ("Row mean centering", "Row mean centering"),
]

KERNEL_OPTIONS: list[tuple[str, str]] = [
    ("Primal (no kernel)", "primal"),
    ("Linear", "linear"),
    ("RBF (Gaussian)", "rbf"),
    ("SAM (spectral angle)", "sam"),
]

CLASSIFIER_OPTIONS: list[tuple[str, str]] = [
    ("k-NN (5-Nearest Neighbour)", "k-NN"),
    ("SVM", "SVM"),
    ("Random Forest", "Random Forest"),
    ("Logistic Regression", "Logistic Regression"),
    ("Decision Tree", "Decision Tree"),
]

DATASET_FILTER = (
    "Dataset Files (*.csv *.xlsx *.mat *.arff);;"
    "CSV Files (*.csv);;Excel Files (*.xlsx);;MATLAB Files (*.mat);;ARFF Files (*.arff)"
)

#: Rows rendered in the transformed-features preview. Wide matrices make full
#: rendering slow, so the cap is surfaced in the caption rather than hidden.
FEATURE_PREVIEW_ROWS = 1000


__all__ = [
    "CLASSIFIER_OPTIONS",
    "DATASET_FILTER",
    "FEATURE_PREVIEW_ROWS",
    "KERNEL_OPTIONS",
    "NORMALIZATION_OPTIONS",
    "STANDARDIZATION_OPTIONS",
]
