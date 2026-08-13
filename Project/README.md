# Adapta — Transfer Learning Studio

A clean-architecture **PyQt5** desktop application for running domain-adaptation
and transfer-learning experiments. Load a source and a target dataset, pick an
adaptation method, fit a classifier on the transformed source domain, and
evaluate it on the transformed target — accuracy, precision, recall, and F1,
alongside per-sample predictions.

Built as the deliverable of the SPL-3 (Software Project Lab) course.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Dataset Formats](#dataset-formats)
- [Configuration Reference](#configuration-reference)
- [Optional PostgreSQL Persistence](#optional-postgresql-persistence)
- [Extending the App](#extending-the-app)
- [Project Notes](#project-notes)
- [License](#license)

## Features

- **9 transfer-learning / domain-adaptation methods** — TCA, JDA, CORAL, BDA,
  OCMTL, DALI, GFK, JGSA, JCSL.
- **Per-method hyperparameters**, auto-enabled per algorithm — kernel type
  (Primal / Linear / RBF / SAM), regularization λ, and latent dimension.
- **Configurable preprocessing pipeline** — 10 normalization options and 6
  standardization options, applied in that order, independently to source and
  target (see [Configuration Reference](#configuration-reference)).
- **Pluggable classifier** on the transformed source — k-NN, SVM, Random
  Forest, Logistic Regression, or Decision Tree.
- **Rich result inspection** — metric cards, sample counts, and three result
  tabs (Predictions / Transformed Features / Dataset Summary), a filterable
  full-screen predictions view, and an HTML-based **PDF report export**.
- **Batch mode** — supply a list of files and the app runs every ordered
  (source, target) pair automatically, reporting per-pair and average metrics.
- **Dark / light theming** driven by a single central style sheet.
- **Optional PostgreSQL persistence** — user accounts and roles, experiment
  history, a dataset registry, and analytics charts. Entirely optional: the
  app runs fully offline without a database.

## Repository Layout

```text
Project/
├── requirements.txt
├── .env.example                # PostgreSQL connection template (optional)
├── Datasets/                   # sample & reference datasets (gitignored)
│   ├── sample_data/            #   small CSVs (source.csv, target.csv)
│   ├── mnist_usps/             #   .mat domain-adaptation datasets
│   ├── RELINK/ NASA/           #   .xlsx defect-prediction datasets
│   └── COL20/ office_caltech_surf/
└── src/
    ├── main.py                 # composition root / wiring entry point
    ├── domain/                 # models, interfaces — no UI or infra imports
    ├── application/            # use cases orchestrating domain logic
    ├── infrastructure/         # dataset loader, TL methods, persistence
    └── presentation/qt/        # Qt widgets, styling, assets
```

## Architecture

Clean architecture with strict inward-pointing dependencies: `domain` imports
nothing from the app; `application` imports only `domain`; `infrastructure`
and `presentation` may import both.

```text
presentation/qt/   MainWindow (Qt widgets) → TransferLearningWorker (QThread)
                              │
                              ▼
application/       RunTransferLearningUseCase.execute(request) → result
                              │
                              ▼
domain/            TransferLearningRequest / Result / Dataset / EvaluationMetrics
                    DatasetLoader (Protocol), TransferLearningMethod (ABC)
                              ▲
                              │ implemented by
infrastructure/    CSVDatasetLoader, TCAMethod, JDAMethod, CORALMethod,
                    BDAMethod, OCMTLMethod, DALIMethod, GFKMethod,
                    JGSAMethod, JCSLMethod
```

**Data flow through a run:**

1. `MainWindow` validates inputs and builds a frozen `TransferLearningRequest`.
2. `TransferLearningWorker` (a `QThread`) runs the use case off the UI thread
   and emits `succeeded(result)` / `failed(str)`.
3. `RunTransferLearningUseCase.execute()` loads both datasets, applies
   normalization then standardization (independently to source and target),
   runs the selected method, then fits the chosen classifier on the
   transformed source and predicts the transformed target.
4. `MainWindow` fills the metric cards, the counts caption, and the result
   tabs; the report view builds an HTML string and prints it via
   `QTextDocument` + `QPrinter`.

**Adding a method** is a single step: implement `TransferLearningMethod` in
`infrastructure/methods/` and instantiate it in `src/main.py:build_window()`.
The method dropdown is populated from `use_case.method_names`, so there is no
second list to keep in sync.

## Getting Started

**Requirements:** Python 3.10+

```bash
cd Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
source venv/bin/activate
python3 src/main.py
```

Typical workflow: pick a method, use **Browse** to choose source and target
datasets, tune the method's parameters, then click **Run Experiment**.

### CLI Options

| Flag | Effect |
| --- | --- |
| `--no-database` | Start without connecting to PostgreSQL (no history/sign-in). |
| `--self-test` | Build the UI offscreen and auto-exit after ~400 ms — a headless startup smoke test. |

```bash
# Headless startup check — exit code 0 means the window built and
# threaded correctly.
QT_QPA_PLATFORM=offscreen python3 src/main.py --self-test
```

## Dataset Formats

`CSVDatasetLoader` handles three file types:

| Format | Handling |
| --- | --- |
| `.csv` / `.xlsx` | Last column is the label; all other columns are coerced to numeric features (missing/non-numeric values → `0.0`). |
| `.mat` | Tries source-specific variable names (`X_src`, `Xs`, `fea`, …) vs. target-specific ones (`X_tar`, `Xt`, `tar_fea`, …), falling back to the largest 2-D array as features and the longest vector as labels. |

Labels are optional. When either dataset has none, the run still adapts and
transforms the features but reports **no metrics** — the metric cards and
predictions tab are disabled for that run.

## Configuration Reference

Every run is configured through dropdowns; the visible label and the value the
use case dispatches on are kept separate so wording can change without
breaking a transform.

<details>
<summary><strong>Normalization</strong></summary>

Min-Max Normalization · Z-score Standardization · Robust Standardization ·
L1 normalization · L2 (Unit Vector) Normalization · Mean Centering ·
Column-wise unit norm · Row-wise unit norm · Frobenius normalization ·
Spectral normalization

</details>

<details>
<summary><strong>Standardization</strong></summary>

Z-score Standardization · Robust Standardization · Mahalanobis Standardization ·
Variance Scaling · Column mean centering · Row mean centering

</details>

<details>
<summary><strong>Kernel type</strong></summary>

Primal (no kernel) · Linear · RBF (Gaussian) · SAM (spectral angle)

</details>

<details>
<summary><strong>Classifier</strong></summary>

k-NN (5-Nearest Neighbour) · SVM · Random Forest · Logistic Regression ·
Decision Tree

</details>

Not every method honors every parameter — the UI greys out controls a method
would ignore, driven by `use_case.supported_parameters(name)`.

| Method | Kernel | Regularization | Latent dim |
| --- | :---: | :---: | :---: |
| TCA | ✓ | ✓ | ✓ |
| JDA | ✓ | ✓ | ✓ |
| CORAL | | | |
| BDA | ✓ | ✓ | ✓ |
| OCMTL | | | ✓ |
| DALI | | ✓ | ✓ |
| GFK | | | ✓ |
| JGSA | | ✓ | ✓ |
| JCSL | | ✓ | ✓ |

## Optional PostgreSQL Persistence

Enabling a database unlocks user accounts (teacher / researcher / student /
admin roles), sign-in, a per-user experiment history, a dataset registry, and
analytics charts.

```bash
cp .env.example .env
# then edit .env and set DATABASE_URL (a free Neon project works well)
```

The app starts fine without it — omit `.env` or pass `--no-database` and you
simply lose persistence and sign-in; the core adaptation workflow is
unaffected.

## Extending the App

**Add a transfer-learning method**

1. Implement `TransferLearningMethod` under `src/infrastructure/methods/`.
2. Instantiate it once in `src/main.py:build_window()`.

That's the whole job — the dropdown and capability matrix update themselves.
A method opts into a UI-tunable parameter by naming its private attribute
`_kernel_type`, `_regularization`, or `_latent_dim`; the use case duck-types
matching request fields onto the instance at run time (see
`TUNABLE_PARAMETERS` in `run_transfer_learning_use_case.py`).

**Add a normalization / standardization option**

Add the `(label, value)` pair to `NORMALIZATION_OPTIONS` /
`STANDARDIZATION_OPTIONS` in `presentation/qt/main_window_ui.py`, then add the
matching branch in `apply_normalization` / `apply_standardization` in the use
case.

## Project Notes

- No automated test suite, linter, or formatter is configured.
  `python3 src/main.py --self-test` is the de facto smoke test.
- `verify_tca.py` / `verify_jda.py` are ad-hoc verification scripts at the
  `Project/` root; run them from `Project/` since they use relative dataset
  paths.
- Sample datasets live in `Datasets/` and are gitignored — populate that
  directory locally before running experiments.

## License

Academic project — provided for educational purposes.
