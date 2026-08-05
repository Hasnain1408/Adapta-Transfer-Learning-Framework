# Transfer Learning Desktop App

A clean-architecture **PyQt5** desktop application for running domain-adaptation /
transfer-learning experiments. It loads a source and a target dataset, applies a
chosen adaptation method, fits a classifier on the transformed source domain, and
evaluates it on the transformed target, reporting accuracy, precision, recall and
F1 alongside per-sample predictions.

Built as the deliverable of the SPL-3 (Software Project Lab) course.

## Features

- **9 transfer-learning / domain-adaptation methods**:
  TCA, JDA, CORAL, BDA, OCMTL, DALI, GFK, JGSA, JCSL.
- **Per-method hyperparameters** auto-enabled per algorithm:
  kernel (primal / linear / RBF / SAM), regularization λ, and latent dimension.
- **Preprocessing pipeline** — normalization followed by standardization,
  applied independently to source and target.
- **Plug-in classifier** on the transformed source: k-NN, SVM, Random Forest,
  Logistic Regression, Decision Tree.
- **Rich result inspection** — metric cards, counts, and three tabs
  (Predictions / Transformed Features / Dataset Summary), a filterable
  full-screen predictions view, and an HTML **PDF report export**.
- **Batch mode** — run every ordered (source, target) pair across a file list.
- **Dark / light theming** with a single central style sheet.
- **Optional PostgreSQL persistence** — user accounts, an experiment history,
  a dataset registry, and analytics charts. Completely optional: the app runs
  fully offline without a database.

## Repository layout

```text
Project/
  requirements.txt
  .env.example          # PostgreSQL connection (optional)
  Datasets/             # sample & reference datasets (gitignored)
    sample_data/        #   small CSVs (source.csv, target.csv)
    mnist_usps/         #   .mat domain-adaptation datasets
    RELINK/ NASA/       #   .xlsx defect-prediction datasets
    COL20/ office_caltech_surf/
  src/
    main.py             # wiring entry point
    domain/             #   models, interfaces (no UI imports)
    application/        #   orchestration use cases
    infrastructure/     #   data loader, persistence, method implementations
    presentation/qt/    #   Qt widgets, styling, assets
requirements.txt
```

Clean architecture with strict inward-pointing dependencies: `domain` imports
nothing from the app; `application` imports `domain`; `infrastructure` and
`presentation` may import both.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To enable history/accounts, copy `.env.example` to `.env` and set a real
PostgreSQL `DATABASE_URL` (a free Neon project works). The app also starts fine
without it — you just lose persistence and sign-in.

## Run

```bash
source venv/bin/activate
python3 src/main.py
```

Typical run: pick a method, click the **Browse** buttons to choose source and
target datasets, tune the method parameters, and press **Run Experiment**.

### Headless startup check

Launches the UI offscreen and auto-exits after ~400 ms. Exit code 0 means the
window built and threaded correctly.

```bash
QT_QPA_PLATFORM=offscreen python3 src/main.py --self-test
```

### CLI options

```bash
python3 src/main.py --no-database   # start without connecting to PostgreSQL
python3 src/main.py --self-test     # build UI offscreen and auto-exit
```

## Dataset formats

The loader (`CSVDatasetLoader`) handles three file types:

- **`.csv` / `.xlsx`** — the last column is the label; all other columns are
  coerced to numeric features (missing values → 0).
- **`.mat`** — tries common variable names (`X_src`, `Xs`, `fea`, … for source;
  `X_tar`, `Xt`, `tar_fea`, … for target), otherwise selects the largest 2-D
  array as features and the longest vector as labels.

Labels are optional. When either dataset has none, the run adapts and transforms
but reports **no metrics** (the metric cards and prediction tab are then
disabled).

## Continuing a run in the code

The entire pipeline lives in
`src/application/run_transfer_learning_use_case.py:execute()`, which:

1. loads both datasets,
2. normalises → standardises each (independently),
3. runs the selected method,
4. fits the classifier on transformed source and predicts the transformed target,
5. builds `EvaluationMetrics`.

### Adding a transfer-learning method

Implement `TransferLearningMethod` under `src/infrastructure/methods/` and
instantiate it once in `src/main.py:build_window()`. That is the whole job —
the method dropdown populates from `use_case.method_names`, so there is no
second list to keep in sync.

A method opts into a UI-tunable parameter simply by naming its attribute
`_kernel_type` / `_regularization` / `_latent_dim`; the use case duck-types the
values onto the instance (see `TUNABLE_PARAMETERS`), and the UI greys out a
control the method would ignore.

## Notes

- No automated test suite is configured. `src/main.py --self-test` is the de
  facto smoke test.
- `requirements.txt` includes `scipy` alongside `scikit-learn` for `.mat`
  loading and `scipy.linalg` (OCMTL).

## License

Academic project — provided for educational purposes.