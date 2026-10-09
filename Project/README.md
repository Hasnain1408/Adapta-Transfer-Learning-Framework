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
- [Execution-ordered source reference](#execution-ordered-source-reference)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Dataset Formats](#dataset-formats)
- [Configuration Reference](#configuration-reference)
- [Optional PostgreSQL Persistence](#optional-postgresql-persistence)
- [Extending the App](#extending-the-app)
- [Project Notes](#project-notes)
- [License](#license)

## Features

- **11 transfer-learning / domain-adaptation methods** — TCA, JDA, CORAL, BDA,
  OCMTL, DALI, GFK, JGSA, JCSL, MEDA, and Proposed.
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

### Build and launch the desktop app

On Linux, create the self-contained executable with:

```bash
./build_desktop.sh
```

The executable is written to `dist/Adapta`. The build also copies it to
`~/Desktop/Adapta` and creates a `~/Desktop/Adapta.desktop` launcher, so it can
be started from the desktop. The launcher uses the supplied Adapta icon and
starts with database support enabled. Start it manually with:

```bash
./dist/Adapta --no-database
```

The build installs the desktop launcher automatically. To install or refresh
it separately after moving the project, run:

```bash
./install_desktop_entry.sh
```

The installer copies the executable and records its current desktop path, so
run it again if the app is rebuilt or moved. The build uses
[Adapta.spec](Adapta.spec), which bundles the Qt
SVG assets and keeps the application console-free for normal desktop use.

For the Desktop copy, put a `.env` file containing `DATABASE_URL` beside the
executable at `~/Desktop/.env`. The app also accepts `DATABASE_URL` as an
environment variable. Without database configuration, it starts offline and
reports that persistence is unavailable.

If the file manager says it cannot find a default application for
`application/x-executable`, it is trying to open the binary as a document.
Right-click `dist/Adapta`, choose **Properties**, enable **Allow executing file
as program**, then choose **Run**. Starting it from a terminal also works:

```bash
./dist/Adapta --no-database
```

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

The composite `DatasetLoader` supports four file types:

| Format | Handling |
| --- | --- |
| `.csv`, `.xlsx`, `.arff` | Last column is the label; all other columns are coerced to numeric features (missing/non-numeric values → `0.0`). |
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
| MEDA | ✓ | ✓ | ✓ |
| Proposed | | ✓ | |

## Execution-ordered source reference

This section is the file-level guide to `src/`. Files are listed in the order
in which a normal application run reaches them. Methods are named so a reader
can jump from the UI event to the corresponding implementation without
searching the whole repository. Private helpers are included when they contain
meaningful application behavior; trivial Qt constructors and package
`__init__.py` files are noted separately.

### 1. Application entry point and dependency wiring

| File | Responsibility and important methods |
| --- | --- |
| [`src/main.py`](src/main.py) | Composition root. `build_catalog()` resolves PostgreSQL and creates repositories, `build_window()` registers all dataset loaders and the 11 adaptation methods, and `main()` parses `--self-test`/`--no-database`, creates `QApplication`, applies the stylesheet, shows `MainWindow`, and closes the database on exit. |
| [`src/domain/interfaces.py`](src/domain/interfaces.py) | Core contracts: `DatasetLoader.load()` and abstract `TransferLearningMethod.name`/`run()`. |
| [`src/domain/models.py`](src/domain/models.py) | Immutable run payloads: `TransferLearningRequest`, `BatchTransferLearningRequest`, `MultiMethodTransferLearningRequest`, `Dataset`, `EvaluationMetrics`, `DomainAlignment`, `NegativeTransferAnalysis`, and result containers. `MethodTransferLearningResult.metrics`, `MultiMethodTransferLearningResult.evaluated()`, `method_ranks()`, `ranked()`, and `best` derive comparison views. |
| [`src/domain/progress.py`](src/domain/progress.py) | Defines pipeline stage constants and immutable `ProgressUpdate`. `fraction` converts stage/pair progress to a normalized value and `headline` formats status text. |
| [`src/domain/run_control.py`](src/domain/run_control.py) | Thread-safe pause/resume/cancel and progress channel. `pause()`, `resume()`, `cancel()`, `checkpoint()`, `set_listener()`, `set_pair()`, `report()`, and `report_iteration()` coordinate workers with the UI; `paused_seconds`, `elapsed_seconds`, and `last_update` expose state. |

### 2. Window construction, navigation, and run selection

`MainWindow` is assembled through mixins, so the UI behavior is distributed
across focused files rather than implemented in one large widget.

| File | Responsibility and important methods |
| --- | --- |
| [`src/presentation/qt/main_window.py`](src/presentation/qt/main_window.py) | Top-level `MainWindow`. `__init__()` composes all UI/run mixins; `_show_landing()`, `_show_workspace()`, `_navigate_from_landing()`, `_open_workspace_tab()`, `_open_history_from_landing()`, `changeEvent()`, `_stop_background_work()`, `closeEvent()`, and `_exit_application()` control navigation and shutdown. |
| [`src/presentation/qt/main_window_ui.py`](src/presentation/qt/main_window_ui.py) | `MainWindowUIMixin` aggregates `MainWindowWidgetsMixin`, `MainWindowMenuMixin`, `MainWindowLayoutMixin`, and `MainWindowSignalsMixin`. |
| [`src/presentation/qt/main_window_widgets.py`](src/presentation/qt/main_window_widgets.py) | Creates shared controls and tables: `_build_widgets()`, `_build_option_combo()`, `_align_headers()`, `_build_table()`, `_field_label()`, `_build_fullscreen_button()`. |
| [`src/presentation/qt/main_window_menu.py`](src/presentation/qt/main_window_menu.py) | `MainWindowMenuMixin._build_menu()` and `_build_title_bar()` create menus and the custom title bar. |
| [`src/presentation/qt/main_window_layout.py`](src/presentation/qt/main_window_layout.py) | Builds the landing page and experiment workspace: `_build_layout()`, `_build_header()`, `_build_sidebar()`, `_build_content()`, dataset/parameter/method/action/metric/result/status builders, and result-tab builders. |
| [`src/presentation/qt/main_window_signals.py`](src/presentation/qt/main_window_signals.py) | `MainWindowSignalsMixin._connect_signals()` connects buttons, combos, tabs, checkboxes, and worker signals to handlers. |
| [`src/presentation/qt/landing_page.py`](src/presentation/qt/landing_page.py) | `LandingPage` presents the feature cards and emits navigation/session actions. |
| [`src/presentation/qt/main_window_run.py`](src/presentation/qt/main_window_run.py) | `MainWindowRunMixin` composes busy, dataset, progress, execution, and workspace mixins. |
| [`src/presentation/qt/main_window_busy.py`](src/presentation/qt/main_window_busy.py) | Prevents conflicting actions while a worker is active with `_is_worker_busy()`, `_busy_description()`, and `_reject_if_busy()`. |
| [`src/presentation/qt/main_window_datasets.py`](src/presentation/qt/main_window_datasets.py) | File selection and batch management: `_pick_source()`, `_pick_target()`, `_swap_datasets()`, `_pick_file()`, `_on_combined_file_toggled()`, collapse/summary helpers, `_on_batch_mode_toggled()`, `_add_batch_files()`, `_remove_selected_batch_files()`, `_clear_batch_files()`, and `_update_batch_caption()`. |
| [`src/presentation/qt/main_window_execution.py`](src/presentation/qt/main_window_execution.py) | Validates controls and dispatches runs. `_run_transfer_learning()` chooses single/batch/multi-method mode; `_run_single_experiment()`, `_run_batch_experiment()`, and `_run_multi_method_experiment()` build requests and workers; `_validated_single_pair()`, `selected_method_names()`, `_sync_method_parameters()`, and result/error handlers complete the flow. |
| [`src/presentation/qt/main_window_progress.py`](src/presentation/qt/main_window_progress.py) | Maps `ProgressUpdate` objects to the status strip. `_on_progress()`, `_set_status()`, `_set_running()`, `_toggle_pause()`, `_cancel_run()`, and `_on_cancelled()` implement progress, pause, and cancellation UX. |
| [`src/presentation/qt/main_window_workspace.py`](src/presentation/qt/main_window_workspace.py) | Creates, replaces, and clears workspace tabs and keeps the active run mode consistent. |
| [`src/presentation/qt/main_window_session.py`](src/presentation/qt/main_window_session.py) | Stores current authenticated user/session state and updates session labels/actions. |
| [`src/presentation/qt/main_window_auth.py`](src/presentation/qt/main_window_auth.py) | Opens authentication dialogs and applies login/logout state to the main window. |
| [`src/presentation/qt/main_window_database.py`](src/presentation/qt/main_window_database.py) | Displays database connection information/errors and opens database-related dialogs. |
| [`src/presentation/qt/main_window_history.py`](src/presentation/qt/main_window_history.py) | Loads catalog history, refreshes the history tab, opens detail views, and persists successful/failed runs. |

### 3. Background execution boundary

| File | Responsibility and important methods |
| --- | --- |
| [`src/presentation/qt/worker.py`](src/presentation/qt/worker.py) | Keeps heavy work off the Qt event loop. `_ProgressReportingWorker.run()` installs `RunControl` and emits success/error/progress/cancelled signals. `TransferLearningWorker`, `BatchTransferLearningWorker`, and `MultiMethodTransferLearningWorker` call `execute()`, `execute_batch()`, and `execute_multi_method()` respectively. |
| [`src/presentation/qt/processing_status.py`](src/presentation/qt/processing_status.py) | `ProcessingStatusStrip` renders stage text, pair/method counters, progress, elapsed time, pause, and cancel controls. |
| [`src/presentation/qt/duration.py`](src/presentation/qt/duration.py) | Small duration-formatting helpers used by status bars and result summaries. |

### 4. Application use cases and the transfer-learning pipeline

| File | Responsibility and important methods |
| --- | --- |
| [`src/application/run_transfer_learning_use_case.py`](src/application/run_transfer_learning_use_case.py) | `RunTransferLearningUseCase` stores the loader/method registry, exposes `method_names`, checks combined-domain files with `dataset_has_combined_domains()`, and reports method capabilities with `supported_parameters()`. It inherits the three execution mixins below. |
| [`src/application/run_single_execution.py`](src/application/run_single_execution.py) | `RunSingleExecutionMixin.execute()` validates the method, loads both files, limits BLAS threads, preprocesses domains, measures an unadapted baseline, then calls `adapt_and_evaluate()`. |
| [`src/application/run_batch_execution.py`](src/application/run_batch_execution.py) | `RunBatchExecutionMixin.execute_batch()` runs every ordered permutation of at least two files, records per-pair errors without losing the batch, supports cancellation, and computes average metrics. |
| [`src/application/run_multi_method_execution.py`](src/application/run_multi_method_execution.py) | `RunMultiMethodExecutionMixin.execute_multi_method()` loads/preprocesses once, measures one shared baseline, runs selected methods sequentially, preserves individual errors, supports cancellation, and returns ranked comparison data. `_method_request()` creates each method-specific request. |
| [`src/application/run_pipeline.py`](src/application/run_pipeline.py) | Shared stages. `load_domains()` loads and validates source/target; `preprocess_domains()` applies normalization then standardization independently; `measure_baseline()` evaluates without adaptation; `adapt_and_evaluate()` injects tunables, calls the selected method, classifies transformed features, computes negative transfer, measures alignment, and assembles `TransferLearningResult`. `PreparedDomains.is_labelled` detects whether both domains have labels. |
| [`src/application/preprocessing.py`](src/application/preprocessing.py) | `apply_normalization()` implements min-max, z-score, robust, L1/L2, centering, column/row, Frobenius, and spectral options. `apply_standardization()` implements z-score, robust, variance, column/row centering, and Mahalanobis whitening. |
| [`src/application/classification.py`](src/application/classification.py) | `build_classifier()` maps UI names to scikit-learn classifiers. `evaluate_classifier()` trains on transformed source data, predicts target data, computes accuracy/precision/recall/F1 and predictions, and `_roc_auc()` adds binary or multiclass AUC where supported. |
| [`src/application/domain_alignment.py`](src/application/domain_alignment.py) | `measure_domain_alignment()` computes before/after MMD and overlap, converts MMD change to a status, and returns `DomainAlignment` while keeping alignment failures non-fatal to the experiment. |
| [`src/application/run_tunables.py`](src/application/run_tunables.py) | `TUNABLE_PARAMETERS` maps request fields to optional private method attributes. `apply_tunable_parameters()` performs duck-typed injection; `supported_parameters_for()` drives the UI capability state. |

### 5. Dataset loading adapters

| File | Responsibility and important methods |
| --- | --- |
| [`src/infrastructure/data/dataset_loader.py`](src/infrastructure/data/dataset_loader.py) | `DatasetFormatLoader` defines the format adapter and `_dataframe_to_dataset()` converts tabular data (last column label, numeric coercion, missing-value tracking). Composite `DatasetLoader.__init__()`, `load()`, `has_combined_domains()`, and `_loader_for()` dispatch by extension. |
| [`src/infrastructure/data/csv_dataset_loader.py`](src/infrastructure/data/csv_dataset_loader.py) | `CSVDatasetLoader.load()` reads CSV with pandas and uses the shared tabular conversion. |
| [`src/infrastructure/data/xlsx_dataset_loader.py`](src/infrastructure/data/xlsx_dataset_loader.py) | `XLSXDatasetLoader.load()` reads Excel through openpyxl and uses the shared conversion. |
| [`src/infrastructure/data/arff_dataset_loader.py`](src/infrastructure/data/arff_dataset_loader.py) | `ARFFDatasetLoader.load()` reads Weka ARFF, decodes byte-valued nominal columns, and uses the shared conversion. |
| [`src/infrastructure/data/mat_dataset_loader.py`](src/infrastructure/data/mat_dataset_loader.py) | `MATDatasetLoader.has_combined_domains()` detects source/target keys. `load()` selects known or fallback feature/label arrays, fixes orientation, and returns a `Dataset`. |

### 6. Transfer-learning method implementations

Every method implements `TransferLearningMethod.name` and `run(source_features,
target_features, source_labels, control)`. Iterative methods call
`RunControl.checkpoint()` and `report_iteration()` so pause/cancel/progress work
the same way for all algorithms.

| File | Algorithm and internal methods |
| --- | --- |
| [`src/infrastructure/methods/tca_method.py`](src/infrastructure/methods/tca_method.py) | `TCAMethod` performs kernel/primal TCA. `_kernel()` builds linear/RBF/SAM kernels and `run()` solves the MMD/variance generalized eigenproblem. |
| [`src/infrastructure/methods/jda_method.py`](src/infrastructure/methods/jda_method.py) | `JDAMethod` alternates marginal/conditional MMD alignment, eigenspace projection, and 1-NN target pseudo-label updates. `_kernel()` and `run()` are the core methods. |
| [`src/infrastructure/methods/coral_method.py`](src/infrastructure/methods/coral_method.py) | `CORALMethod.run()` whitens source covariance and re-colors it with target covariance; it has no tunable UI parameters. |
| [`src/infrastructure/methods/bda_method.py`](src/infrastructure/methods/bda_method.py) | `BDAMethod` implements BDA/W-BDA. `_kernel()` builds kernels and `run()` balances marginal/conditional distributions, class weights, eigenspace learning, and pseudo-label updates. |
| [`src/infrastructure/methods/ocmtl_method.py`](src/infrastructure/methods/ocmtl_method.py) | `OCMTLMethod.run()` computes class balancing weights and alternates SVD-based source/target orthogonal core extraction, stopping on subspace-angle regression. |
| [`src/infrastructure/methods/dali_method.py`](src/infrastructure/methods/dali_method.py) | `DALIMethod` combines distribution alignment, locality preserving projections, and target discriminative scatter. Helpers `_l2_normalize_rows()`, `_within_scatter()`, `_between_scatter()`, `_calculate_lpp()`, `_marginal_distribution()`, and `_conditional_distribution()` build the objective; `run()` iterates the coupled eigensystem. |
| [`src/infrastructure/methods/gfk_method.py`](src/infrastructure/methods/gfk_method.py) | `GFKMethod._full_pca_basis()` and `_gfk_kernel()` construct the Grassmann geodesic flow kernel; `run()` projects both domains through its square root. |
| [`src/infrastructure/methods/jgsa_method.py`](src/infrastructure/methods/jgsa_method.py) | `JGSAMethod` jointly aligns distributions and geometry while preserving source discrimination. Scatter, distribution, normalization, and `run()` helpers support coupled source/target projections and pseudo-label refinement. |
| [`src/infrastructure/methods/jcsl_method.py`](src/infrastructure/methods/jcsl_method.py) | `JCSLMethod._pca_top()` finds the target subspace; `run()` optimizes source hinge classification, weight regularization, and source/target subspace alignment with minibatch SGD. |
| [`src/infrastructure/methods/meda_method.py`](src/infrastructure/methods/meda_method.py) | `MEDAMethod._kernel()`, `_proxy_a_distance()`, and `_estimate_mu()` support GFK embedding, dynamic marginal/conditional balance, and kernel ridge adaptation in `run()`. |
| [`src/infrastructure/methods/proposed_method.py`](src/infrastructure/methods/proposed_method.py) | `ProposedMethod` combines covariance alignment and OCMTL-style extraction. `_l2_normalise_rows()`, `_l2_normalise_cols()`, `_pseudo_labels()`, `_phase1()`, `_phase2()`, `_conditional_matrix()`, and `run()` implement rank selection, alternating SVD, pseudo-label refinement, and final projection. |

### 7. Domain analytics and comparison helpers

| File | Responsibility and important methods |
| --- | --- |
| [`src/domain/mmd.py`](src/domain/mmd.py) | MMD measurement. `_subsample()`, `_squared_distances()`, `_median_bandwidth()`, `compute_mmd()`, `percentage_change()`, and `classify_alignment()` calculate deterministic RBF/linear divergence and user-facing alignment status. |
| [`src/domain/overlap.py`](src/domain/overlap.py) | Nearest-neighbour mixing estimate. `_subsample()`, `_as_2d()`, `compute_overlap()`, and `overlap_change()` quantify source/target intermixing. |
| [`src/domain/ranking.py`](src/domain/ranking.py) | `competition_ranks()` assigns tie-aware metric ranks; `average_metric_ranks()` averages ranks across accuracy, precision, recall, F1, and AUC. |

### 8. Authentication, catalogue, and PostgreSQL persistence

Persistence is optional. When configured, startup creates/migrates the schema,
then the UI uses the application catalogue facade rather than issuing SQL.

| File | Responsibility and important methods |
| --- | --- |
| [`src/domain/auth.py`](src/domain/auth.py) | `hash_password()` creates salted PBKDF2-HMAC-SHA256 hashes; `verify_password()` validates them with constant-time comparison; `_derive()` performs key derivation. |
| [`src/domain/entities.py`](src/domain/entities.py) | Durable records: `User`, `DatasetRecord`, `ExperimentRecord`, `EvaluationRecord`, and `ExperimentHistoryEntry`. `UserRole`/`ExperimentStatus`/`DatasetFormat` provide parsing and display helpers; evaluation properties calculate MMD/overlap and baseline deltas. |
| [`src/domain/repositories.py`](src/domain/repositories.py) | Protocols for user, dataset, experiment, and evaluation repositories. |
| [`src/application/authentication_use_case.py`](src/application/authentication_use_case.py) | `AuthenticationUseCase.register()` validates and hashes new accounts; `login()` verifies credentials. `AuthenticationError` is the presentation-facing validation exception. |
| [`src/application/experiment_catalog_use_case.py`](src/application/experiment_catalog_use_case.py) | `ExperimentCatalogUseCase` facade combines the four catalogue mixins and receives repository dependencies. |
| [`src/application/catalog_users.py`](src/application/catalog_users.py) | `CatalogUsersMixin` implements create/update/delete/get/list user operations and `ensure_default_user()`. |
| [`src/application/catalog_datasets.py`](src/application/catalog_datasets.py) | `CatalogDatasetsMixin.register_dataset()` derives metadata from a loaded `Dataset`; list/get/delete expose the registry. |
| [`src/application/catalog_experiments.py`](src/application/catalog_experiments.py) | `CatalogExperimentsMixin.record_success()` persists datasets, experiment configuration, evaluation, alignment, and baseline data; `record_failure()` stores errors; list/history/get/delete methods read or remove runs. |
| [`src/application/catalog_queries.py`](src/application/catalog_queries.py) | `CatalogQueriesMixin.domain_pairs()`, `results_for_pair()`, and `statistics()` support comparison and dashboard views. |
| [`src/application/catalog_summaries.py`](src/application/catalog_summaries.py) | Pure `class_names()`, `dtype_summary()`, and defensive `safe_count()` helpers. |
| [`src/infrastructure/persistence/config.py`](src/infrastructure/persistence/config.py) | Resolves PostgreSQL DSNs from explicit input, `DATABASE_URL`, `.env`, or PG variables. `load_env_file()`, `resolve_dsn()`, `_ensure_sslmode()`, and `describe_dsn()` handle parsing, TLS defaults, and credential masking. |
| [`src/infrastructure/persistence/driver.py`](src/infrastructure/persistence/driver.py) | Optional psycopg2 import and `DatabaseUnavailable` error, allowing offline startup when the driver/database is unavailable. |
| [`src/infrastructure/persistence/schema.py`](src/infrastructure/persistence/schema.py) | `SCHEMA_STATEMENTS` creates/migrates users, datasets, experiments, evaluations, constraints, and indexes idempotently. |
| [`src/infrastructure/persistence/database.py`](src/infrastructure/persistence/database.py) | `Database` manages connection/reconnect/retry, schema deployment, transactions, parameterized fetch/execute helpers, table inspection, drop, close, and context-manager lifecycle. |
| [`src/infrastructure/persistence/postgres_mappers.py`](src/infrastructure/persistence/postgres_mappers.py) | `_to_user()`, `_to_dataset()`, `_to_experiment()`, `_to_evaluation()`, `_as_list()`, and `_is_unique_violation()` translate database rows to entities and normalize driver errors. |
| [`src/infrastructure/persistence/postgres_user_repository.py`](src/infrastructure/persistence/postgres_user_repository.py) | `PostgresUserRepository` provides add/update/delete/get/get-by-email/list/count for `users`. |
| [`src/infrastructure/persistence/postgres_dataset_repository.py`](src/infrastructure/persistence/postgres_dataset_repository.py) | `PostgresDatasetRepository` provides add/upsert/update/delete/get/get-by-path/list/count for registered datasets. |
| [`src/infrastructure/persistence/postgres_experiment_repository.py`](src/infrastructure/persistence/postgres_experiment_repository.py) | `PostgresExperimentRepository` provides add/update/status/delete/get/list/history/count, including one joined history query. |
| [`src/infrastructure/persistence/postgres_evaluation_repository.py`](src/infrastructure/persistence/postgres_evaluation_repository.py) | `PostgresEvaluationRepository` provides idempotent add, get, get-for-experiment, delete, list, and count for evaluations. |

### 9. Result views, charts, reports, and dialogs

These files are reached after a worker emits a result or when the user opens a
secondary feature from the landing page/history/comparison workspace.

| Files | Responsibility |
| --- | --- |
| [`src/presentation/qt/main_window_results.py`](src/presentation/qt/main_window_results.py), [`src/presentation/qt/result_tables.py`](src/presentation/qt/result_tables.py), [`src/presentation/qt/metric_card.py`](src/presentation/qt/metric_card.py), [`src/presentation/qt/dataset_summary_view.py`](src/presentation/qt/dataset_summary_view.py) | Render metric cards, predictions, transformed feature tables, dataset metadata, filters, and fullscreen result views. |
| [`src/presentation/qt/visualization_tab.py`](src/presentation/qt/visualization_tab.py), [`src/presentation/qt/visualization_analysis.py`](src/presentation/qt/visualization_analysis.py), [`src/presentation/qt/visualization_projection.py`](src/presentation/qt/visualization_projection.py), [`src/presentation/qt/visualization_plots.py`](src/presentation/qt/visualization_plots.py) | Build the visualization tab, projection choices, distribution/alignment analysis, and matplotlib plots. |
| [`src/presentation/qt/charts_widget.py`](src/presentation/qt/charts_widget.py), [`src/presentation/qt/charts_plots.py`](src/presentation/qt/charts_plots.py), [`src/presentation/qt/comparison_charts.py`](src/presentation/qt/comparison_charts.py) | Display single-run analytics and multi-method comparison charts. |
| [`src/presentation/qt/history_tab.py`](src/presentation/qt/history_tab.py), [`src/presentation/qt/experiment_detail_view.py`](src/presentation/qt/experiment_detail_view.py), [`src/presentation/qt/pair_detail_view.py`](src/presentation/qt/pair_detail_view.py) | Present persisted history, one experiment’s details, and batch/comparison pair details. |
| [`src/presentation/qt/comparison_tab.py`](src/presentation/qt/comparison_tab.py), [`src/presentation/qt/comparison_model.py`](src/presentation/qt/comparison_model.py), [`src/presentation/qt/comparison_export_dialog.py`](src/presentation/qt/comparison_export_dialog.py), [`src/presentation/qt/comparison_export.py`](src/presentation/qt/comparison_export.py) | Load pair results, rank methods, configure export scope, and export comparison tables/charts. |
| [`src/presentation/qt/main_window_report.py`](src/presentation/qt/main_window_report.py), [`src/presentation/qt/report.py`](src/presentation/qt/report.py), [`src/presentation/qt/report_single.py`](src/presentation/qt/report_single.py), [`src/presentation/qt/report_batch.py`](src/presentation/qt/report_batch.py), [`src/presentation/qt/report_multi_method.py`](src/presentation/qt/report_multi_method.py), [`src/presentation/qt/report_history.py`](src/presentation/qt/report_history.py), [`src/presentation/qt/report_comparison.py`](src/presentation/qt/report_comparison.py), [`src/presentation/qt/report_common.py`](src/presentation/qt/report_common.py), [`src/presentation/qt/export_options_dialog.py`](src/presentation/qt/export_options_dialog.py) | Select report type, compose HTML for single/batch/multi-method/history/comparison data, share common formatting, and print through Qt PDF support. |
| [`src/presentation/qt/auth_dialogs.py`](src/presentation/qt/auth_dialogs.py), [`src/presentation/qt/auth_pages.py`](src/presentation/qt/auth_pages.py), [`src/presentation/qt/user_dialogs.py`](src/presentation/qt/user_dialogs.py), [`src/presentation/qt/database_dialogs.py`](src/presentation/qt/database_dialogs.py), [`src/presentation/qt/dataset_registry_dialog.py`](src/presentation/qt/dataset_registry_dialog.py) | Login/register pages and dialogs, user administration, database status, and dataset registry CRUD. |
| [`src/presentation/qt/dialog_utils.py`](src/presentation/qt/dialog_utils.py), [`src/presentation/qt/message_box.py`](src/presentation/qt/message_box.py), [`src/presentation/qt/safe_toolbar.py`](src/presentation/qt/safe_toolbar.py), [`src/presentation/qt/title_bar.py`](src/presentation/qt/title_bar.py), [`src/presentation/qt/ui_widgets.py`](src/presentation/qt/ui_widgets.py) | Shared safe dialog, message, toolbar, title-bar, and widget construction helpers. |

### 10. Learning materials and visual language

| Files | Responsibility |
| --- | --- |
| [`src/presentation/qt/learning_materials_dialog.py`](src/presentation/qt/learning_materials_dialog.py), [`src/presentation/qt/learning_materials_content.py`](src/presentation/qt/learning_materials_content.py), [`src/presentation/qt/learning_materials_diagrams.py`](src/presentation/qt/learning_materials_diagrams.py) | Open the learning-materials browser, provide shared article/HTML content helpers, and draw explanatory diagrams. |
| [`src/presentation/qt/learning_materials_tca.py`](src/presentation/qt/learning_materials_tca.py), [`src/presentation/qt/learning_materials_jda.py`](src/presentation/qt/learning_materials_jda.py), [`src/presentation/qt/learning_materials_bda.py`](src/presentation/qt/learning_materials_bda.py), [`src/presentation/qt/learning_materials_coral.py`](src/presentation/qt/learning_materials_coral.py), [`src/presentation/qt/learning_materials_ocmtl.py`](src/presentation/qt/learning_materials_ocmtl.py), [`src/presentation/qt/learning_materials_dali.py`](src/presentation/qt/learning_materials_dali.py), [`src/presentation/qt/learning_materials_gfk.py`](src/presentation/qt/learning_materials_gfk.py), [`src/presentation/qt/learning_materials_meda.py`](src/presentation/qt/learning_materials_meda.py) | Algorithm-specific educational pages and diagrams. |
| [`src/presentation/qt/learning_materials_jcsl.py`](src/presentation/qt/learning_materials_jcsl.py), [`src/presentation/qt/learning_materials_normalization.py`](src/presentation/qt/learning_materials_normalization.py), [`src/presentation/qt/learning_materials_knn.py`](src/presentation/qt/learning_materials_knn.py), [`src/presentation/qt/learning_materials_svm.py`](src/presentation/qt/learning_materials_svm.py), [`src/presentation/qt/learning_materials_logistic.py`](src/presentation/qt/learning_materials_logistic.py), [`src/presentation/qt/learning_materials_decision_tree.py`](src/presentation/qt/learning_materials_decision_tree.py), [`src/presentation/qt/learning_materials_random_forest.py`](src/presentation/qt/learning_materials_random_forest.py) | JCSL, preprocessing, and classifier reference pages shown from the learning-materials dialog. |
| [`src/presentation/qt/style.py`](src/presentation/qt/style.py), [`src/presentation/qt/style_base.py`](src/presentation/qt/style_base.py), [`src/presentation/qt/style_controls.py`](src/presentation/qt/style_controls.py), [`src/presentation/qt/style_data.py`](src/presentation/qt/style_data.py), [`src/presentation/qt/style_dialogs.py`](src/presentation/qt/style_dialogs.py), [`src/presentation/qt/style_panels.py`](src/presentation/qt/style_panels.py), [`src/presentation/qt/style_chrome.py`](src/presentation/qt/style_chrome.py), [`src/presentation/qt/palette.py`](src/presentation/qt/palette.py) | Central palette, QSS fragments, dark/light theme construction, chrome, panels, controls, dialogs, data widgets, and palette tokens used by plots/reports. |
| [`src/presentation/qt/main_window_theme.py`](src/presentation/qt/main_window_theme.py), [`src/presentation/qt/ui_options.py`](src/presentation/qt/ui_options.py) | Theme switching and the canonical normalization, standardization, kernel, classifier, and display option values. |
| [`src/presentation/qt/__init__.py`](src/presentation/qt/__init__.py), [`src/presentation/__init__.py`](src/presentation/__init__.py), [`src/application/__init__.py`](src/application/__init__.py), [`src/domain/__init__.py`](src/domain/__init__.py), [`src/infrastructure/__init__.py`](src/infrastructure/__init__.py), [`src/infrastructure/data/__init__.py`](src/infrastructure/data/__init__.py), [`src/infrastructure/methods/__init__.py`](src/infrastructure/methods/__init__.py), [`src/infrastructure/persistence/__init__.py`](src/infrastructure/persistence/__init__.py) | Package markers and public import surfaces; they do not add runtime workflow logic. |

### 11. Static assets

[`src/presentation/qt/assets/`](src/presentation/qt/assets/) contains the
application icon plus the SVG chevrons, spin-box arrows, and check mark used
by the Qt stylesheet. They are loaded by the style/theme layer and bundled by
the desktop build.

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
