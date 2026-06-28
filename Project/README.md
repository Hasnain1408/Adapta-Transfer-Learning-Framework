# Transfer Learning Desktop App (PyQt)

A clean-architecture desktop application for running Transfer Learning methods.

## Current Features

- Method dropdown (currently includes `TCA` only)
- Dynamic source and target CSV/XLSX dataset inputs
- Background execution using `QThread`
- Result preview table for transformed target dataset

## Folder Structure

```text
transfer_learning_desktop_app/
  requirements.txt
  sample_data/
    source.csv
    target.xlsx
  src/
    main.py
    domain/
      interfaces.py
      models.py
    application/
      run_transfer_learning_use_case.py
    infrastructure/
      data/
        csv_dataset_loader.py
      methods/
        tca_method.py
    presentation/
      qt/
        main_window.py
        worker.py
```

## Setup

```bash
cd transfer_learning_desktop_app
pip install -r requirements.txt
```

## Run

```bash
PYTHONPATH=src python src/main.py
```

## Quick Startup Test (headless)

```bash
PYTHONPATH=src python src/main.py --self-test
```

## Dataset Format

- CSV or XLSX file required
- Last column is treated as label
- All other columns are treated as numeric features
