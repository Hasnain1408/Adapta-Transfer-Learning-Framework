# Jupyter Notebook Setup and Run Guide (VS Code)

This guide explains the full process to run a `.ipynb` notebook in VS Code, including:
- Creating a virtual environment (venv)
- Installing dependencies
- Running notebook cells in the correct environment
- Running Python code inside that same venv

This is tailored for this project and notebook:
- `code practice/TCA-algorithm/tca-algo.ipynb`

## 1. Prerequisites

Install these first:
1. Python 3.10+ (3.12 works)
2. VS Code
3. VS Code extensions:
   - Python (Microsoft)
   - Jupyter (Microsoft)

Check Python:

```bash
python3 --version
```

## 2. Open the Project in VS Code

Open this folder in VS Code:

```bash
/home/hasnain/Desktop/BSSE/SPL-3
```

Then open:

```text
code practice/TCA-algorithm/tca-algo.ipynb
```

## 3. Create a Virtual Environment (venv)

From workspace root:

```bash
cd "/home/hasnain/Desktop/BSSE/SPL-3"
python3 -m venv .venv
```

This creates:

```text
/home/hasnain/Desktop/BSSE/SPL-3/.venv
```

## 4. Activate venv

Linux/macOS:

```bash
source .venv/bin/activate
```

If active, your shell prompt shows `(.venv)`.

## 5. Install Notebook Dependencies in venv

With venv activated:

```bash
python3 -m pip install --upgrade pip
pip install -r "code practice/TCA-algorithm/requirements.txt"
```

Why these packages are needed:
- `jupyter`, `ipykernel`: notebook kernel/runtime
- `numpy`, `scipy`: matrix operations and eigen decomposition
- `pandas`: data loading and tabular handling
- `scikit-learn`: LogisticRegression and metrics
- `matplotlib`, `seaborn`: plots
- `openpyxl`: required for reading local `.xlsx` NASA files

## 6. Select the Correct Python Interpreter in VS Code

1. Press `Ctrl+Shift+P`
2. Run: `Python: Select Interpreter`
3. Choose:

```text
/home/hasnain/Desktop/BSSE/SPL-3/.venv/bin/python
```

## 7. Select Notebook Kernel

In `tca-algo.ipynb`:
1. Click `Select Kernel` (top-right)
2. Choose Python from `.venv`

Expected kernel interpreter should point to `.venv/bin/python`.

## 8. Recommended Execution Flow for tca-algo.ipynb

Because this notebook currently has helper cells and some sections out of conceptual order, use this safe workflow:

1. `Kernel` -> `Restart Kernel and Clear All Outputs`
2. Run install/helper cells only if needed (only when package import errors appear)
3. Run imports cell
4. Run dataset loader definition cell(s)
5. Run dataset loading cell
6. Run TCA function definition cell
7. Run preprocessing/evaluation function cells
8. Run experiment cells (CM1->JM1, JM1->KC1, KC1->PC1)
9. Run visualization and summary cells

If a cell fails with `ModuleNotFoundError`, install missing package in the same active kernel:

```python
%pip install <package-name>
```

Then restart kernel and rerun from imports.

## 9. Running Code in the Same venv (Terminal)

After activation:

```bash
source /home/hasnain/Desktop/BSSE/SPL-3/.venv/bin/activate
python your_script.py
```

One-line run without activation:

```bash
/home/hasnain/Desktop/BSSE/SPL-3/.venv/bin/python your_script.py
```

This guarantees script execution uses the same environment as notebook kernel.

## 10. Optional: Save Dependencies to requirements.txt

From active venv:

```bash
pip freeze > requirements.txt
```

Install later on another machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 11. Troubleshooting

### A) Kernel dies with `No module named ipykernel_launcher`
Cause: kernel interpreter does not have `ipykernel`.

Fix:
```bash
source .venv/bin/activate
pip install ipykernel jupyter
```
Then reselect kernel in notebook.

### B) `ModuleNotFoundError: pandas` (or numpy/sklearn/etc.)
Cause: package not installed in selected kernel environment.

Fix:
```python
%pip install pandas
```
Restart kernel, rerun cells.

### C) `ImportError: openpyxl`
Cause: pandas cannot read `.xlsx` without `openpyxl`.

Fix:
```python
%pip install openpyxl
```

### D) Dataset URL returns 404
Cause: remote source moved or unavailable.

Fix options:
1. Use local NASA dataset files from:
   - `code practice/Datasets/NASA/`
2. Update URLs in loader to valid sources.

## 12. Quick Command Checklist

```bash
cd "/home/hasnain/Desktop/BSSE/SPL-3"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install jupyter ipykernel numpy scipy pandas scikit-learn matplotlib seaborn openpyxl
```

Then in VS Code:
1. Select interpreter `.venv/bin/python`
2. Open `tca-algo.ipynb`
3. Select same kernel
4. Restart kernel
5. Run cells in order

---

If you keep notebook work and terminal runs on the same `.venv`, package mismatch and kernel issues are minimized.