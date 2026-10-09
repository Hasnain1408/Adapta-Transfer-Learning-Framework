# Adapta — Transfer Learning Studio

<p align="center">
  <strong>A Clean-Architecture Desktop Environment for Domain Adaptation and Transfer Learning Research & Experimentation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/GUI-PyQt5-green?logo=qt&logoColor=white" alt="PyQt5" />
  <img src="https://img.shields.io/badge/Architecture-Clean%20Architecture-orange" alt="Clean Architecture" />
  <img src="https://img.shields.io/badge/Database-PostgreSQL%20(Optional)-336791?logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Cross--Platform-lightgrey" alt="Platform" />
  <img src="https://img.shields.io/badge/Academic-SPL--3%20%7C%20BSSE-purple" alt="Academic SPL-3" />
</p>

---

## 📌 Executive Summary

**Adapta** is an enterprise-grade, clean-architecture desktop application built with **Python** and **PyQt5** designed for benchmarking, analyzing, and visualizing **Homogeneous Domain Adaptation** and **Transfer Learning** algorithms.

Domain Adaptation bridges the distribution gap between a labeled *source domain* and an unlabeled or target domain. **Adapta** provides an end-to-end laboratory where researchers and students can load heterogeneous dataset formats (`.csv`, `.xlsx`, `.arff`, `.mat`), apply mathematical preprocessing pipelines, run state-of-the-art transfer learning algorithms, evaluate baseline vs. adapted performance across multiple classifiers, visualize feature embeddings (PCA / t-SNE / MMD alignment), compare multiple methods side-by-side, and generate comprehensive publication-ready PDF reports.

> Developed as the capstone deliverable for **Software Project Lab 3 (SPL-3)** in the Bachelor of Science in Software Engineering (BSSE) curriculum.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Architecture & Design Principles](#-architecture--design-principles)
- [Supported Domain Adaptation Methods](#-supported-domain-adaptation-methods)
- [Preprocessing & Classification Pipeline](#-preprocessing--classification-pipeline)
- [Dataset Compatibility](#-dataset-compatibility)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Packaging as a Desktop Application](#-packaging-as-a-desktop-application)
- [Database Persistence (Optional)](#-database-persistence-optional)
- [Extending the Platform](#-extending-the-platform)
- [Project Artifacts & Deliverables](#-project-artifacts--deliverables)
- [License & Academic Credits](#-license--academic-credits)

---

## 🌟 Key Features

### 1. 🧮 Comprehensive Transfer Learning Suite
- **11 Domain Adaptation Algorithms**: Transfer Component Analysis (TCA), Joint Distribution Adaptation (JDA), Correlation Alignment (CORAL), Balanced Distribution Adaptation (BDA), Orthogonal Core Multi-Task Learning (OCMTL), Domain Adaptation via Locality and Invariance (DALI), Geodesic Flow Kernel (GFK), Joint Geometrical and Statistical Alignment (JGSA), Joint Classification and Subspace Learning (JCSL), Manifold Embedded Distribution Alignment (MEDA), and a custom **Proposed Method**.
- **Dynamic Parameter Binding**: UI dynamically activates parameter controls (Kernel type, Regularization \lambda, Subspace Latent Dimension d) based on the active algorithm's theoretical capabilities.

### 2. 🔬 Robust Pipeline & Diagnostics
- **Configurable Preprocessing**: 10 Normalization techniques and 6 Standardization strategies executed independently per domain.
- **Pluggable Supervised Classifiers**: Train downstream classifiers on adapted source representations (k-NN, Support Vector Machines, Random Forest, Logistic Regression, Decision Tree).
- **Comprehensive Metrics**: Quantitative evaluation with Accuracy, Precision, Recall, Macro/Micro F1-Score, and ROC-AUC along with per-instance predictions and classification confusion data.
- **Negative Transfer & Divergence Analysis**: Real-time measurement of Maximum Mean Discrepancy (MMD) reduction and distribution overlap.

### 3. 📊 Interactive Visualizations & Analytics
- **Dimensionality Reduction**: Visual 2D/3D distribution plots using PCA and t-SNE before and after adaptation.
- **Metric Comparison Radar & Bar Charts**: Compare accuracy improvements and distribution shifts against the unadapted baseline.
- **Automated Publication-Ready Reports**: High-quality HTML/PDF report generation containing tables, metrics, parameter configurations, and distribution figures.

### 4. 🚀 Advanced Experimentation Modes
- **Single Experiment Workspace**: Interactive tweaking of parameters, real-time logs, and granular inspection.
- **Batch Evaluation Engine**: Execute all ordered (Source, Target) permutations across arbitrary batches of dataset files with aggregation.
- **Multi-Method Benchmark**: Compare up to 11 algorithms sequentially on the exact same dataset pair under identical baseline conditions.

### 5. 📚 Built-in Interactive Learning Studio
- In-app interactive learning module explaining theoretical foundations, mathematical formulations, and flowcharts for every supported algorithm, preprocessing technique, and classifier.

### 6. 🗄️ Optional Enterprise Database Persistence
- Full authentication system (Admin, Researcher, Teacher, Student roles) backed by PostgreSQL.
- Historical experiment repository, dataset registry, and long-term benchmark tracking (works 100% offline if database is absent).

---

## 📂 Repository Structure

```text
.
├── Project/                      # Core application source code & runtime environment
│   ├── src/                      # Clean Architecture Python implementation
│   │   ├── domain/               # Core domain entities, contracts, interfaces, and metrics
│   │   ├── application/          # Use cases, orchestration, and pipeline services
│   │   ├── infrastructure/       # Loaders, TL algorithms, and PostgreSQL repositories
│   │   └── presentation/qt/      # PyQt5 GUI, custom components, themes, and learning studio
│   ├── Datasets/                 # Benchmark datasets (MNIST/USPS, RELINK, NASA, etc.)
│   ├── Adapta.spec               # PyInstaller specification for standalone binary compilation
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # PostgreSQL configuration template
│   └── cmd.txt                   # Quick terminal commands reference
├── Proposal/                     # Formal Project Proposal (BSSE SPL-3)
│   ├── Project Proposal.pdf      # Detailed proposal specification
│   └── Proposal Presentation.pptx# Defense presentation slide deck
├── SRS/                          # Software Requirements Specification
│   ├── BSSE1408.pdf              # IEEE 830-standard SRS document
│   └── BSSE1408.pptx             # SRS defense presentation
├── note(spl).txt                 # Developer notes and deployment instructions
└── README.md                     # Root project documentation
```

---

## 🏗️ Architecture & Design Principles

Adapta strictly adheres to **Clean Architecture** (Ports and Adapters / Hexagonal) with pure inward dependency flows:

```
+-------------------------------------------------------------------------+
|                          Presentation Layer (Qt)                        |
|   [MainWindow] ---> [Workers / QThread] ---> [Visualizations & Reports] |
+------------------------------------+------------------------------------+
                                     | calls
                                     v
+-------------------------------------------------------------------------+
|                         Application Layer (Use Cases)                   |
|   [RunTransferLearningUseCase] [AuthenticationUseCase] [CatalogUseCase] |
|   - Preprocessing Pipeline     - Classifier Pipeline   - Alignment Calc |
+------------------------------------+------------------------------------+
                                     | depends on
                                     v
+-------------------------------------------------------------------------+
|                            Domain Layer (Core)                          |
|   [Entities: Dataset, Request, Result, Metrics, User, ProgressUpdate]   |
|   [Interfaces: TransferLearningMethod, DatasetLoader, Repositories]     |
+------------------------------------+------------------------------------+
                                     ^ implements interfaces
                                     |
+------------------------------------+------------------------------------+
|                      Infrastructure Layer (Adapters)                    |
|   - Algorithm Implementations: TCA, JDA, CORAL, BDA, MEDA, etc.         |
|   - Dataset Readers: CSV, Excel (.xlsx), ARFF, MATLAB (.mat)            |
|   - PostgreSQL Persistence: Repositories & Connection Management        |
+-------------------------------------------------------------------------+
```

### Architectural Highlights:
- **Zero UI Leakage**: Domain models and application pipelines contain no Qt or UI dependencies, making core transfer learning routines headless-testable and reusable.
- **Asynchronous Execution**: Computationally demanding adaptation and iterative optimization routines run on background `QThread` workers with thread-safe `RunControl` hooks for real-time progress reporting, pause, resume, and graceful cancellation.
- **Open-Closed Principle**: Adding a new algorithm requires only implementing `TransferLearningMethod` under `infrastructure/methods/` and registering it in `src/main.py`.

---

## 🧠 Supported Domain Adaptation Methods

| Algorithm | Method Name | Kernel Support | Regularization (\lambda) | Latent Dim (d) | Primary Objective / Description |
|:---|:---|:---:|:---:|:---:|:---|
| **TCA** | Transfer Component Analysis | Primal, Linear, RBF | ✅ | ✅ | Minimizes MMD across domains in RKHS while preserving data variance. |
| **JDA** | Joint Distribution Adaptation | Primal, Linear, RBF | ✅ | ✅ | Simultaneously adapts marginal P(X) and conditional P(Y \mid X) distributions using target pseudo-labels. |
| **CORAL** | Correlation Alignment | — | — | — | Aligns second-order statistics (covariance matrices) of source and target domains via linear transformation. |
| **BDA** | Balanced Distribution Adaptation | Primal, Linear, RBF | ✅ | ✅ | Adaptively balances marginal and conditional distribution discrepancy using a balance factor \mu. |
| **OCMTL** | Orthogonal Core Multi-Task Learning | — | — | ✅ | Learns orthogonal core subspace transformations shared across domains. |
| **DALI** | Domain Adaptation via Locality & Invariance | — | ✅ | ✅ | Combines distribution alignment with locality preserving projections (LPP) and discriminative scatter. |
| **GFK** | Geodesic Flow Kernel | — | — | ✅ | Models domain shifts along the geodesic path on a Grassmann manifold. |
| **JGSA** | Joint Geometrical & Statistical Alignment | — | ✅ | ✅ | Coupled projection reducing statistical divergence while aligning geometrical domain structures. |
| **JCSL** | Joint Classification & Subspace Learning | — | ✅ | ✅ | Jointly optimizes target subspace alignment and source margin classifier via mini-batch SGD. |
| **MEDA** | Manifold Embedded Distribution Alignment | Linear, RBF | ✅ | ✅ | Learns domain-invariant classifier under Grassmann manifold embedding and dynamic distribution alignment. |
| **Proposed** | Hybrid Covariance-Core Adaptation | — | ✅ | — | Novel hybrid formulation combining covariance matching, SVD orthogonal core extraction, and iterative pseudo-labeling. |

---

## ⚙️ Preprocessing & Classification Pipeline

### 1. Normalization Techniques
- Mean Centering
- Z-Score Standardization (\mu=0, \sigma=1)
- Frobenius Normalization
- Spectral Normalization
- L1 Normalization (Taxicab unit norm)
- Min-Max Normalization [0, 1]
- Column-wise Unit Norm
- Robust Standardization (Median / IQR)

### 2. Downstream Classifiers
- **k-NN**: k-Nearest Neighbors (k=1)
- **Logistic Regression**: Linear model with multinomial/cross-entropy optimization
- **Random Forest**: Ensemble classifier with bagging
- **Decision Tree**: CART decision tree classifier
- **SVM**: Support Vector Machine with Radial Basis Function kernel

---

## 📁 Dataset Compatibility

The composite `DatasetLoader` automatically recognizes file extensions and parses tabular and matrix formats:

| Format | Extension | Structure & Parsing Rules |
|:---|:---|:---|
| **MATLAB Workspace** | `.mat` | Dual-mode parsing: detects combined domain variables (`Xs`, `Xt`, `Ys`, `Yt`, `fea`, `gnd`) or extracts dominant 2D array and target vectors. |
| **Comma-Separated Values** | `.csv` | Tabular; numeric features coerced; last column interpreted as ground-truth class label. |
| **Microsoft Excel** | `.xlsx` | Multi-column spreadsheet parsed via `openpyxl`; non-numeric values replaced by 0.0. |
| **Weka ARFF** | `.arff` | Attribute-Relation File Format; decodes nominal string/byte categories into standard discrete labels. |

---

## 🚀 Installation & Setup

### Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows
- **Python**: Version `3.10` or higher
- **PostgreSQL**: Version `13+` (Optional, required only for persistence and user accounts)

### 1. Clone the Repository
```bash
git clone https://github.com/Hasnain1408/Adapta-Transfer-Learning-Framework.git
cd SPL-3
```

### 2. Create and Activate a Virtual Environment
```bash
cd Project
python3 -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Launch GUI
```bash
# Ensure virtual environment is active inside Project/
python3 src/main.py
```

### Headless Verification / Smoke Test
Verify that dependency wiring, Qt initialization, and threading pass cleanly:
```bash
QT_QPA_PLATFORM=offscreen python3 src/main.py --self-test
```

### Offline Mode (Bypass Database)
```bash
python3 src/main.py --no-database
```

---

## 📦 Packaging as a Desktop Application

You can build a standalone Linux binary using the included PyInstaller specification:

### 1. Build Binary
```bash
cd Project
pyinstaller --clean Adapta.spec
```
The compiled standalone executable will be generated at `Project/dist/Adapta`.

### 2. Install Desktop Entry (Linux `.desktop` Launcher)
To integrate Adapta with your Linux desktop application menu:

1. Create a desktop launcher file:
   ```bash
   nano ~/.local/share/applications/Adapta.desktop
   ```
2. Paste the following configuration (adjust path if needed):
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Adapta Transfer Learning Studio
   Comment=Run transfer-learning and domain-adaptation experiments
   Exec=/home/hasnain/Desktop/BSSE/Semester-08/SPL-3/Project/dist/Adapta
   Icon=/home/hasnain/Desktop/BSSE/Semester-08/SPL-3/Project/src/presentation/qt/assets/app_icon.svg
   Terminal=false
   Categories=Science;Education;Development;
   StartupNotify=true
   ```
3. Register the desktop file:
   ```bash
   chmod +x ~/.local/share/applications/Adapta.desktop
   update-desktop-database ~/.local/share/applications/
   ```

---

## 🗄️ Database Persistence (Optional)

Adapta includes an optional PostgreSQL storage backend for recording experimental history, managing user accounts (RBAC), and cataloging benchmark datasets.

### Configuration
1. Copy the environment template:
   ```bash
   cd Project
   cp .env.example .env
   ```
2. Configure your connection string in `.env`:
   ```env
   DATABASE_URL=postgresql://<username>:<password>@<host>:<port>/<database>?sslmode=require
   ```
3. Start the application normally:
   ```bash
   python3 src/main.py
   ```
*If `.env` is omitted or invalid, the application seamlessly defaults to offline in-memory execution mode without crashing.*

---

## 🔌 Extending the Platform

### Adding a New Domain Adaptation Method

1. **Create the Method Class**: Implement `TransferLearningMethod` in `src/infrastructure/methods/`:
   ```python
   from domain.interfaces import TransferLearningMethod
   from domain.run_control import RunControl
   import numpy as np

   class MyNewMethod(TransferLearningMethod):
       def __init__(self, latent_dim: int = 10, regularization: float = 1.0):
           self._latent_dim = latent_dim
           self._regularization = regularization

       @property
       def name(self) -> str:
           return "My New Method"

       def run(
           self,
           source_features: np.ndarray,
           target_features: np.ndarray,
           source_labels: np.ndarray,
           control: RunControl,
       ) -> tuple[np.ndarray, np.ndarray]:
           # 1. Periodically check pause/cancel tokens
           control.checkpoint()

           # 2. Compute adaptation transformations
           # ...

           # 3. Report intermediate progress if iterative
           control.report_iteration(current=1, total=10)

           return transformed_source, transformed_target
   ```

2. **Register the Method**: Add the instantiation in `src/main.py` inside `build_window()`:
   ```python
   from infrastructure.methods.my_new_method import MyNewMethod

   methods = [
       # ...
       MyNewMethod(),
   ]
   ```
*The GUI dropdowns, parameter guards, batch runner, and comparison matrices will automatically discover and support the new method.*

---

## 📄 Project Artifacts & Deliverables

The repository includes complete academic and project documentation:
- **Proposal**: Located in [`Proposal/`](Proposal/)
  - `Project Proposal.pdf`: Comprehensive project roadmap, background, problem statement, and scope.
  - `Proposal Presentation.pptx`: Proposal defense presentation deck.
- **SRS (Software Requirements Specification)**: Located in [`SRS/`](SRS/)
  - `BSSE1408.pdf`: Complete IEEE 830 compliant functional and non-functional specifications.
  - `BSSE1408.pptx`: System design and requirements presentation.

---

## 📜 License & Academic Credits

- **Course**: Software Project Lab 3 (SPL-3)
- **Degree**: Bachelor of Science in Software Engineering (BSSE)
- **Author**: Hasnain Sheikh (BSSE 1408)
- **Supervision & Review**: Dr. Mohammad Shoyaib, 
Professor, Institute of Information Technology (IIT), University of Dhaka

*This project is released for academic, research, and educational purposes.*
