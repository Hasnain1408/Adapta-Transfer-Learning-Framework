from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextDocument
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar


class LearningMaterialsDialog(QDialog):
    """Dialog surfacing comprehensive learning materials for Transfer Learning methods,
    classifiers, evaluation metrics, and feature preprocessing techniques.
    """

    def __init__(self, palette: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setWindowTitle("Learning Materials & Domain Adaptation Guide")
        self.resize(1024, 680)
        self.setMinimumSize(850, 550)
        self.setStyleSheet(build_style_sheet(palette))

        self._build_ui()

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self.setStyleSheet(build_style_sheet(palette))
        self._update_all_content()

    def select_topic(self, category_index: int, item_index: int = 0) -> None:
        """Utility to programmatically open a specific tab and topic."""
        if 0 <= category_index < self._tabs.count():
            self._tabs.setCurrentIndex(category_index)
            list_widget = self._category_lists[category_index]
            if 0 <= item_index < list_widget.count():
                list_widget.setCurrentRow(item_index)

    def _build_ui(self) -> None:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("📖 Learning Materials")
        title.setObjectName("appTitle")
        subtitle = QLabel("Comprehensive reference for transfer learning methods, classifiers, metrics, and data scaling")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch(1)

        layout.addWidget(header)

        # Main Tab Widget
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._category_lists: list[QListWidget] = []
        self._category_browsers: list[QTextBrowser] = []

        categories = [
            ("Transfer Learning Methods", self._get_tl_methods_data()),
            ("Classifiers", self._get_classifiers_data()),
            ("Evaluation Metrics", self._get_metrics_data()),
            ("Normalization & Scaling", self._get_preprocessing_data()),
        ]

        for tab_name, items in categories:
            tab_widget, list_widget, browser = self._build_category_tab(items)
            self._tabs.addTab(tab_widget, tab_name)
            self._category_lists.append(list_widget)
            self._category_browsers.append(browser)

        layout.addWidget(self._tabs, 1)

        install_title_bar(self, "Learning Materials & Domain Adaptation Guide", body)

    def _build_category_tab(
        self, items: list[tuple[str, str]]
    ) -> tuple[QWidget, QListWidget, QTextBrowser]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Sidebar List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Filter topics…")
        search_input.setClearButtonEnabled(True)

        list_widget = QListWidget()
        list_widget.setFixedWidth(240)

        for title, _ in items:
            list_widget.addItem(title)

        left_layout.addWidget(search_input)
        left_layout.addWidget(list_widget, 1)

        # Right panel: Content Browser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        splitter.addWidget(left_widget)
        splitter.addWidget(browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Connect signals
        def on_selection_changed():
            row = list_widget.currentRow()
            if 0 <= row < len(items):
                html = self._render_article_html(items[row][0], items[row][1])
                browser.setHtml(html)

        list_widget.currentRowChanged.connect(on_selection_changed)

        def filter_items(text: str):
            text_lower = text.lower().strip()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setHidden(text_lower not in item.text().lower())

        search_input.textChanged.connect(filter_items)

        if items:
            list_widget.setCurrentRow(0)

        return container, list_widget, browser

    def _update_all_content(self) -> None:
        categories_data = [
            self._get_tl_methods_data(),
            self._get_classifiers_data(),
            self._get_metrics_data(),
            self._get_preprocessing_data(),
        ]
        for idx, items in enumerate(categories_data):
            list_widget = self._category_lists[idx]
            browser = self._category_browsers[idx]
            row = list_widget.currentRow()
            if 0 <= row < len(items):
                html = self._render_article_html(items[row][0], items[row][1])
                browser.setHtml(html)

    def _render_article_html(self, title: str, body_content: str) -> str:
        p = self._palette
        return f"""
        <html>
        <head>
        <style>
            body {{
                font-family: 'Segoe UI', 'Inter', Helvetica, Arial, sans-serif;
                background-color: {p['surface']};
                color: {p['text']};
                margin: 16px;
                line-height: 1.6;
                font-size: 13px;
            }}
            h1 {{
                color: {p['accent']};
                font-size: 20px;
                margin-top: 0;
                margin-bottom: 8px;
                border-bottom: 2px solid {p['border']};
                padding-bottom: 6px;
            }}
            h2 {{
                color: {p['text']};
                font-size: 15px;
                margin-top: 18px;
                margin-bottom: 6px;
            }}
            h3 {{
                color: {p['text_secondary']};
                font-size: 13px;
                margin-top: 12px;
                margin-bottom: 4px;
            }}
            p {{
                margin-top: 4px;
                margin-bottom: 12px;
                color: {p['text']};
            }}
            .badge {{
                display: inline-block;
                background-color: {p['accent_soft']};
                color: {p['accent']};
                font-weight: bold;
                font-size: 11px;
                padding: 3px 8px;
                border-radius: 4px;
                margin-bottom: 12px;
            }}
            .card {{
                background-color: {p['surface_alt']};
                border: 1px solid {p['border_strong']};
                border-radius: 8px;
                padding: 12px 14px;
                margin: 10px 0;
            }}
            .formula {{
                font-family: 'Consolas', 'Courier New', monospace;
                background-color: {p['surface_sunken']};
                color: {p['metric_teal']};
                padding: 8px 12px;
                border-left: 3px solid {p['accent']};
                border-radius: 4px;
                font-size: 13px;
                margin: 8px 0;
            }}
            ul {{
                margin-top: 4px;
                margin-bottom: 12px;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 4px;
            }}
            strong {{
                color: {p['text']};
            }}
        </style>
        </head>
        <body>
            <h1>{title}</h1>
            {body_content}
        </body>
        </html>
        """

    # ------------------------------------------------------------------
    # Data Providers
    # ------------------------------------------------------------------
    def _get_tl_methods_data(self) -> list[tuple[str, str]]:
        return [
            (
                "TCA (Transfer Component Analysis)",
                """
                <span class="badge">Feature-Based Domain Adaptation</span>
                <p><strong>Transfer Component Analysis (TCA)</strong> aims to find a shared lower-dimensional subspace where the domain discrepancy between the source and target data distributions is minimized, while preserving the geometric structure and variance of the data.</p>
                
                <h2>Core Mechanism</h2>
                <p>TCA maps high-dimensional data into a Reproducing Kernel Hilbert Space (RKHS) using a kernel function (Linear, RBF, or Spectral). It computes the <strong>Maximum Mean Discrepancy (MMD)</strong> distance between the source and target domain marginal distributions $P(X_S)$ and $P(X_T)$ and minimizes it.</p>
                
                <div class="formula">
                min_W  tr(W^T K L K W) + λ tr(W^T W)<br/>
                s.t.  W^T K H K W = I
                </div>
                
                <h2>Key Hyperparameters</h2>
                <ul>
                    <li><strong>Latent Dimensions (k):</strong> Number of projection vectors in the target subspace. Higher dimensions capture more variance but may retain domain-specific noise.</li>
                    <li><strong>Regularization (λ):</strong> Controls the trade-off between variance preservation and domain distance minimization.</li>
                    <li><strong>Kernel Type:</strong> Determines the non-linear mapping (Linear, RBF/Gaussian, SAM spectral angle mapper).</li>
                </ul>
                
                <h2>When to Use</h2>
                <ul>
                    <li>When feature dimensions are continuous and subject to marginal distribution shift $P(X_S) \\neq P(X_T)$.</li>
                    <li>When target labels are completely unannotated during adaptation.</li>
                </ul>
                """,
            ),
            (
                "JDA (Joint Domain Adaptation)",
                """
                <span class="badge">Joint Marginal & Conditional Adaptation</span>
                <p><strong>Joint Domain Adaptation (JDA)</strong> extends TCA by simultaneously adapting both the <em>Marginal Distribution</em> $P(X)$ and the <em>Conditional Distribution</em> $P(Y|X)$ across source and target domains.</p>
                
                <h2>Core Mechanism</h2>
                <p>Because target labels are unavailable initially, JDA utilizes a base classifier trained on source data to generate <strong>pseudo-labels</strong> for the target domain. It then computes MMD distance matrices for both global marginal features and class-wise conditional features, iteratively refining target predictions over multiple iterations.</p>
                
                <div class="formula">
                min_W  tr(W^T K (L_0 + \\sum_{c=1}^C L_c) K W) + λ tr(W^T W)
                </div>
                
                <h2>Key Hyperparameters</h2>
                <ul>
                    <li><strong>Iterations:</strong> Number of pseudo-label update steps (typically 5–10 iterations until predictions stabilize).</li>
                    <li><strong>Latent Dimensions:</strong> Dimension of the joint subspace.</li>
                    <li><strong>Regularization (λ):</strong> Matrix conditioning term.</li>
                </ul>
                
                <h2>When to Use</h2>
                <ul>
                    <li>When domain shift affects both global feature distribution and class-conditional boundaries.</li>
                    <li>When iterative refinement significantly improves domain alignment quality.</li>
                </ul>
                """,
            ),
            (
                "CORAL (Correlation Alignment)",
                """
                <span class="badge">Second-Order Statistic Alignment</span>
                <p><strong>Correlation Alignment (CORAL)</strong> is a lightweight yet effective domain adaptation method that minimizes domain shift by aligning the second-order statistics (covariance matrices) of the source and target feature distributions.</p>
                
                <h2>Core Mechanism</h2>
                <p>CORAL applies a linear transformation to the source features to match the covariance matrix of the target domain features without projecting into high-dimensional kernel spaces.</p>
                
                <div class="formula">
                C_S = cov(X_S), &nbsp; C_T = cov(X_T)<br/>
                X_S^{aligned} = X_S \\cdot C_S^{-1/2} \\cdot C_T^{1/2}
                </div>
                
                <h2>Key Highlights</h2>
                <ul>
                    <li><strong>Parameter-Free:</strong> Does not require tuning latent dimensions or kernel functions.</li>
                    <li><strong>Computationally Efficient:</strong> Solves covariance decomposition directly using SVD / Matrix square roots.</li>
                </ul>
                
                <h2>When to Use</h2>
                <ul>
                    <li>For rapid baseline domain adaptation across large feature spaces.</li>
                    <li>When domain discrepancy is dominated by linear covariance mismatch.</li>
                </ul>
                """,
            ),
            (
                "BDA (Balanced Domain Adaptation)",
                """
                <span class="badge">Adaptive Distribution Balancing</span>
                <p><strong>Balanced Domain Adaptation (BDA)</strong> addresses the fundamental challenge in transfer learning where either marginal shift $P(X)$ or conditional shift $P(Y|X)$ may dominate the domain discrepancy.</p>
                
                <h2>Core Mechanism</h2>
                <p>BDA introduces a dynamic balance factor $\\mu \\in [0, 1]$ to weight the contribution of marginal and conditional MMD matrix terms dynamically.</p>
                
                <div class="formula">
                D_{BDA} = (1 - \\mu) D_{marginal} + \\mu \\sum_{c=1}^C D_{conditional}^{(c)}
                </div>
                
                <ul>
                    <li><strong>μ → 0:</strong> Emphasizes marginal distribution alignment (equivalent to TCA).</li>
                    <li><strong>μ → 1:</strong> Emphasizes class conditional distribution alignment.</li>
                    <li><strong>μ ≈ 0.5:</strong> Balanced adaptation (W-BDA).</li>
                </ul>
                
                <h2>When to Use</h2>
                <ul>
                    <li>When the relative importance of marginal vs conditional shift is unknown prior to experimentation.</li>
                </ul>
                """,
            ),
            (
                "OCMTL (Optimal Class-Margin Transfer Learning)",
                """
                <span class="badge">Margin-Maximizing Subspace Learning</span>
                <p><strong>Optimal Class-Margin Transfer Learning (OCMTL)</strong> combines subspace domain alignment with explicit class margin maximization.</p>
                
                <h2>Core Mechanism</h2>
                <p>While aligning source and target domain feature distributions, OCMTL optimizes a projection matrix that enforces wide decision margins between distinct class clusters in the source domain. This prevents ambiguous class overlap in the transformed target domain.</p>
                
                <h2>Key Hyperparameters</h2>
                <ul>
                    <li><strong>Latent Dimensions:</strong> Size of the projected margin space.</li>
                    <li><strong>Max Iterations:</strong> Max optimization steps for convergence.</li>
                </ul>
                """,
            ),
        ]

    def _get_classifiers_data(self) -> list[tuple[str, str]]:
        return [
            (
                "k-NN (5-Nearest Neighbor)",
                """
                <span class="badge">Instance-Based Non-Parametric</span>
                <p>The <strong>k-Nearest Neighbors (k-NN)</strong> algorithm predicts the class label of a target sample based on the majority vote of its <em>k</em> (default: 5) nearest neighbors in the adapted feature space.</p>
                
                <div class="card">
                <strong>Distance Metric:</strong> Standard Euclidean Distance $d(x, y) = \\sqrt{\\sum (x_i - y_i)^2}$
                </div>
                
                <h2>Strengths & Limitations</h2>
                <ul>
                    <li><strong>Pros:</strong> Non-parametric, makes no assumption about class linearity, effective after successful manifold alignment.</li>
                    <li><strong>Cons:</strong> Sensitive to noisy features and unscaled dimensions (mitigated by preprocessing).</li>
                </ul>
                """,
            ),
            (
                "SVM (Support Vector Machine)",
                """
                <span class="badge">Maximum Margin Classifier</span>
                <p><strong>Support Vector Machines (SVM)</strong> construct an optimal decision boundary (hyperplane) that maximizes the margin of separation between classes in the transformed domain space.</p>
                
                <div class="card">
                <strong>Kernel Function:</strong> Radial Basis Function (RBF / Gaussian) or Linear Kernel.<br/>
                <strong>Objective:</strong> $\\min_{w, b} \\frac{1}{2} ||w||^2 + C \\sum \\xi_i$
                </div>
                
                <h2>Strengths & Limitations</h2>
                <ul>
                    <li><strong>Pros:</strong> High generalization capability, robust in high-dimensional latent spaces.</li>
                    <li><strong>Cons:</strong> Choice of regularization parameter <em>C</em> and gamma affects boundary sensitivity.</li>
                </ul>
                """,
            ),
            (
                "Random Forest",
                """
                <span class="badge">Ensemble Learning (Bagging)</span>
                <p><strong>Random Forest</strong> builds an ensemble of multiple decision trees trained on bootstrap samples with randomized feature subsets, taking the majority vote for final classification.</p>
                
                <h2>Key Highlights</h2>
                <ul>
                    <li><strong>Robustness:</strong> Highly resistant to overfitting and feature noise.</li>
                    <li><strong>Feature Importance:</strong> Naturally evaluates non-linear feature interactions.</li>
                </ul>
                """,
            ),
            (
                "Logistic Regression",
                """
                <span class="badge">Linear Probabilistic Classifier</span>
                <p><strong>Logistic Regression</strong> models class probabilities using a sigmoid (binary) or softmax (multiclass) function over linear combinations of input features.</p>
                
                <div class="formula">
                P(Y = 1 | X) = \\frac{1}{1 + e^{-(w^T X + b)}}
                </div>
                
                <h2>When to Use</h2>
                <ul>
                    <li>When linear separability is achieved through domain adaptation transformation.</li>
                    <li>When fast computation and calibrated probability outputs are needed.</li>
                </ul>
                """,
            ),
            (
                "Decision Tree",
                """
                <span class="badge">Hierarchical Rule-Based Classifier</span>
                <p><strong>Decision Tree</strong> splits data recursively at feature nodes based on impurity criteria such as Gini Impurity or Entropy.</p>
                
                <div class="formula">
                Gini = 1 - \\sum_{i=1}^C p_i^2
                </div>
                
                <h2>When to Use</h2>
                <ul>
                    <li>For transparent decision path inspection and baseline benchmarking.</li>
                </ul>
                """,
            ),
        ]

    def _get_metrics_data(self) -> list[tuple[str, str]]:
        return [
            (
                "Accuracy",
                """
                <span class="badge">Overall Classification Correctness</span>
                <p><strong>Accuracy</strong> measures the total proportion of target domain samples that were correctly classified by the model.</p>
                
                <div class="formula">
                Accuracy = \\frac{TP + TN}{TP + TN + FP + FN} = \\frac{\\text{Correct Predictions}}{\\text{Total Target Samples}}
                </div>
                
                <div class="card">
                <strong>Interpretation:</strong> An accuracy of 0.8500 indicates that 85% of all target domain samples received correct predictions.
                </div>
                """,
            ),
            (
                "Precision",
                """
                <span class="badge">Positive Predictive Value</span>
                <p><strong>Precision</strong> quantifies the proportion of positive class predictions that were actually correct.</p>
                
                <div class="formula">
                Precision = \\frac{TP}{TP + FP}
                </div>
                
                <h2>Why It Matters</h2>
                <p>High Precision means the model produces very few False Positives ($FP$), ensuring high reliability when predicting positive outcomes.</p>
                """,
            ),
            (
                "Recall (Sensitivity)",
                """
                <span class="badge">True Positive Rate</span>
                <p><strong>Recall</strong> measures the proportion of actual positive samples that were correctly identified by the model.</p>
                
                <div class="formula">
                Recall = \\frac{TP}{TP + FN}
                </div>
                
                <h2>Why It Matters</h2>
                <p>High Recall means the model produces very few False Negatives ($FN$), ensuring critical positive cases are not missed.</p>
                """,
            ),
            (
                "F1 Score",
                """
                <span class="badge">Harmonic Mean of Precision & Recall</span>
                <p>The <strong>F1 Score</strong> is the harmonic mean of Precision and Recall, providing a single balanced performance metric.</p>
                
                <div class="formula">
                F1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}
                </div>
                
                <div class="card">
                <strong>Macro Averaging in Multiclass:</strong> For datasets with >2 target classes, macro-averaging computes precision and recall independently for each class before averaging, treating all classes equally regardless of class size.
                </div>
                """,
            ),
            (
                "Confusion Matrix Terminology",
                """
                <span class="badge">Evaluation Breakdown</span>
                <p>The prediction evaluation builds upon four fundamental outcomes:</p>
                <ul>
                    <li><strong>True Positives (TP):</strong> Positive samples correctly predicted as Positive.</li>
                    <li><strong>True Negatives (TN):</strong> Negative samples correctly predicted as Negative.</li>
                    <li><strong>False Positives (FP):</strong> Negative samples incorrectly predicted as Positive (Type I error).</li>
                    <li><strong>False Negatives (FN):</strong> Positive samples incorrectly predicted as Negative (Type II error).</li>
                </ul>
                """,
            ),
        ]

    def _get_preprocessing_data(self) -> list[tuple[str, str]]:
        return [
            (
                "Min-Max Normalization",
                """
                <span class="badge">Bounded Feature Scaling [0, 1]</span>
                <p>Rescales features linearly to a fixed range between 0 and 1.</p>
                <div class="formula">x' = \\frac{x - x_{min}}{x_{max} - x_{min}}</div>
                """,
            ),
            (
                "Z-score Standardization",
                """
                <span class="badge">Zero Mean, Unit Variance</span>
                <p>Centers features around mean 0 with a standard deviation of 1.</p>
                <div class="formula">x' = \\frac{x - \\mu}{\\sigma}</div>
                """,
            ),
            (
                "Robust Standardization",
                """
                <span class="badge">Outlier-Resistant Scaling</span>
                <p>Uses Median and Interquartile Range (IQR) for scaling data containing heavy outliers.</p>
                <div class="formula">x' = \\frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}</div>
                """,
            ),
            (
                "L1 & L2 Unit Vector Normalization",
                """
                <span class="badge">Sample-wise Vector Scaling</span>
                <p>Scales individual sample rows to unit L1 norm ($||x||_1 = 1$) or unit L2 Euclidean norm ($||x||_2 = 1$).</p>
                """,
            ),
            (
                "Mahalanobis Standardization",
                """
                <span class="badge">Covariance Decorrelation</span>
                <p>Transforms features to decouple inter-feature correlation using inverse square root covariance decomposition.</p>
                <div class="formula">X_{mah} = (X - \\mu) \\Sigma^{-1/2}</div>
                """,
            ),
        ]
