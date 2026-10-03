"""Static reference content shown in the Learning Materials dialog.

Each constant is a list of ``(topic title, HTML body)`` pairs — plain data
with no Qt dependency, kept apart from ``learning_materials_dialog.py`` so
that file is just widget/rendering logic and this one is just content to
edit when a method, classifier, metric, or preprocessing option changes.
"""

from __future__ import annotations

TL_METHODS_CONTENT: list[tuple[str, str]] = [
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

CLASSIFIERS_CONTENT: list[tuple[str, str]] = [
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

METRICS_CONTENT: list[tuple[str, str]] = [
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

PREPROCESSING_CONTENT: list[tuple[str, str]] = [
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

__all__ = [
    "CLASSIFIERS_CONTENT",
    "METRICS_CONTENT",
    "PREPROCESSING_CONTENT",
    "TL_METHODS_CONTENT",
]
