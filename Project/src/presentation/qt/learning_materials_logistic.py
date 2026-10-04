"""Complete learning material for Logistic Regression classifier.

This is a self-contained module that provides the full Logistic Regression
educational content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "Logistic Regression".
"""

from __future__ import annotations 

LOGISTIC_CONTENT :list [tuple [str ,str ]]=[
(
"Logistic Regression",
"""
        <span class="badge">Linear Probabilistic Classifier</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>Despite its name containing the word "regression", <strong>Logistic Regression</strong> is a foundational supervised linear classification algorithm. It models the conditional probability that a given input sample belongs to a particular class, mapping continuous linear combinations of input features into calibrated probability estimates in the range [0, 1].</p>

        <p>In domain adaptation, Logistic Regression serves as an essential benchmark. When domain adaptation techniques (such as CORAL, TCA, or OCMTL) succeed in projecting cross-domain data into a linearly separable space, Logistic Regression offers lightning-fast training, high numerical stability, and probabilistic outputs ideal for ROC-AUC scoring.</p>

        <div class="card">
        <strong>Central Principle:</strong> Model the log-odds of class membership as a linear function of the input features, converting linear scores into probability estimates via the non-linear Sigmoid (logistic) function.
        </div>

        <!-- ── Mathematical Formulation: Binary Case ────────────── -->
        <h2>Mathematical Formulation (Binary Classification)</h2>
        <p>Given an input feature vector x ∈ ℝ<sup>d</sup>, a linear model computes the score:</p>
        <div class="formula">
        z = w<sup>T</sup> x + b = w<sub>1</sub> x<sub>1</sub> + w<sub>2</sub> x<sub>2</sub> + … + w<sub>d</sub> x<sub>d</sub> + b
        </div>
        <p>where <strong>w ∈ ℝ<sup>d</sup></strong> is the weight vector and <strong>b ∈ ℝ</strong> is the bias intercept.</p>

        <p>To convert this unrestricted linear value z ∈ (−∞, +∞) into a valid probability P ∈ (0, 1), Logistic Regression applies the <strong>Sigmoid (logistic) function</strong>:</p>

        <div class="formula">
        P(Y = 1 | X = x) = σ(z) = 1 / ( 1 + e<sup>−z</sup> ) = 1 / ( 1 + e<sup>−(w<sup>T</sup> x + b)</sup> )
        </div>

        <p>Complementarily, the probability of the negative class is:</p>
        <div class="formula">
        P(Y = 0 | X = x) = 1 − P(Y = 1 | X = x) = e<sup>−z</sup> / ( 1 + e<sup>−z</sup> )
        </div>

        <!-- ── The Logit & Odds Ratio ────────────────────────────── -->
        <h2>The Odds Ratio and Logit Link</h2>
        <p>The <strong>odds</strong> of an event is the ratio of probability of occurrence to the probability of non-occurrence:</p>
        <div class="formula">
        Odds = P / (1 − P) = e<sup>w<sup>T</sup> x + b</sup>
        </div>

        <p>Taking the natural logarithm yields the <strong>logit function (log-odds)</strong>, revealing the linear core of the model:</p>
        <div class="formula">
        logit(P) = ln( P / (1 − P) ) = w<sup>T</sup> x + b
        </div>

        <div class="card">
        <strong>Decision Boundary:</strong> Classification assigns class 1 if P ≥ 0.5, which corresponds to z ≥ 0 (w<sup>T</sup> x + b ≥ 0). Thus, the decision boundary is a flat <strong>hyperplane</strong> separating the feature space into two half-spaces.
        </div>

        <!-- ── Loss Function: Binary Cross-Entropy ───────────────── -->
        <h2>Loss Function: Binary Cross-Entropy (Log Loss)</h2>
        <p>Because ordinary squared error produces a non-convex loss surface when combined with the sigmoid function, Logistic Regression is trained by <strong>Maximum Likelihood Estimation (MLE)</strong>. Minimizing the negative log-likelihood over <em>n</em> training samples yields the <strong>Log Loss</strong> objective with L2 regularization:</p>

        <div class="formula">
        J(w, b) = −(1/n) Σ<sub>i=1</sub><sup>n</sup> [ y<sub>i</sub> ln(p<sub>i</sub>) + (1 − y<sub>i</sub>) ln(1 − p<sub>i</sub>) ] + (1 / (2C)) ‖w‖<sub>2</sub><sup>2</sup>
        </div>

        <p>Where:</p>
        <ul>
            <li><strong>y<sub>i</sub> ∈ {0, 1}:</strong> True class label.</li>
            <li><strong>p<sub>i</sub> = σ(w<sup>T</sup> x<sub>i</sub> + b):</strong> Predicted probability for instance <em>i</em>.</li>
            <li><strong>C:</strong> Inverse regularization strength (smaller C enforces stronger regularization, penalizing large weights).</li>
        </ul>

        <!-- ── Multi-Class Logistic Regression (Softmax) ─────────── -->
        <h2>Multi-Class Classification: Multinomial Softmax</h2>
        <p>For datasets with C &gt; 2 classes, Logistic Regression generalizes via the <strong>Softmax function</strong> (also known as Multinomial Logistic Regression):</p>

        <div class="formula">
        P(Y = c | X = x) = e<sup>w<sub>c</sub><sup>T</sup> x + b<sub>c</sub></sup> / ( Σ<sub>j=1</sub><sup>C</sup> e<sup>w<sub>j</sub><sup>T</sup> x + b<sub>j</sub></sup> )
        </div>

        <p>Each class maintains its own weight vector w<sub>c</sub>. The denominator normalizes all exponential scores so they sum strictly to 1.0, forming a valid categorical probability distribution.</p>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Domain Adaptation</h2>
        <ul>
            <li><strong>Calibrated Probabilities for ROC-AUC:</strong> Unlike hard-boundary classifiers, Logistic Regression natively produces continuous prediction probabilities, enabling exact calculation of Receiver Operating Characteristic Area Under the Curve (ROC-AUC).</li>
            <li><strong>Verifying Linear Separability:</strong> When domain adaptation methods align feature distributions (such as TCA with linear kernel, or CORAL), Logistic Regression tests whether the adapted features have achieved linear separability between classes.</li>
            <li><strong>Low Overfitting Risk:</strong> With its linear hypothesis space and L2 regularization penalty, it is less prone to overfitting on limited source training datasets than deep architectures or deep decision trees.</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters & Optimization</h2>
        <ul>
            <li><strong>C (Inverse Regularization):</strong> Default C = 1.0. Smaller C increases shrinkage on weights, avoiding overfitting; larger C fits source data more aggressively.</li>
            <li><strong>max_iter:</strong> Maximum iterations for solver convergence. In this studio's engine, <strong>max_iter = 1000</strong> ensures complete convergence across all datasets.</li>
            <li><strong>Solvers:</strong> L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is the standard quasi-Newton solver used for multinomial problems; LibLinear is suited for smaller L1-penalized problems.</li>
            <li><strong>random_state:</strong> Ensures deterministic, reproducible benchmark runs (set to 42 in this application).</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Extremely fast training and microsecond-level inference speed.</li>
            <li>Highly interpretable: weight coefficients directly indicate the positive or negative contribution of each feature to the log-odds of a class.</li>
            <li>Outputs well-calibrated prediction probabilities.</li>
            <li>Robust against small amounts of noise when regularized.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Strict assumption of linear decision boundaries in the input space; cannot model complex non-linear XOR relationships without kernel projections or feature engineering.</li>
            <li>Prone to degradation when severe multicollinearity exists unless L2/L1 regularization is tuned.</li>
        </ul>
        </div>

        <!-- ── Implementation in This Application ────────────────── -->
        <h2>Implementation in This Studio</h2>
        <p>In <code>src/application/classification.py</code>, Logistic Regression is configured as:</p>
        <div class="formula">
        LogisticRegression(max_iter=1000, random_state=42)
        </div>
        <p>It outputs both categorical predictions and continuous decision probabilities utilized for AUC computation across adapted source-to-target runs.</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> Logistic Regression is the industry workhorse of linear probabilistic classification. In transfer learning studio experiments, it pairs exceptionally well with linear subspace methods (CORAL, linear TCA, OCMTL) by providing transparent, calibrated class probabilities and competitive ROC-AUC performance.
        </div>
        """,
),
]

__all__ =["LOGISTIC_CONTENT"]
