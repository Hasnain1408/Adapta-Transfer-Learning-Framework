"""Complete learning material for Correlation Alignment (CORAL).

This is a self-contained module that provides the full CORAL educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "CORAL (Correlation Alignment)".

Based on:
Sun, Feng, and Saenko, "Correlation Alignment for Unsupervised Domain Adaptation",
Domain Adaptation in Computer Vision Applications, Springer, 2017.
"""

from __future__ import annotations 

CORAL_CONTENT :list [tuple [str ,str ]]=[
(
"CORAL (Correlation Alignment)",
"""
        <span class="badge">Second-Order Statistic Alignment</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>Most domain adaptation techniques seek to reduce the divergence between domains by projecting data into lower-dimensional subspaces or high-dimensional kernel Hilbert spaces (e.g. TCA, GFK). While effective, these methods frequently require tuning subspace dimensions, choosing kernel bandwidths, or solving computationally demanding optimization problems.</p>

        <p><strong>CORrelation ALignment (CORAL)</strong> introduces an elegant, "frustratingly easy" alternative for unsupervised domain adaptation. Instead of learning projection subspaces, CORAL aligns the <strong>second-order statistics (covariance matrices)</strong> of the source and target feature distributions directly in the original feature space.</p>

        <div class="card">
        <strong>Central Idea:</strong> Minimize domain shift by applying a linear transformation that "whitens" source features and "re-colors" them with target domain covariances, matching second-order feature correlations without altering the dimensionality of the feature space.
        </div>

        <!-- ── The Intuition of Feature Correlation ──────────────── -->
        <h2>Intuition: Why Align Covariance?</h2>
        <p>Features in any domain are not independent; they exhibit distinct correlations that characterize the domain's structure. For example, in an image dataset of office objects, the presence of a "keyboard" feature is strongly correlated with a "monitor" feature.</p>

        <p>When domain shift occurs (due to changes in lighting, background, or camera angle), these feature correlations become distorted:</p>
        <ul>
            <li>The means of features shift (first-order statistics).</li>
            <li>The inter-feature covariances diverge (second-order statistics).</li>
        </ul>

        <p>By standardizing feature means to zero and aligning covariance matrices, the geometric shape and orientation of the source feature cloud are reshaped to match the target feature cloud.</p>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>Let source training data be X<sub>S</sub> ∈ ℝ<sup>n<sub>s</sub> × d</sup> and unlabeled target data be X<sub>T</sub> ∈ ℝ<sup>n<sub>t</sub> × d</sup>, both normalized to zero mean.</p>

        <p>The sample covariance matrices of source and target domains are:</p>
        <div class="formula">
        C<sub>S</sub> = cov(X<sub>S</sub>) = (1 / (n<sub>s</sub> − 1)) · X<sub>S</sub><sup>T</sup> X<sub>S</sub> + I<sub>d</sub><br/><br/>
        C<sub>T</sub> = cov(X<sub>T</sub>) = (1 / (n<sub>t</sub> − 1)) · X<sub>T</sub><sup>T</sup> X<sub>T</sub> + I<sub>d</sub>
        </div>
        <p>where the identity matrix I<sub>d</sub> acts as a small regularization term ensuring both matrices are strictly positive definite and full rank.</p>

        <!-- ── Objective & Closed-Form Solution ──────────────────── -->
        <h2>Optimization Objective & Closed-Form Solution</h2>
        <p>CORAL seeks a linear transformation matrix <strong>A ∈ ℝ<sup>d × d</sup></strong> applied to source features such that the squared Frobenius norm distance between the transformed source covariance and target covariance is minimized:</p>

        <div class="formula">
        min<sub>A</sub> &nbsp; ‖ C<sub>S</sub><sup>aligned</sup> − C<sub>T</sub> ‖<sub>F</sub><sup>2</sup> = ‖ A<sup>T</sup> C<sub>S</sub> A − C<sub>T</sub> ‖<sub>F</sub><sup>2</sup>
        </div>

        <p>Remarkably, this objective has an exact, analytical closed-form solution:</p>

        <div class="formula">
        A = C<sub>S</sub><sup>−1/2</sup> · C<sub>T</sub><sup>1/2</sup>
        </div>

        <p>The adapted source data is obtained simply by matrix multiplication:</p>
        <div class="formula">
        X<sub>S</sub><sup>aligned</sup> = X<sub>S</sub> · A = X<sub>S</sub> · C<sub>S</sub><sup>−1/2</sup> · C<sub>T</sub><sup>1/2</sup>
        </div>

        <!-- ── The Two-Step Whitening and Re-Coloring Process ────── -->
        <h2>The Whitening and Re-Coloring Interpretation</h2>
        <p>The closed-form transformation A can be intuitively decomposed into two geometric operations:</p>

        <ol>
            <li><strong>Whitening the source data (C<sub>S</sub><sup>−1/2</sup>):</strong><br/>
            Multiplying X<sub>S</sub> by C<sub>S</sub><sup>−1/2</sup> decorrelates the source features and scales them so that their covariance becomes the identity matrix I. The data cloud becomes spherical (isotropic).</li>
            <li><strong>Re-coloring with target covariance (C<sub>T</sub><sup>1/2</sup>):</strong><br/>
            Multiplying the whitened source features by C<sub>T</sub><sup>1/2</sup> stretches and rotates the spherical data cloud to adopt the exact correlation structure and orientation of the target domain.</li>
        </ol>

        <div class="card">
        <strong>Geometric Progression:</strong><br/>
        Source Data (Covariance C<sub>S</sub>) → [Whitening × C<sub>S</sub><sup>−1/2</sup>] → Spherical Data (Covariance I) → [Re-coloring × C<sub>T</sub><sup>1/2</sup>] → Aligned Source Data (Covariance C<sub>T</sub>)
        </div>

        <!-- ── Computing Matrix Square Roots ─────────────────────── -->
        <h2>Computing Matrix Square Roots</h2>
        <p>Because covariance matrices are symmetric and positive semi-definite, their fractional powers (C<sup>1/2</sup> and C<sup>−1/2</sup>) are efficiently computed via Singular Value Decomposition (SVD) or Eigendecomposition:</p>

        <div class="formula">
        C = U · Σ · U<sup>T</sup><br/><br/>
        C<sup>1/2</sup> = U · Σ<sup>1/2</sup> · U<sup>T</sup><br/><br/>
        C<sup>−1/2</sup> = U · Σ<sup>−1/2</sup> · U<sup>T</sup>
        </div>

        <p>Here Σ is the diagonal matrix of eigenvalues, and fractional powers are applied element-wise to the diagonal values.</p>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete CORAL Algorithm</h2>
        <ol>
            <li><strong>Step 1 (Centering):</strong> Compute domain means μ<sub>S</sub> and μ<sub>T</sub>, and center features so that both domains have zero mean:
            X̃<sub>S</sub> = X<sub>S</sub> − μ<sub>S</sub>, &nbsp; X̃<sub>T</sub> = X<sub>T</sub> − μ<sub>T</sub>.</li>
            <li><strong>Step 2 (Covariance Computation):</strong> Compute covariance matrices C<sub>S</sub> and C<sub>T</sub> with regularization:
            C<sub>S</sub> = (1/(n<sub>s</sub>−1)) X̃<sub>S</sub><sup>T</sup> X̃<sub>S</sub> + λ I, &nbsp;
            C<sub>T</sub> = (1/(n<sub>t</sub>−1)) X̃<sub>T</sub><sup>T</sup> X̃<sub>T</sub> + λ I.</li>
            <li><strong>Step 3 (Eigendecomposition):</strong> Decompose C<sub>S</sub> and C<sub>T</sub> to compute C<sub>S</sub><sup>−1/2</sup> and C<sub>T</sub><sup>1/2</sup>.</li>
            <li><strong>Step 4 (Transformation):</strong> Compute aligned source features:
            X<sub>S</sub><sup>aligned</sup> = X̃<sub>S</sub> · C<sub>S</sub><sup>−1/2</sup> · C<sub>T</sub><sup>1/2</sup>.</li>
            <li><strong>Step 5 (Classifier Training & Evaluation):</strong> Train standard classifier (e.g. SVM, Logistic Regression, k-NN) on (X<sub>S</sub><sup>aligned</sup>, Y<sub>S</sub>) and directly evaluate on centered target features X̃<sub>T</sub>.</li>
        </ol>

        <!-- ── Key Strengths & Highlights ────────────────────────── -->
        <h2>Key Strengths & Practical Advantages</h2>
        <ul>
            <li><strong>Parameter-Free:</strong> No latent subspace dimension k to tune; no kernel bandwidth to search; no iteration count to stop.</li>
            <li><strong>Analytical & Fast:</strong> Requires only two d × d matrix decompositions. Runs in fractions of a second even on thousands of samples.</li>
            <li><strong>Fully Unsupervised:</strong> Operates entirely without target labels or pseudo-labels, avoiding error propagation.</li>
            <li><strong>Preserves Feature Dimensions:</strong> Maintains the original d-dimensional feature space, preserving interpretability.</li>
            <li><strong>Extensible:</strong> Easily extended to linear classifiers (CORAL-LDA) or deep neural networks (Deep CORAL loss).</li>
        </ul>

        <!-- ── When to Use CORAL ─────────────────────────────────── -->
        <h2>When to Use CORAL</h2>
        <ul>
            <li>As the first, essential domain adaptation baseline for any cross-domain dataset.</li>
            <li>When feature dimensions d are small to moderate (d &lt; 5,000) where d × d covariance matrices can be inverted quickly.</li>
            <li>When domain shift is predominantly linear or second-order in nature.</li>
            <li>When real-time adaptation speed is critical.</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> CORAL is a remarkably simple yet powerful unsupervised domain adaptation method. By whitening the source data and re-coloring it with target second-order statistics via X<sub>S</sub> · C<sub>S</sub><sup>−1/2</sup> · C<sub>T</sub><sup>1/2</sup>, it aligns feature correlations in a single closed-form matrix operation.
        </div>
        """,
),
]

__all__ =["CORAL_CONTENT"]
