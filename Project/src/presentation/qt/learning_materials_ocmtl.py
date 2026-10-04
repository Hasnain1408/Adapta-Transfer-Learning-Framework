"""Complete learning material for Orthogonal Core-extraction for Transfer Learning (OCMTL).

This is a self-contained module that provides the full OCMTL educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "OCMTL (Orthogonal Core-extraction for Transfer Learning)".

Based on:
Nahid, Islam, Hossain, Alam, and Shoyaib, "OCMTL: Transfer learning by
orthogonal core-extraction of a matrix and its application in cross-project defect prediction",
Alexandria Engineering Journal, Elsevier, 2025.
"""

from __future__ import annotations 

OCMTL_CONTENT :list [tuple [str ,str ]]=[
(
"OCMTL (Orthogonal Core-extraction for Transfer Learning)",
"""
        <span class="badge">Orthogonal Core-Extraction & Pseudo-Label-Free Transfer</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In high-stakes transfer learning domains such as <strong>Cross-Project Defect Prediction (CPDP)</strong>, software defect data from target projects is initially scarce or completely unannotated. While traditional transfer learning methods (like TCA, JDA, or BDA) have demonstrated success, they face several critical hurdles in practical software engineering pipelines:</p>

        <ul>
            <li><strong>Sensitivity to Hyperparameters:</strong> Many existing methods require tuning complex regularization weights, kernel types, and subspace dimensionalities, where sub-optimal choices degrade performance.</li>
            <li><strong>Dependence on Pseudo-Labels:</strong> Methods relying on iterative target pseudo-labeling are vulnerable to confirmation bias and negative transfer — if early pseudo-labels are incorrect, subsequent distribution alignment reinforces those errors.</li>
            <li><strong>Multicollinearity:</strong> Software engineering metric datasets contain highly collinear features (e.g. lines of code, cyclomatic complexity, Halstead metrics) that destabilize traditional matrix inversions.</li>
        </ul>

        <p>To overcome these challenges, Nahid et al. (<em>Alexandria Engineering Journal</em>, 2025) proposed <strong>Orthogonal Core-extraction of a Matrix (OCM)</strong> and its transfer learning extension: <strong>OCMTL</strong>.</p>

        <div class="card">
        <strong>Central Idea:</strong> Extract shared orthogonal basis cores between source and target domains via alternating SVD projections without requiring any target pseudo-labels, capturing maximum data variance while naturally eliminating multicollinearity.
        </div>

        <!-- ── The Orthogonal Core-Extraction Principle ──────────── -->
        <h2>The Foundation: Orthogonal Core-Extraction (OCM)</h2>
        <p>Before transfer learning, <strong>OCM</strong> is formulated as an iterative, parameter-free matrix decomposition technique. Given a matrix X:</p>

        <div class="formula">
        X = A · C<sup>T</sup> + E
        </div>

        <p>where <strong>A</strong> is an orthogonal basis matrix capturing principal directional vectors, <strong>C</strong> is the core matrix encoding feature loadings, and <strong>E</strong> is residual noise.</p>

        <p>Unlike standard PCA or factor analysis, OCM repeatedly extracts orthogonal rank-one components that achieve the maximum achievable variance along perpendicular axes, effectively stripping out redundant multicollinear relationships without requiring matrix inversion.</p>

        <!-- ── Transfer Learning Extension: OCMTL ────────────────── -->
        <h2>Extending OCM to Transfer Learning (OCMTL)</h2>
        <p>In transfer learning, source data X<sub>S</sub> and target data X<sub>T</sub> share the same feature metrics but differ in distribution. OCMTL identifies a shared projection core <strong>C<sub>t</sub> ∈ ℝ<sup>d × k</sup></strong> that simultaneously aligns both domains into a common subspace.</p>

        <p>Critically, OCMTL avoids pseudo-labels by using <strong>subspace angle minimization</strong> between source and target orthogonal cores.</p>

        <!-- ── Class Weighting Matrix H_sc ───────────────────────── -->
        <h2>Source Class Balancing (H<sub>sc</sub>)</h2>
        <p>Software defect datasets suffer from extreme class imbalance (typically 5–15% defective instances vs 85–95% non-defective). To prevent majority non-defective instances from dominating the learned orthogonal core, OCMTL constructs a diagonal class-weighting matrix <strong>H<sub>sc</sub></strong>:</p>

        <div class="formula">
        (H<sub>sc</sub>)<sub>ii</sub> = 1 / √(N<sub>c</sub>) &nbsp;&nbsp; if sample i belongs to class c
        </div>

        <p>where N<sub>c</sub> is the total number of source samples in class c. This weighting balances the influence of defect and non-defect instances during orthogonal core calculation.</p>

        <!-- ── Alternating Orthogonal Projection Mechanism ──────── -->
        <h2>Alternating Orthogonal SVD Projections</h2>
        <p>OCMTL aligns source and target subspaces through an alternating sequence of Singular Value Decompositions (SVD):</p>

        <ol>
            <li><strong>Initialize:</strong> A random projection core C<sub>t</sub> ∈ ℝ<sup>d × k</sup> (orthonormalized).</li>
            <li><strong>Source Subspace Alignment:</strong>
                <div class="formula">
                A<sub>ss</sub> = X<sub>S</sub> · C<sub>t</sub> &nbsp;→&nbsp; SVD(A<sub>ss</sub>) = U<sub>As</sub> Σ<sub>As</sub> V<sub>As</sub><sup>T</sup><br/><br/>
                C<sub>ss</sub> = X<sub>S</sub><sup>T</sup> · H<sub>sc</sub> · U<sub>As</sub> &nbsp;→&nbsp; SVD(C<sub>ss</sub>) = U<sub>Cs</sub> Σ<sub>Cs</sub> V<sub>Cs</sub><sup>T</sup>
                </div>
                Here U<sub>Cs</sub> represents the class-weighted source feature basis core.
            </li>
            <li><strong>Target Subspace Alignment:</strong>
                <div class="formula">
                A<sub>tt</sub> = X<sub>T</sub> · U<sub>Cs</sub> &nbsp;→&nbsp; SVD(A<sub>tt</sub>) = U<sub>At</sub> Σ<sub>At</sub> V<sub>At</sub><sup>T</sup><br/><br/>
                C<sub>tt</sub> = X<sub>T</sub><sup>T</sup> · U<sub>At</sub> &nbsp;→&nbsp; SVD(C<sub>tt</sub>) = U<sub>Ct</sub> Σ<sub>Ct</sub> V<sub>Ct</sub><sup>T</sup>
                </div>
                Here U<sub>Ct</sub> represents the target feature basis core updated to match source information.
            </li>
            <li><strong>Convergence Check via Subspace Angles:</strong><br/>
            Compute the maximum principal subspace angle between source core U<sub>Cs</sub> and target core U<sub>Ct</sub>:
                <div class="formula">
                θ<sub>max</sub> = max( subspace_angles(U<sub>Cs</sub>, U<sub>Ct</sub>) )
                </div>
            If θ<sub>max</sub> increases or converges, the iteration halts.
            </li>
        </ol>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete OCMTL Algorithm</h2>
        <ol>
            <li><strong>Input:</strong> Source data X<sub>S</sub> ∈ ℝ<sup>n<sub>s</sub> × d</sup> with labels Y<sub>S</sub>, unlabeled target data X<sub>T</sub> ∈ ℝ<sup>n<sub>t</sub> × d</sup>, latent dimension k (default: 2 to 10), max iterations (default: 20).</li>
            <li><strong>Step 1:</strong> Compute source class frequencies and assemble diagonal weighting matrix H<sub>sc</sub>.</li>
            <li><strong>Step 2:</strong> Initialize random transformation core C<sub>t</sub> ∈ ℝ<sup>d × k</sup>.</li>
            <li><strong>Step 3:</strong> Alternating projection loop (it = 1 to max_iterations):
                <ul>
                    <li>Project source data: A<sub>ss</sub> = X<sub>S</sub> C<sub>t</sub>; compute left singular vectors U<sub>As</sub>.</li>
                    <li>Project class-weighted features: C<sub>ss</sub> = X<sub>S</sub><sup>T</sup> H<sub>sc</sub> U<sub>As</sub>; compute left singular vectors U<sub>Cs</sub>.</li>
                    <li>Project target data using source core: A<sub>tt</sub> = X<sub>T</sub> U<sub>Cs</sub>; compute left singular vectors U<sub>At</sub>.</li>
                    <li>Project target features: C<sub>tt</sub> = X<sub>T</sub><sup>T</sup> U<sub>At</sub>; compute left singular vectors U<sub>Ct</sub>.</li>
                    <li>Compute maximum principal angle between U<sub>Cs</sub> and U<sub>Ct</sub>.</li>
                    <li>If angle stops decreasing, terminate and restore the best C<sub>t</sub>.</li>
                </ul>
            </li>
            <li><strong>Step 4:</strong> Project both domains: Z<sub>S</sub> = X<sub>S</sub> C<sub>t</sub>, &nbsp; Z<sub>T</sub> = X<sub>T</sub> C<sub>t</sub>.</li>
            <li><strong>Step 5:</strong> Train standard classifier on (Z<sub>S</sub>, Y<sub>S</sub>) and predict labels on Z<sub>T</sub>.</li>
        </ol>

        <!-- ── Key Advantages of OCMTL ───────────────────────────── -->
        <h2>Key Advantages & Experimental Benchmarks</h2>
        <ul>
            <li><strong>Zero Pseudo-Label Dependence:</strong> Completely avoids generating pseudo-labels for the target domain, making it immune to confirmation bias.</li>
            <li><strong>Inherent Multicollinearity Mitigation:</strong> Orthogonal cores decompose shared variation along perpendicular axes, eliminating redundant feature correlation.</li>
            <li><strong>Class Imbalance Robustness:</strong> Native H<sub>sc</sub> weighting ensures minority positive defect classes are preserved.</li>
            <li><strong>State-of-the-Art Benchmark Results:</strong> On 4 major software engineering benchmarks (NASA, RELINK, SOFTLAB, AEEEM), OCMTL achieved an average <strong>AUC of 71%</strong>, outperforming 11 competing methods (including TCA, TCA+, JDA, and JCSL).</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters</h2>
        <ul>
            <li><strong>Latent Dimension (k):</strong> Number of orthogonal core components (typically 2 to 10). Because OCM captures maximum variance in the top singular vectors, small values of k (e.g. 2 or 5) often yield peak performance.</li>
            <li><strong>Max Iterations:</strong> Typically 10 to 20; usually converges within 5 to 8 alternating steps.</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> OCMTL is an innovative, pseudo-label-free transfer learning technique specifically tailored for tabular and cross-project defect prediction datasets. By extracting shared orthogonal basis cores via alternating SVD projections and balancing class frequencies with H<sub>sc</sub>, it delivers high predictive AUC without the vulnerability of pseudo-label feedback loops.
        </div>
        """,
),
]

__all__ =["OCMTL_CONTENT"]
