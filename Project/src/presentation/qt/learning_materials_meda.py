"""Complete learning material for Manifold Embedded Distribution Alignment (MEDA).

This is a self-contained module that provides the full MEDA educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "MEDA (Manifold Embedded Distribution Alignment)".

Based on:
Wang, Feng, Chen, Yu, Huang, and Yu, "Visual Domain Adaptation with
Manifold Embedded Distribution Alignment", ACM Multimedia (MM), 2018.
"""

from __future__ import annotations 

MEDA_CONTENT :list [tuple [str ,str ]]=[
(
"MEDA (Manifold Embedded Distribution Alignment)",
"""
        <span class="badge">Grassmann Manifold & Dynamic Distribution Alignment</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>Traditional domain adaptation techniques frequently exhibit three fundamental limitations:</p>
        <ol>
            <li><strong>Degenerated Feature Representations:</strong> Projecting complex visual data into arbitrary Euclidean subspaces without respecting the underlying Riemannian manifold geometry can distort intrinsic feature relationships.</li>
            <li><strong>Manual / Equal Distribution Weighting:</strong> Methods like JDA treat marginal and conditional distributions equally, while BDA requires manually tuning the trade-off parameter μ. In real-world data, the relative importance of marginal and conditional shifts is unknown beforehand and varies across datasets.</li>
            <li><strong>Two-Stage Disconnection:</strong> Performing feature projection in one step and then training a standard classifier in a second, independent step often produces sub-optimal classifiers that are not directly optimized for the aligned distribution.</li>
        </ol>

        <p><strong>Manifold Embedded Distribution Alignment (MEDA)</strong>, presented by Wang et al. at ACM Multimedia 2018, addresses all three issues simultaneously in a unified, end-to-end learning framework.</p>

        <div class="card">
        <strong>Central Idea:</strong> Embed source and target data into a Grassmann manifold via Geodesic Flow Kernel (GFK), dynamically estimate the marginal vs. conditional balance factor μ directly from data using proxy 𝒜-distance, and optimize the domain-adapted classifier directly via regularized Structural Risk Minimization in closed form.
        </div>

        <!-- ── The Three Innovations of MEDA ─────────────────────── -->
        <h2>The Three Pillars of MEDA</h2>

        <h3>1. Manifold Feature Learning (GFK Embedding)</h3>
        <p>MEDA first transforms features onto the Grassmann manifold 𝔾(d, D) using the Geodesic Flow Kernel (GFK). By incorporating the continuous geodesic flow of subspaces connecting source and target, GFK provides a domain-invariant geometric foundation that avoids feature distortion.</p>

        <h3>2. Dynamic Distribution Alignment (𝒜-Distance)</h3>
        <p>Instead of guessing or grid-searching the balance factor μ, MEDA <strong>quantifies distribution discrepancy directly from data</strong> using proxy 𝒜-distance. Let d<sub>M</sub> be the marginal 𝒜-distance and d<sub>c</sub> be the conditional 𝒜-distance for class c. MEDA dynamically calculates:</p>

        <div class="formula">
        μ = d<sub>M</sub> / (d<sub>M</sub> + Σ<sub>c=1</sub><sup>C</sup> d<sub>c</sub>)
        </div>

        <p>If the marginal shift is severe relative to class-conditional shifts, μ automatically drops; if the domains share similar feature distributions, μ automatically increases to focus on class boundaries.</p>

        <h3>3. Direct Classifier Learning (Structural Risk Minimization)</h3>
        <p>Rather than learning an intermediate feature projection, MEDA formulates an end-to-end kernel ridge regression that solves for the classification function <strong>f(x)</strong> directly.</p>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Unified Optimization Objective</h2>
        <p>MEDA formulates a regularized Structural Risk Minimization objective for the classification function <em>f</em> in a Reproducing Kernel Hilbert Space (RKHS) ℋ<sub>K</sub>:</p>

        <div class="formula">
        min<sub>f ∈ ℋ<sub>K</sub></sub> &nbsp; Σ<sub>i=1</sub><sup>n<sub>s</sub></sup> ℓ(f(x<sub>i</sub>), y<sub>i</sub>) + η · ‖f‖<sub>K</sub><sup>2</sup> + λ · D̄<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>) + ρ · S<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>)
        </div>

        <p>Where the four terms represent:</p>
        <ul>
            <li><strong>Σ ℓ(f(x<sub>i</sub>), y<sub>i</sub>):</strong> Squared loss on labeled source samples (ensuring high accuracy on the source task).</li>
            <li><strong>η · ‖f‖<sub>K</sub><sup>2</sup>:</strong> RKHS squared norm penalty (controlling classifier model complexity).</li>
            <li><strong>λ · D̄<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>):</strong> Dynamic distribution alignment term:
                <div class="formula">
                D̄<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>) = (1 − μ) · D<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>) + μ · Σ<sub>c=1</sub><sup>C</sup> D<sub>f</sub><sup>(c)</sup>(P<sub>S</sub>, P<sub>T</sub>)
                </div>
            </li>
            <li><strong>ρ · S<sub>f</sub>(P<sub>S</sub>, P<sub>T</sub>):</strong> Manifold regularization term: tr(F<sup>T</sup> L F) using Graph Laplacian L (ensuring smooth predictions across neighboring points).</li>
        </ul>

        <!-- ── Closed-Form Analytical Solution ───────────────────── -->
        <h2>Closed-Form Analytical Solution</h2>
        <p>By the Representer Theorem, the optimal function <em>f</em> can be expanded over kernel evaluations: f(x) = Σ<sub>i=1</sub><sup>n</sup> α<sub>i</sub> K(x<sub>i</sub>, x), or in matrix form F = K A, where A = [α<sub>1</sub>, …, α<sub>n</sub>]<sup>T</sup> ∈ ℝ<sup>n × C</sup>.</p>

        <p>Setting the derivative of the objective with respect to A to zero yields a linear system with a direct <strong>closed-form solution</strong>:</p>

        <div class="formula">
        [ (E + λ M + ρ L) · K + η I ] · A = Ỹ
        </div>

        <p>Where:</p>
        <ul>
            <li><strong>E:</strong> Diagonal indicator matrix with E<sub>ii</sub> = 1 for source samples and 0 for target samples.</li>
            <li><strong>M:</strong> Dynamic MMD matrix M = (1 − μ) M<sub>0</sub> + μ Σ<sub>c</sub> M<sub>c</sub>.</li>
            <li><strong>L:</strong> Graph Laplacian matrix (L = D − W).</li>
            <li><strong>Ỹ:</strong> n × C label matrix with one-hot encoded rows for source samples and zeros for target samples.</li>
            <li><strong>K:</strong> (n<sub>s</sub> + n<sub>t</sub>) × (n<sub>s</sub> + n<sub>t</sub>) kernel matrix.</li>
        </ul>

        <p>The optimal coefficient matrix A is obtained by inverting the regularized system matrix:</p>
        <div class="formula">
        A = [ (E + λ M + ρ L) · K + η I ]<sup>−1</sup> · Ỹ
        </div>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete MEDA Algorithm</h2>
        <ol>
            <li><strong>Input:</strong> Source domain (X<sub>S</sub>, Y<sub>S</sub>), unlabeled target domain X<sub>T</sub>, GFK dimension d, regularization parameters (λ, η, ρ), iterations T.</li>
            <li><strong>Step 1 (Manifold Embedding):</strong> Embed features into the Grassmann manifold via GFK: X̃<sub>S</sub> = X<sub>S</sub> G<sup>1/2</sup>, &nbsp; X̃<sub>T</sub> = X<sub>T</sub> G<sup>1/2</sup>.</li>
            <li><strong>Step 2 (Kernel Matrix):</strong> Construct kernel matrix K over [X̃<sub>S</sub>; X̃<sub>T</sub>] using linear, RBF, or SAM kernel.</li>
            <li><strong>Step 3 (Initial Pseudo-Labels):</strong> Train an initial classifier (e.g. Linear SVM or 1-NN) on source data to generate initial target pseudo-labels Ŷ<sub>T</sub>.</li>
            <li><strong>Step 4: Iterative Adaptation (t = 1 to T):</strong>
                <ul>
                    <li>Compute marginal 𝒜-distance d<sub>M</sub> and conditional 𝒜-distances d<sub>c</sub> for each class.</li>
                    <li>Update the dynamic balance factor: μ = d<sub>M</sub> / (d<sub>M</sub> + Σ<sub>c</sub> d<sub>c</sub>).</li>
                    <li>Construct the dynamic MMD matrix M = (1 − μ) M<sub>0</sub> + μ Σ<sub>c</sub> M<sub>c</sub>.</li>
                    <li>Solve for coefficient matrix A = [ (E + λ M + ρ L) K + η I ]<sup>−1</sup> Ỹ.</li>
                    <li>Compute predictions F = K A; assign updated target pseudo-labels: ŷ<sub>j</sub><sup>t</sup> = argmax<sub>c</sub> F<sub>(n<sub>s</sub>+j), c</sub>.</li>
                </ul>
            </li>
            <li><strong>Output:</strong> Final target class predictions Ŷ<sub>T</sub>.</li>
        </ol>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters</h2>
        <ul>
            <li><strong>Subspace Dimension (d):</strong> Dimension of GFK manifold embedding (typically 20).</li>
            <li><strong>λ (Distribution Alignment Weight):</strong> Regularization weight for the MMD discrepancy term (typically 1.0 to 10.0).</li>
            <li><strong>η (Ridge Regularization):</strong> Numerical stability term ensuring the linear system is strictly invertible (typically 0.1 to 1.0).</li>
            <li><strong>ρ (Manifold Regularization):</strong> Weight of the Graph Laplacian smoothness term (typically 1.0).</li>
            <li><strong>Kernel Type:</strong> 'rbf' with bandwidth γ, 'linear', or 'sam'.</li>
            <li><strong>Iterations (T):</strong> Typically converges rapidly within 5 to 10 iterations.</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> MEDA represents one of the most comprehensive classical domain adaptation frameworks. By unifying Grassmann manifold embedding (GFK), dynamic distribution balancing (via data-driven 𝒜-distance), and direct classifier optimization in a single closed-form system, it eliminates manual parameter guessing while achieving state-of-the-art cross-domain visual recognition accuracy.
        </div>
        """,
),
]

__all__ =["MEDA_CONTENT"]
