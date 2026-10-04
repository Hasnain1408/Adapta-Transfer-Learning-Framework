"""Complete learning material for Visual Domain Adaptation through Locality Information (DALI).

This is a self-contained module that provides the full DALI educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "DALI (Domain Adaptation through Locality Information)".

Based on:
Devika, Sanodiya, Jose, and Mathew, "Visual Domain Adaptation through Locality Information",
Engineering Applications of Artificial Intelligence, Elsevier, 2023.
"""

from __future__ import annotations 

DALI_CONTENT :list [tuple [str ,str ]]=[
(
"DALI (Domain Adaptation through Locality Information)",
"""
        <span class="badge">Manifold Locality & Distribution Alignment</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In unsupervised domain adaptation, methods like TCA and JDA reduce cross-domain distribution divergence by minimizing Maximum Mean Discrepancy (MMD). However, statistical distribution matching alone has a significant blind spot:</p>

        <p><strong>It ignores the local geometric structure and intrinsic manifold geometry of the data.</strong></p>

        <p>When high-dimensional data is projected purely to minimize statistical means across domains, nearby points on the data manifold can be torn apart, destroying discriminative neighborhood relationships. Furthermore, existing methods often neglect target domain variance and class cluster compactness.</p>

        <p><strong>Visual Domain Adaptation through Locality Information (DALI)</strong> (published in <em>Engineering Applications of Artificial Intelligence</em>, 2023) is a modern transfer learning framework that solves these challenges by combining:</p>
        <ol>
            <li><strong>Marginal & Conditional Distribution Alignment</strong> via MMD;</li>
            <li><strong>Locality Preservation</strong> via Graph Laplacian embedding (LPP); and</li>
            <li><strong>Target Discriminative Scatter Optimization</strong> (maximizing between-class scatter while minimizing within-class scatter).</li>
        </ol>

        <div class="card">
        <strong>Central Idea:</strong> Bridge domain shift by simultaneously aligning marginal and conditional distributions while strictly preserving local manifold geometry and maximizing target class separability.
        </div>

        <!-- ── Three Core Pillars of DALI ───────────────────────── -->
        <h2>The Three Pillars of DALI</h2>

        <h3>1. Distribution Alignment (MMD)</h3>
        <p>DALI matches both marginal distributions P(X) and conditional distributions P(Y | X) using the combined MMD matrix M = M<sub>0</sub> + N, where M<sub>0</sub> aligns global feature distributions and N aligns class-specific distributions using target pseudo-labels.</p>

        <h3>2. Locality Preserving Projection (LPP)</h3>
        <p>Real-world visual features live on non-linear sub-manifolds. DALI constructs a k-nearest neighbor affinity graph <strong>W</strong> over the feature space and builds the Graph Laplacian <strong>L = D − W</strong>. By minimizing tr(A<sup>T</sup> X L X<sup>T</sup> A), points that are close in the original manifold are forced to stay close in the adapted projection space.</p>

        <h3>3. Target Class Scatter Optimization</h3>
        <p>To ensure that target representations are not only aligned but also highly discriminative for classification, DALI utilizes target pseudo-labels to model:</p>
        <ul>
            <li><strong>Within-Class Scatter (S<sub>w</sub>):</strong> Measures the compactness of samples belonging to the same class. DALI minimizes S<sub>w</sub> so classes form tight clusters.</li>
            <li><strong>Between-Class Scatter (S<sub>b</sub>):</strong> Measures the distance between distinct class centers. DALI maximizes S<sub>b</sub> so different classes remain well-separated.</li>
        </ul>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation & Objective</h2>
        <p>Let X = [X<sub>S</sub>, X<sub>T</sub>] be the combined data matrix. DALI formulates a unified optimization problem for the projection matrix <strong>A</strong>:</p>

        <div class="formula">
        min<sub>A</sub> &nbsp; tr(A<sup>T</sup> [ X M X<sup>T</sup> + β · S<sub>w</sub> + δ · L<sub>feat</sub> ] A) + λ · ‖A‖<sub>F</sub><sup>2</sup><br/><br/>
        subject to: &nbsp; A<sup>T</sup> [ X<sub>T</sub> H<sub>T</sub> X<sub>T</sub><sup>T</sup> + ε · S<sub>b</sub> ] A = I
        </div>

        <p>Where:</p>
        <ul>
            <li><strong>X M X<sup>T</sup>:</strong> MMD distribution discrepancy term (marginal + conditional).</li>
            <li><strong>S<sub>w</sub>:</strong> Target within-class scatter matrix (penalizing intra-class spread).</li>
            <li><strong>L<sub>feat</sub>:</strong> Locality preserving Graph Laplacian matrix (preserving manifold neighborhood structure).</li>
            <li><strong>X<sub>T</sub> H<sub>T</sub> X<sub>T</sub><sup>T</sup>:</strong> Target domain variance preservation term (preventing projection collapse).</li>
            <li><strong>S<sub>b</sub>:</strong> Target between-class scatter matrix (enforcing wide inter-class separation).</li>
            <li><strong>β, δ, ε, λ:</strong> Hyperparameter trade-off coefficients.</li>
        </ul>

        <!-- ── Solution via Generalized Eigenvalue Problem ───────── -->
        <h2>Eigenvalue Decomposition Solution</h2>
        <p>Let the numerator matrix be Q and the denominator constraint matrix be P:</p>
        <div class="formula">
        Q = X M X<sup>T</sup> + β S<sub>w</sub> + δ L<sub>feat</sub> + λ I<br/><br/>
        P = X<sub>T</sub> H<sub>T</sub> X<sub>T</sub><sup>T</sup> + ε S<sub>b</sub>
        </div>

        <p>The optimization reduces to the generalized eigenvalue problem:</p>
        <div class="formula">
        Q · A = P · A · Φ
        </div>

        <p>The optimal transformation matrix A is obtained by extracting the <em>k</em> eigenvectors corresponding to the smallest eigenvalues of P<sup>−1</sup> Q (or leading eigenvectors of Q<sup>−1</sup> P).</p>

        <!-- ── Iterative Pseudo-Labeling ─────────────────────────── -->
        <h2>Iterative Refinement Procedure</h2>
        <p>Because true target labels are unavailable, target conditional distributions and scatter matrices (S<sub>w</sub>, S<sub>b</sub>) rely on <strong>pseudo-labels</strong>. DALI alternates between:</p>

        <div class="card">
        <strong>Iterative Loop:</strong><br/>
        1. Current pseudo-labels → Compute conditional MMD (N) and target scatters (S<sub>w</sub>, S<sub>b</sub>)<br/>
        2. Construct Q and P matrices → Solve generalized eigenproblem for A<br/>
        3. Project data Z<sub>S</sub> = A<sup>T</sup> X<sub>S</sub>, Z<sub>T</sub> = A<sup>T</sup> X<sub>T</sub><br/>
        4. Retrain base classifier on (Z<sub>S</sub>, Y<sub>S</sub>) → Predict updated pseudo-labels for Z<sub>T</sub><br/>
        5. Repeat until target labels stabilize or maximum iterations reached
        </div>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete DALI Algorithm</h2>
        <ol>
            <li><strong>Input:</strong> Source data (X<sub>S</sub>, Y<sub>S</sub>), unlabeled target data X<sub>T</sub>, latent dimension k, hyperparameters (β, δ, ε, λ), iterations T.</li>
            <li><strong>Step 1:</strong> L2-normalize feature rows to standardize scale.</li>
            <li><strong>Step 2:</strong> Compute target centering matrix H<sub>T</sub> and target covariance S<sub>T</sub> = X<sub>T</sub> H<sub>T</sub> X<sub>T</sub><sup>T</sup>.</li>
            <li><strong>Step 3:</strong> Build nearest-neighbor graph W and compute Graph Laplacian L = D − W over the feature manifold.</li>
            <li><strong>Step 4:</strong> Train initial 1-NN classifier on source features and generate initial target pseudo-labels.</li>
            <li><strong>Step 5 (Iteration t = 1 to T):</strong>
                <ul>
                    <li>Compute within-class scatter S<sub>w</sub> and between-class scatter S<sub>b</sub> from current target pseudo-labels.</li>
                    <li>Construct marginal M<sub>0</sub> and conditional N discrepancy matrices; form M = M<sub>0</sub> + N.</li>
                    <li>Assemble system matrices Q and P.</li>
                    <li>Solve the generalized eigenvalue problem Q A = P A Φ.</li>
                    <li>Select leading k eigenvectors to form projection matrix A.</li>
                    <li>Transform features: Z<sub>S</sub> = A<sup>T</sup> X<sub>S</sub>, Z<sub>T</sub> = A<sup>T</sup> X<sub>T</sub>.</li>
                    <li>Update target pseudo-labels using 1-NN on (Z<sub>S</sub>, Y<sub>S</sub>).</li>
                </ul>
            </li>
            <li><strong>Output:</strong> Adapted representations Z<sub>S</sub>, Z<sub>T</sub> and target predictions.</li>
        </ol>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters</h2>
        <ul>
            <li><strong>Latent Dimension (k):</strong> Dimension of the projected feature subspace (typically 10 to 50).</li>
            <li><strong>β (Within-Class Scatter Weight):</strong> Controls compactness of target class clusters (default: 0.4).</li>
            <li><strong>δ (Locality Weight):</strong> Controls importance of manifold neighborhood preservation (default: 0.2).</li>
            <li><strong>ε (Between-Class Scatter Weight):</strong> Controls separation between distinct target classes (default: 0.1).</li>
            <li><strong>Regularization (λ):</strong> Numerical stability term (default: 1.0).</li>
            <li><strong>Iterations (T):</strong> Number of pseudo-label refinement cycles (typically 10).</li>
        </ul>

        <!-- ── When to Use DALI ──────────────────────────────────── -->
        <h2>When to Use DALI</h2>
        <ul>
            <li>On complex image, face, or object recognition datasets (e.g. Office-Caltech, Office-Home, PIE) where data lies on complex non-linear visual manifolds.</li>
            <li>When preserving local neighborhood relationships is crucial to prevent class boundary blurring.</li>
            <li>When target classes require strong discrimination and high intra-class compactness.</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> DALI elevates domain adaptation by uniting statistical MMD alignment with manifold locality preservation (LPP) and target scatter optimization (S<sub>w</sub>, S<sub>b</sub>). It ensures that while domains are aligned, local geometry is respected and target classes remain sharply separated.
        </div>
        """,
),
]

__all__ =["DALI_CONTENT"]
