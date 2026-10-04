"""Complete learning material for Geodesic Flow Kernel (GFK).

This is a self-contained module that provides the full GFK educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "GFK (Geodesic Flow Kernel)".

Based on:
Gong, Shi, Sha, and Grauman, "Geodesic Flow Kernel for Unsupervised Domain Adaptation",
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
"""

from __future__ import annotations 

GFK_CONTENT :list [tuple [str ,str ]]=[
(
"GFK (Geodesic Flow Kernel)",
"""
        <span class="badge">Grassmann Manifold & Continuous Subspace Interpolation</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In cross-domain visual recognition and transfer learning, source and target data frequently occupy different low-dimensional linear subspaces due to differences in camera viewpoint, lighting, or background conditions.</p>

        <p>Earlier subspace-based methods attempted to bridge this gap by choosing a single intermediate subspace or learning a direct projection from source to target. However, selecting a single intermediate subspace is inherently arbitrary and inevitably loses valuable information along the transition path.</p>

        <p><strong>Geodesic Flow Kernel (GFK)</strong>, introduced by Gong et al. at CVPR 2012, proposes a radically elegant solution:</p>

        <div class="card">
        <strong>Central Idea:</strong> Model domain adaptation by constructing a continuous geodesic path on the <strong>Grassmann manifold</strong> between the source and target subspaces, and integrate over an <strong>infinite number of intermediate subspaces</strong> connecting the two domains in closed form.
        </div>

        <!-- ── The Grassmann Manifold Perspective ────────────────── -->
        <h2>The Grassmann Manifold Formulation</h2>
        <p>A <strong>Grassmann manifold</strong> 𝔾(d, D) is the mathematical space of all <em>d</em>-dimensional linear subspaces of ℝ<sup>D</sup>. On this manifold, an entire subspace is represented as a single point.</p>

        <p>Let P<sub>S</sub> ∈ ℝ<sup>D × d</sup> and P<sub>T</sub> ∈ ℝ<sup>D × d</sup> be orthonormal bases of the source and target subspaces (typically extracted via PCA). On the Grassmannian manifold:</p>
        <ul>
            <li>Source domain is a point: <strong>S<sub>S</sub> ∈ 𝔾(d, D)</strong></li>
            <li>Target domain is a point: <strong>S<sub>T</sub> ∈ 𝔾(d, D)</strong></li>
        </ul>

        <p>The shortest path connecting two points on a curved manifold is called a <strong>geodesic</strong>. The geodesic flow between S<sub>S</sub> and S<sub>T</sub> forms a continuous trajectory of subspaces:</p>
        <div class="formula">
        Φ(t) ∈ 𝔾(d, D), &nbsp;&nbsp; with &nbsp; t ∈ [0, 1]
        </div>
        <p>where Φ(0) = P<sub>S</sub> (pure source subspace) and Φ(1) = P<sub>T</sub> (pure target subspace). As <em>t</em> moves from 0 to 1, Φ(t) smoothly traverses all intermediate representations.</p>

        <div class="card">
        <strong>Continuous Subspace Trajectory:</strong><br/>
        Source Subspace Φ(0) ───→ Intermediate Subspaces Φ(t) ───→ Target Subspace Φ(1)
        </div>

        <!-- ── Integrating Infinite Subspaces: The GFK Kernel ───── -->
        <h2>The Geodesic Flow Kernel (G)</h2>
        <p>Rather than choosing a fixed discrete <em>t</em>, GFK projects any sample <strong>x</strong> into all subspaces Φ(t) simultaneously and integrates the inner product over the entire continuous path:</p>

        <div class="formula">
        k<sub>GFK</sub>(x<sub>i</sub>, x<sub>j</sub>) = ∫<sub>0</sub><sup>1</sup> (Φ(t)<sup>T</sup> x<sub>i</sub>)<sup>T</sup> (Φ(t)<sup>T</sup> x<sub>j</sub>) dt = x<sub>i</sub><sup>T</sup> [ ∫<sub>0</sub><sup>1</sup> Φ(t) Φ(t)<sup>T</sup> dt ] x<sub>j</sub>
        </div>

        <p>The integral over continuous subspaces defines the <strong>Geodesic Flow Kernel matrix G ∈ ℝ<sup>D × D</sup></strong>:</p>

        <div class="formula">
        G = ∫<sub>0</sub><sup>1</sup> Φ(t) Φ(t)<sup>T</sup> dt<br/><br/>
        k<sub>GFK</sub>(x<sub>i</sub>, x<sub>j</sub>) = x<sub>i</sub><sup>T</sup> G x<sub>j</sub>
        </div>

        <p>The matrix <strong>G</strong> is symmetric and positive semi-definite (PSD). Projecting any vector <strong>x</strong> through the matrix square root <strong>G<sup>1/2</sup></strong> yields a new adapted feature vector <strong>z = G<sup>1/2</sup> x</strong> such that:</p>
        <div class="formula">
        z<sub>i</sub><sup>T</sup> z<sub>j</sub> = (G<sup>1/2</sup> x<sub>i</sub>)<sup>T</sup> (G<sup>1/2</sup> x<sub>j</sub>) = x<sub>i</sub><sup>T</sup> G x<sub>j</sub>
        </div>

        <!-- ── Closed-Form Computation via Principal Angles ──────── -->
        <h2>Closed-Form Computation via Principal Angles</h2>
        <p>The continuous integral ∫<sub>0</sub><sup>1</sup> Φ(t) Φ(t)<sup>T</sup> dt has a closed-form analytical solution computed via the <strong>principal angles</strong> between source and target subspaces:</p>

        <ol>
            <li>Compute the SVD of the cross-subspace projection P<sub>S</sub><sup>T</sup> P<sub>T</sub> = U<sub>1</sub> Γ V<sup>T</sup>.</li>
            <li>The diagonal elements of Γ are the cosines of principal angles: cos(θ<sub>i</sub>) = γ<sub>i</sub>.</li>
            <li>Let principal angles be θ = [θ<sub>1</sub>, …, θ<sub>d</sub>]. The geodesic flow Φ(t) can be decomposed into trigonometric functions of θ.</li>
            <li>Integrating sin(2θ t) and cos(2θ t) over t ∈ [0, 1] yields simple analytical diagonal blocks:
                <div class="formula">
                Λ<sub>11</sub> = diag(0.5 · [1 + sin(2θ) / (2θ)])<br/>
                Λ<sub>12</sub> = diag(0.5 · [cos(2θ) − 1] / (2θ))<br/>
                Λ<sub>22</sub> = diag(0.5 · [1 − sin(2θ) / (2θ)])
                </div>
            </li>
            <li>Assembling these blocks into an orthogonal basis produces the exact matrix <strong>G</strong> in closed form.</li>
        </ol>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete GFK Algorithm</h2>
        <ol>
            <li><strong>Input:</strong> Source samples X<sub>S</sub> ∈ ℝ<sup>n<sub>s</sub> × D</sup>, Target samples X<sub>T</sub> ∈ ℝ<sup>n<sub>t</sub> × D</sup>, Subspace dimension d.</li>
            <li><strong>Step 1:</strong> Center X<sub>S</sub> and X<sub>T</sub> to zero mean.</li>
            <li><strong>Step 2:</strong> Compute full orthonormal PCA basis P<sub>S</sub> from source covariance and target subspace basis P<sub>T</sub> from target covariance.</li>
            <li><strong>Step 3:</strong> Compute principal angles between P<sub>S</sub> and P<sub>T</sub> via SVD.</li>
            <li><strong>Step 4:</strong> Construct the closed-form matrix blocks (Λ<sub>11</sub>, Λ<sub>12</sub>, Λ<sub>22</sub>) and assemble the PSD matrix G.</li>
            <li><strong>Step 5:</strong> Compute the symmetric matrix square root G<sup>1/2</sup>.</li>
            <li><strong>Step 6:</strong> Project both domains: Z<sub>S</sub> = X<sub>S</sub> G<sup>1/2</sup>, &nbsp; Z<sub>T</sub> = X<sub>T</sub> G<sup>1/2</sup>.</li>
            <li><strong>Step 7:</strong> Train standard classifier (1-NN, SVM) on (Z<sub>S</sub>, Y<sub>S</sub>) and evaluate on Z<sub>T</sub>.</li>
        </ol>

        <!-- ── Key Strengths & Highlights ────────────────────────── -->
        <h2>Key Strengths & Highlights</h2>
        <ul>
            <li><strong>Non-Iterative:</strong> Computes the closed-form kernel G directly without any optimization loops or iterative pseudo-labeling.</li>
            <li><strong>Fully Unsupervised:</strong> Requires zero target labels and zero pseudo-labels, eliminating error propagation.</li>
            <li><strong>Continuous Manifold Integration:</strong> Captures the infinite spectrum of intermediate domain representations between source and target.</li>
            <li><strong>Extremely Fast:</strong> Computable in milliseconds using standard matrix operations (PCA and SVD).</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters</h2>
        <ul>
            <li><strong>Subspace Dimension (d):</strong> The dimensionality of the linear subspaces on the Grassmannian manifold (typically 10 to 40). Can be set via cross-validation or subspace angle criteria.</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> GFK represents a landmark theoretical advance in transfer learning. By treating domain adaptation as a geodesic trajectory on the Grassmann manifold and integrating over an infinite continuum of intermediate subspaces, GFK derives a closed-form kernel G that maps source and target data into a domain-invariant space without iterations or heuristics.
        </div>
        """,
),
]

__all__ =["GFK_CONTENT"]
