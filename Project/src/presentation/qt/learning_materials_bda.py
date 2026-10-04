"""Complete learning material for Balanced Distribution Adaptation (BDA).

This is a self-contained module that provides the full BDA educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "BDA (Balanced Distribution Adaptation)".

Based on:
Wang et al., "Balanced Distribution Adaptation for Transfer Learning",
IEEE International Conference on Data Mining (ICDM), 2017.
"""

from __future__ import annotations 

BDA_CONTENT :list [tuple [str ,str ]]=[
(
"BDA (Balanced Distribution Adaptation)",
"""
        <span class="badge">Adaptive Marginal & Conditional Distribution Balancing</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In real-world domain adaptation tasks, transferring knowledge from a labeled source domain to an unlabeled target domain requires aligning probability distributions across domains.</p>

        <p>Prior feature-based methods typically focused on either:</p>
        <ul>
            <li><strong>Marginal distribution adaptation</strong> alone (e.g. TCA), which minimizes disparity between feature spaces P(X<sub>S</sub>) and P(X<sub>T</sub>) without accounting for class labels; or</li>
            <li><strong>Joint adaptation</strong> (e.g. JDA), which simultaneously adapts both marginal distributions P(X) and conditional distributions P(Y | X), but assigns them <em>equal importance</em>.</li>
        </ul>

        <p>However, assuming marginal and conditional distributions are equally important is often sub-optimal in practical applications. When domains are very dissimilar, the marginal shift dominates; when domains are already relatively close, conditional class alignment requires more emphasis.</p>

        <p><strong>Balanced Distribution Adaptation (BDA)</strong> addresses this limitation by introducing an adaptive balance factor <strong>μ</strong> that dynamically weights the importance of marginal and conditional distribution discrepancies.</p>

        <div class="card">
        <strong>Central Idea:</strong> Explicitly balance marginal and conditional distribution discrepancies via a factor μ ∈ [0, 1], generalizing earlier methods (such as TCA and JDA) as special cases.
        </div>

        <!-- ── The Fundamental Limitation of Existing Methods ───── -->
        <h2>The Core Problem: Unequal Importance of Distributions</h2>
        <p>Consider the total domain divergence between source domain D<sub>S</sub> and target domain D<sub>T</sub>. The discrepancy consists of two components:</p>
        <ol>
            <li><strong>Marginal discrepancy:</strong> D(P(X<sub>S</sub>), P(X<sub>T</sub>))</li>
            <li><strong>Conditional discrepancy:</strong> D(P(Y<sub>S</sub> | X<sub>S</sub>), P(Y<sub>T</sub> | X<sub>T</sub>))</li>
        </ol>

        <p>In standard JDA, both components are simply summed with equal weights (1 : 1):</p>
        <div class="formula">
        D<sub>JDA</sub>(D<sub>S</sub>, D<sub>T</sub>) = D(P(X<sub>S</sub>), P(X<sub>T</sub>)) + D(P(Y<sub>S</sub> | X<sub>S</sub>), P(Y<sub>T</sub> | X<sub>T</sub>))
        </div>

        <p>This equal weighting fails when:</p>
        <ul>
            <li><strong>Substantial feature shift exists:</strong> If the data collection conditions or sensor environments differ drastically, marginal discrepancy dominates and conditional pseudo-labels are initially too noisy to trust equally.</li>
            <li><strong>Fine-grained boundaries matter:</strong> If feature spaces are already somewhat aligned, marginal adaptation provides little marginal gain, while conditional boundaries require primary attention.</li>
        </ul>

        <!-- ── Mathematical Formulation & Balance Factor μ ──────── -->
        <h2>Mathematical Formulation of BDA</h2>
        <p>BDA models the overall distribution distance as a convex combination governed by the balance factor <strong>μ ∈ [0, 1]</strong>:</p>

        <div class="formula">
        D<sub>BDA</sub>(D<sub>S</sub>, D<sub>T</sub>) = (1 − μ) · D(P(X<sub>S</sub>), P(X<sub>T</sub>)) + μ · D(P(Y<sub>S</sub> | X<sub>S</sub>), P(Y<sub>T</sub> | X<sub>T</sub>))
        </div>

        <div class="card">
        <strong>Behavior across the range of μ:</strong><br/>
        <ul>
            <li><strong>μ → 0:</strong> Only marginal distribution adaptation is performed (equivalent to <strong>TCA</strong>).</li>
            <li><strong>μ = 0.5:</strong> Marginal and conditional distributions receive equal weight (equivalent to <strong>JDA</strong>).</li>
            <li><strong>μ → 1:</strong> Adaptation focuses entirely on class-conditional distributions.</li>
        </ul>
        BDA therefore acts as a unified framework that subsumes earlier methods as special cases.
        </div>

        <!-- ── MMD Representation & Discrepancy Matrix M ─────────── -->
        <h2>Maximum Mean Discrepancy Formulation</h2>
        <p>Using Maximum Mean Discrepancy (MMD) in a Reproducing Kernel Hilbert Space (RKHS), the marginal discrepancy matrix M<sub>0</sub> and per-class conditional discrepancy matrices M<sub>c</sub> are combined into a single balance matrix <strong>M</strong>:</p>

        <div class="formula">
        M = (1 − μ) · M<sub>0</sub> + μ · Σ<sub>c=1</sub><sup>C</sup> M<sub>c</sub>
        </div>

        <p>Where for class c ∈ {1, …, C}:</p>
        <ul>
            <li><strong>(M<sub>0</sub>)<sub>ij</sub>:</strong> 1/(n<sub>s</sub><sup>2</sup>) for source pairs, 1/(n<sub>t</sub><sup>2</sup>) for target pairs, and −1/(n<sub>s</sub> n<sub>t</sub>) for cross-domain pairs.</li>
            <li><strong>(M<sub>c</sub>)<sub>ij</sub>:</strong> 1/(n<sub>s,c</sub><sup>2</sup>) for source pairs in class c, 1/(n<sub>t,c</sub><sup>2</sup>) for target pairs predicted as class c, and −1/(n<sub>s,c</sub> n<sub>t,c</sub>) for cross-domain pairs in class c (0 otherwise).</li>
        </ul>

        <!-- ── BDA Optimization Problem ──────────────────────────── -->
        <h2>Optimization Objective</h2>
        <p>BDA seeks an optimal projection transformation matrix <strong>A</strong> that minimizes distribution discrepancy while preserving maximum data variance:</p>

        <div class="formula">
        min<sub>A</sub> &nbsp; tr(A<sup>T</sup> K M K A) + λ · tr(A<sup>T</sup> A)<br/><br/>
        subject to: &nbsp; A<sup>T</sup> K H K A = I
        </div>

        <p>Where:</p>
        <ul>
            <li><strong>K:</strong> (n<sub>s</sub> + n<sub>t</sub>) × (n<sub>s</sub> + n<sub>t</sub>) kernel matrix over all source and target samples.</li>
            <li><strong>H:</strong> Centering matrix H = I − (1/(n<sub>s</sub> + n<sub>t</sub>)) · 𝟏𝟏<sup>T</sup>.</li>
            <li><strong>λ:</strong> Regularization parameter guarding against rank deficiency.</li>
            <li><strong>A:</strong> Projection matrix whose columns represent the leading adaptation vectors.</li>
        </ul>

        <!-- ── Solution via Generalized Eigenvalue Problem ───────── -->
        <h2>Eigenvalue Solution</h2>
        <p>By applying the Lagrange multiplier technique, the constrained optimization problem transforms into a Rayleigh quotient solved by generalized eigenvalue decomposition:</p>

        <div class="formula">
        (K M K + λ I) · A = K H K · A · Φ
        </div>

        <p>The transformation matrix A is formed by the <em>k</em> smallest eigenvectors of (K M K + λ I)<sup>−1</sup> K H K (or equivalently the leading eigenvectors of the inverse formulation).</p>

        <!-- ── Weighted BDA (W-BDA) for Class Imbalance ───────────── -->
        <h2>Handling Class Imbalance: Weighted BDA (W-BDA)</h2>
        <p>A frequent challenge in transfer learning is <strong>class imbalance</strong> — the distribution of class proportions differs sharply between source and target domains (e.g. 90% positive in source vs 20% positive in target).</p>

        <p>Standard conditional adaptation assumes uniform class weights, causing majority classes to dominate the alignment at the expense of minority classes. <strong>W-BDA</strong> solves this by introducing per-class weighting coefficients <strong>w<sub>c</sub></strong> into the conditional matrices:</p>

        <div class="formula">
        M<sub>W-BDA</sub> = (1 − μ) · M<sub>0</sub> + μ · Σ<sub>c=1</sub><sup>C</sup> w<sub>c</sub> · M<sub>c</sub>
        </div>

        <p>Where w<sub>c</sub> is inversely proportional to class frequencies, guaranteeing that minority classes are not overwhelmed during distribution alignment.</p>

        <!-- ── Complete Step-by-Step Algorithm ──────────────────── -->
        <h2>Complete BDA / W-BDA Workflow</h2>
        <ol>
            <li><strong>Input:</strong> Source data X<sub>S</sub> with labels Y<sub>S</sub>, unlabeled target data X<sub>T</sub>, subspace dimension k, balance factor μ, regularization λ, iterations T.</li>
            <li><strong>Step 1:</strong> Construct kernel matrix K from [X<sub>S</sub>; X<sub>T</sub>] using linear, RBF, or SAM kernel.</li>
            <li><strong>Step 2:</strong> Compute marginal discrepancy matrix M<sub>0</sub> and centering matrix H.</li>
            <li><strong>Step 3:</strong> Train an initial base classifier (e.g. 1-NN or SVM) on X<sub>S</sub> and predict initial target pseudo-labels Ŷ<sub>T</sub>.</li>
            <li><strong>Step 4: Iterative Alignment Loop (t = 1 to T):</strong>
                <ul>
                    <li>Construct class-conditional matrices M<sub>c</sub> for each class c using current pseudo-labels Ŷ<sub>T</sub>.</li>
                    <li>Assemble the balanced matrix M = (1 − μ) M<sub>0</sub> + μ Σ<sub>c</sub> M<sub>c</sub> (or apply class weights for W-BDA).</li>
                    <li>Solve the generalized eigenvalue problem (K M K + λ I) A = K H K A Φ.</li>
                    <li>Extract leading k eigenvectors to construct projection matrix A.</li>
                    <li>Project source and target data: Z<sub>S</sub> = A<sup>T</sup> K<sub>S</sub>, Z<sub>T</sub> = A<sup>T</sup> K<sub>T</sub>.</li>
                    <li>Retrain base classifier on (Z<sub>S</sub>, Y<sub>S</sub>) and update target pseudo-labels Ŷ<sub>T</sub> = Classifier(Z<sub>T</sub>).</li>
                </ul>
            </li>
            <li><strong>Output:</strong> Adapted representations Z<sub>S</sub>, Z<sub>T</sub> and final target predictions Ŷ<sub>T</sub>.</li>
        </ol>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters</h2>
        <ul>
            <li><strong>Balance Factor (μ):</strong> The defining parameter of BDA (range [0, 1]). Set close to 0 when domains differ widely; set close to 0.5–0.8 when domains share close feature characteristics. Default: 0.5 or task-tuned.</li>
            <li><strong>Subspace Dimension (k):</strong> Number of latent projection vectors (typically 5 to 50).</li>
            <li><strong>Regularization (λ):</strong> Stability coefficient (typically 0.1 to 1.0).</li>
            <li><strong>Iterations (T):</strong> Number of pseudo-label refinement passes (typically 5 to 10).</li>
            <li><strong>Kernel Type:</strong> Linear for high-dimensional text/bag-of-words; RBF for continuous visual features.</li>
        </ul>

        <!-- ── Method Comparison ─────────────────────────────────── -->
        <h2>BDA vs. TCA vs. JDA</h2>
        <div class="card">
        <table style="width:100%; border-collapse: collapse; font-size:12px;">
            <tr style="border-bottom: 1px solid #444; text-align: left;">
                <th style="padding: 6px;">Method</th>
                <th style="padding: 6px;">Marginal P(X)</th>
                <th style="padding: 6px;">Conditional P(Y|X)</th>
                <th style="padding: 6px;">Balancing Mechanism</th>
                <th style="padding: 6px;">Class Imbalance</th>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 6px;"><strong>TCA</strong></td>
                <td style="padding: 6px;">Yes (Weight = 1)</td>
                <td style="padding: 6px;">No</td>
                <td style="padding: 6px;">None</td>
                <td style="padding: 6px;">No</td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 6px;"><strong>JDA</strong></td>
                <td style="padding: 6px;">Yes (Weight = 1)</td>
                <td style="padding: 6px;">Yes (Weight = 1)</td>
                <td style="padding: 6px;">Fixed 1:1 equal</td>
                <td style="padding: 6px;">No</td>
            </tr>
            <tr>
                <td style="padding: 6px;"><strong>BDA / W-BDA</strong></td>
                <td style="padding: 6px;">Yes (Weight = 1 − μ)</td>
                <td style="padding: 6px;">Yes (Weight = μ)</td>
                <td style="padding: 6px;"><strong>Adaptive via μ ∈ [0, 1]</strong></td>
                <td style="padding: 6px;"><strong>Yes (via W-BDA)</strong></td>
            </tr>
        </table>
        </div>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> BDA breaks the assumption of equal distribution importance in transfer learning. By dynamically balancing marginal and conditional distributions via parameter μ and addressing class imbalance via W-BDA, it achieves significantly better adaptation flexibility across diverse cross-domain scenarios.
        </div>
        """,
),
]

__all__ =["BDA_CONTENT"]
