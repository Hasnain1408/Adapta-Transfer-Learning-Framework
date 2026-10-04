"""Complete learning material for Transfer Component Analysis (TCA).

This is a self-contained module that provides the full TCA educational
content as a single (title, html_body) pair.  It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog as one scrollable article under the sidebar entry "TCA".

The content is adapted from the extended guide originally in
Learning_materials_content/tca.txt.
"""

from __future__ import annotations 

TCA_CONTENT :list [tuple [str ,str ]]=[
(
"TCA (Transfer Component Analysis)",
"""
        <span class="badge">Feature-Based Domain Adaptation</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In many machine learning problems, we train a model using data from one domain and then want to use that model on data from another domain.
        Although the two domains may contain the same types of features and represent a related problem, their data distributions can be different.</p>

        <p>For example, suppose we train a defect prediction model using software project <strong>A</strong> and want to apply it to software project <strong>B</strong>.
        The features in both projects may represent similar software characteristics, but their distributions may be different.
        A model trained directly on project A may therefore not perform as well on project B.</p>

        <p>This problem is known as <strong>domain adaptation</strong>.</p>

        <p>One important idea in domain adaptation is to find a new feature representation in which the source and target domains become more similar.
        <strong>Transfer Component Analysis (TCA)</strong> is a feature extraction method designed for this purpose.</p>

        <div class="card">
        <strong>Main Idea:</strong> Find a new feature space in which the source and target data distributions are close to each other, while preserving useful information from the original data.
        </div>

        <p>TCA achieves this by learning a set of <strong>transfer components</strong> using a kernel method and <strong>Maximum Mean Discrepancy (MMD)</strong>.</p>

        <!-- ── The Domain Adaptation Problem ─────────────────────── -->
        <h2>The Domain Adaptation Problem</h2>
        <p>Consider two domains:</p>
        <ul>
            <li><strong>Source domain:</strong> labeled data is available.</li>
            <li><strong>Target domain:</strong> data is available, but labels are unavailable or limited.</li>
        </ul>

        <p>We can represent them as:</p>
        <div class="formula">
        D<sub>S</sub> = {(x<sub>1</sub><sup>S</sup>, y<sub>1</sub><sup>S</sup>), …, (x<sub>n₁</sub><sup>S</sup>, y<sub>n₁</sub><sup>S</sup>)}
        </div>
        <p>and</p>
        <div class="formula">
        D<sub>T</sub> = {x<sub>1</sub><sup>T</sup>, …, x<sub>n₂</sub><sup>T</sup>}
        </div>

        <p>The source domain contains both features and labels, while the target domain contains only features.
        The goal is to predict the unknown target labels.</p>

        <p>The important assumption is that the source and target domains are related, but their input distributions can be different:</p>
        <div class="formula">
        P(X<sub>S</sub>) ≠ Q(X<sub>T</sub>)
        </div>

        <div class="card">
        <strong>In simple words:</strong> The source and target contain related information, but the way their data is distributed is different.
        </div>

        <!-- ── Why Is Distribution Difference a Problem? ─────────── -->
        <h2>Why Is Distribution Difference a Problem?</h2>
        <p>Suppose a machine learning model is trained using the following source data:</p>

        <div class="formula">
        Source domain<br/>
        Feature 1 &nbsp;&nbsp; Feature 2<br/>
        &nbsp;&nbsp; 10 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 20<br/>
        &nbsp;&nbsp; 12 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 22<br/>
        &nbsp;&nbsp; 11 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 19<br/>
        &nbsp;&nbsp; 13 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 21
        </div>

        <p>Now suppose the target domain contains:</p>

        <div class="formula">
        Target domain<br/>
        Feature 1 &nbsp;&nbsp; Feature 2<br/>
        &nbsp;&nbsp; 30 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 40<br/>
        &nbsp;&nbsp; 32 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 42<br/>
        &nbsp;&nbsp; 31 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 39<br/>
        &nbsp;&nbsp; 33 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 41
        </div>

        <p>The features may describe the same underlying problem, but their distributions are clearly different.
        A model trained only on the source representation may therefore have difficulty generalizing to the target domain.</p>

        <p>TCA addresses this by creating a <strong>new representation</strong> of the data.
        Instead of directly using X<sub>S</sub> and X<sub>T</sub>, TCA learns a transformation such that:</p>
        <div class="formula">
        X'<sub>S</sub> = φ(X<sub>S</sub>) &nbsp;&nbsp; and &nbsp;&nbsp; X'<sub>T</sub> = φ(X<sub>T</sub>)
        </div>
        <p>where the transformed source and target distributions are closer.</p>

        <div class="card">
        <strong>Conceptually:</strong><br/>
        Original feature space → <em>TCA</em> → Common / transfer feature space<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├── Source representation<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;└── Target representation
        </div>

        <!-- ── What Is TCA? ──────────────────────────────────────── -->
        <h2>What Is TCA?</h2>
        <p><strong>Transfer Component Analysis (TCA)</strong> is a domain adaptation method that learns a set of common components from source and target data.
        These components are called <strong>transfer components</strong>.</p>

        <p>The components are designed to satisfy two important requirements:</p>
        <ol>
            <li>Reduce the difference between the source and target distributions.</li>
            <li>Preserve useful structure in the original data.</li>
        </ol>

        <p>The resulting representation can then be used with standard machine learning algorithms such as classifiers or regression models.</p>

        <div class="card">
        <strong>Workflow:</strong><br/><br/>
        Source data ──┐<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TCA<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑<br/>
        Target data ──┘<br/><br/>
        Source transformed data → Train ML model → Predict on target transformed data
        </div>

        <p>Thus, TCA does not itself have to be the final classifier or regression algorithm.
        Instead, it produces a better representation on which a conventional machine learning method can be trained.</p>

        <!-- ── Feature Representation ────────────────────────────── -->
        <h2>Feature Representation</h2>
        <p>A <strong>feature representation</strong> is the way data is represented so that a machine learning algorithm can work with it.</p>

        <p>Suppose the original data contains:</p>
        <div class="formula">
        X = [x₁, x₂, x₃, …, x<sub>d</sub>]
        </div>

        <p>TCA transforms the original representation into a new representation:</p>
        <div class="formula">
        X' = [z₁, z₂, …, z<sub>m</sub>] &nbsp;&nbsp; where usually m &lt; d
        </div>

        <p>The new features are not necessarily individual original features. They are <strong>learned components</strong> that combine information from the original data.
        This is why TCA can also be viewed as a <strong>feature extraction</strong> method.</p>

        <div class="card">
        <strong>Key insight:</strong> These components are learned using <em>both</em> source and target data, rather than learning the representation from the source domain alone.
        </div>

        <!-- ── Maximum Mean Discrepancy (MMD) ────────────────────── -->
        <h2>Maximum Mean Discrepancy (MMD)</h2>
        <p>One of the most important concepts behind TCA is <strong>Maximum Mean Discrepancy</strong>, or <strong>MMD</strong>.</p>

        <p>MMD is a measure of how different two probability distributions are. In the context of domain adaptation:</p>
        <ul>
            <li><strong>Large MMD</strong> → source and target are more different.</li>
            <li><strong>Small MMD</strong> → source and target are more similar.</li>
        </ul>

        <p>The paper measures the difference between distributions by comparing the mean representations of the two domains in a <strong>Reproducing Kernel Hilbert Space (RKHS)</strong>.</p>

        <p>The empirical MMD is written as:</p>
        <div class="formula">
        Dist(X, Y) = ‖ (1/n₁) Σ φ(xᵢ) − (1/n₂) Σ φ(yⱼ) ‖<sub>ℋ</sub>
        </div>

        <div class="card">
        <strong>In simple terms:</strong> Map the source and target samples into a feature space, calculate their average representations, and measure how far those averages are from each other.
        </div>

        <p>TCA tries to find a representation where this distance is small.</p>

        <!-- ── Why Does TCA Use a Kernel? ────────────────────────── -->
        <h2>Why Does TCA Use a Kernel?</h2>
        <p>TCA uses a <strong>kernel-based approach</strong>.
        Instead of explicitly constructing a complicated nonlinear transformation, TCA uses a kernel function to work with the data in a higher-dimensional feature space.</p>

        <p>The kernel function is:</p>
        <div class="formula">
        k(xᵢ, xⱼ) = φ(xᵢ)<sup>T</sup> φ(xⱼ)
        </div>

        <p>This is known as the <strong>kernel trick</strong>. The advantage is that TCA can capture nonlinear relationships without explicitly calculating the potentially very high-dimensional feature mapping φ.</p>

        <p>The kernel values are collected into a <strong>kernel matrix</strong>:</p>
        <div class="formula">
        K = [ K<sub>S,S</sub> &nbsp; K<sub>S,T</sub> ]<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;[ K<sub>T,S</sub> &nbsp; K<sub>T,T</sub> ]
        </div>

        <ul>
            <li><strong>K<sub>S,S</sub>:</strong> relationships between source samples</li>
            <li><strong>K<sub>T,T</sub>:</strong> relationships between target samples</li>
            <li><strong>K<sub>S,T</sub>:</strong> relationships between source and target samples</li>
            <li><strong>K<sub>T,S</sub>:</strong> relationships between target and source samples</li>
        </ul>

        <p>This kernel matrix is central to the TCA optimization process.</p>

        <!-- ── The Core Idea of TCA ──────────────────────────────── -->
        <h2>The Core Idea of TCA</h2>

        <div class="card">
        <strong>Goal:</strong> Find a transformation that minimizes the difference between the source and target distributions in the new feature space.
        </div>

        <p>Suppose the learned transformation is represented by <strong>W</strong>.
        The original data is projected using W, producing a new representation.
        TCA wants:</p>
        <div class="formula">
        Source distribution ≈ Target distribution
        </div>

        <p>At the same time, TCA should not simply destroy all information in the data. Therefore, TCA balances two goals:</p>
        <ul>
            <li><strong>Goal 1 — Reduce domain difference:</strong> Make source and target distributions close.</li>
            <li><strong>Goal 2 — Preserve useful information:</strong> Avoid producing a meaningless or collapsed representation.</li>
        </ul>

        <!-- ── TCA Objective Function ────────────────────────────── -->
        <h2>TCA Objective Function</h2>
        <p>The main optimization problem presented in the paper is:</p>

        <div class="formula">
        min<sub>W</sub> &nbsp; tr(W<sup>T</sup>W) + μ · tr(W<sup>T</sup>KLKW)
        </div>

        <p>subject to:</p>
        <div class="formula">
        W<sup>T</sup>KHKW = I
        </div>

        <p>Although the equation looks complicated, its meaning is relatively simple. We can understand it by examining each part below.</p>

        <!-- ── Understanding W ───────────────────────────────────── -->
        <h2>Understanding W</h2>
        <p>The matrix <strong>W</strong> contains the <strong>transfer components</strong> learned by TCA.
        It determines how the original data will be transformed into the new feature space.</p>

        <div class="formula">
        Original data →<sup>W</sup>→ Transfer representation
        </div>

        <p>If we want <em>m</em> transfer components, W contains the information needed to construct those <em>m</em> components.</p>

        <!-- ── Understanding the Regularization Term ─────────────── -->
        <h2>Understanding the Regularization Term</h2>
        <p>The first term is:</p>
        <div class="formula">
        tr(W<sup>T</sup>W)
        </div>

        <p>This is a <strong>regularization term</strong>.
        Its purpose is to control the complexity of the learned transformation — TCA does not want to learn an unnecessarily complicated transformation.</p>

        <p>The paper notes that this regularization also helps avoid rank-deficiency problems when solving the resulting generalized eigenvalue problem.</p>

        <!-- ── Understanding the Domain-Discrepancy Term ──────────── -->
        <h2>Understanding the Domain-Discrepancy Term</h2>
        <p>The second term is:</p>
        <div class="formula">
        μ · tr(W<sup>T</sup>KLKW)
        </div>

        <p>This is the important domain adaptation part.
        The matrices <strong>K</strong> and <strong>L</strong> work together to represent the difference between the source and target distributions.</p>

        <p>The matrix <strong>L</strong> is constructed so that:</p>
        <ul>
            <li>source–source relationships receive <strong>positive</strong> values,</li>
            <li>target–target relationships receive <strong>positive</strong> values,</li>
            <li>source–target relationships receive <strong>negative</strong> values.</li>
        </ul>

        <div class="card">
        <strong>Interpretation:</strong> tr(W<sup>T</sup>KLKW) represents the <strong>domain gap in the learned feature space</strong>.
        </div>

        <!-- ── Understanding μ ───────────────────────────────────── -->
        <h2>Understanding μ (Trade-off Parameter)</h2>
        <p>The parameter <strong>μ</strong> controls the importance of the domain-discrepancy term:</p>
        <div class="formula">
        Objective = Regularization + μ × Domain discrepancy
        </div>

        <p>Therefore, μ controls the <strong>trade-off</strong> between controlling the transformation and reducing the difference between domains.</p>

        <!-- ── Understanding the Constraint ──────────────────────── -->
        <h2>Understanding the Constraint</h2>
        <div class="formula">
        W<sup>T</sup>KHKW = I
        </div>

        <p>Without a constraint, a trivial solution such as <strong>W = 0</strong> could make the transformed data collapse.
        If every sample were mapped to the same point, the source and target distributions could appear identical —
        but the resulting representation would contain <strong>no useful information</strong>.</p>

        <p>The constraint prevents this trivial collapse and preserves meaningful variation in the transformed data.</p>

        <div class="card">
        <strong>Objective in one sentence:</strong> Find transfer components that reduce the source-target distribution difference, keep the transformation controlled, and prevent the transformed data from collapsing into a trivial representation.
        </div>

        <!-- ── From Optimization to Eigenvalue Decomposition ──────── -->
        <h2>From Optimization to Eigenvalue Decomposition</h2>
        <p>The TCA optimization problem can be reformulated as an eigenvalue problem.
        The paper shows that the solution is obtained from the eigenvectors corresponding to the leading eigenvalues of:</p>
        <div class="formula">
        (I + μ KLK)<sup>−1</sup> KHK
        </div>

        <p>TCA selects the leading <em>m</em> eigenvectors, and these eigenvectors form the learned transfer components.</p>

        <div class="card">
        <strong>Step-by-step:</strong><br/>
        1. Build K (kernel matrix)<br/>
        2. Build L (MMD matrix)<br/>
        3. Build H (centering matrix)<br/>
        4. Construct the eigenvalue problem<br/>
        5. Perform eigendecomposition<br/>
        6. Select leading eigenvectors<br/>
        7. → Transfer components
        </div>

        <!-- ── What Are the Transfer Components? ─────────────────── -->
        <h2>What Are the Transfer Components?</h2>
        <p>The selected eigenvectors are called <strong>transfer components</strong>.
        Suppose we want <em>m</em> components:</p>
        <div class="formula">
        W = [w₁, w₂, …, w<sub>m</sub>]
        </div>

        <p>These components define the new feature space.
        The original source and target data are then represented using these components:</p>
        <div class="formula">
        X<sub>S</sub> → Z<sub>S</sub> &nbsp;&nbsp; and &nbsp;&nbsp; X<sub>T</sub> → Z<sub>T</sub>
        </div>

        <div class="card">
        <strong>Key property:</strong> Z<sub>S</sub> and Z<sub>T</sub> should have a <strong>smaller distribution difference</strong> than the original representations.
        </div>

        <!-- ── Complete TCA Workflow ─────────────────────────────── -->
        <h2>Complete TCA Workflow</h2>

        <p><strong>Step 1:</strong> Obtain source data D<sub>S</sub> (with labels) and target data D<sub>T</sub> (unlabeled).</p>
        <p><strong>Step 2:</strong> Combine the feature data — source and target samples are considered together.</p>
        <p><strong>Step 3:</strong> Construct the kernel matrix K.</p>
        <p><strong>Step 4:</strong> Construct the MMD matrix L.</p>
        <p><strong>Step 5:</strong> Construct the centering matrix H = I − (1/(n₁+n₂)) · 𝟏𝟏<sup>T</sup>.</p>
        <p><strong>Step 6:</strong> Solve the eigenvalue problem: (I + μ KLK)<sup>−1</sup> KHK.</p>
        <p><strong>Step 7:</strong> Select leading m eigenvectors → Transfer Components.</p>
        <p><strong>Step 8:</strong> Transform both source and target data.</p>
        <p><strong>Step 9:</strong> Train a classifier on transformed source data.</p>
        <p><strong>Step 10:</strong> Predict target labels using the trained model.</p>

        <!-- ── A Simple Example ──────────────────────────────────── -->
        <h2>A Simple Example</h2>
        <p>Imagine that we want to transfer a software defect prediction model.</p>

        <div class="formula">
        Source project: 100 labeled instances, 20 features<br/>
        Target project: 200 unlabeled instances, 20 features
        </div>

        <p>The source and target projects use the same 20 features, but their distributions are different.</p>

        <p><strong>Without TCA:</strong></p>
        <div class="formula">
        X<sub>S</sub> → Model → apply to X<sub>T</sub> &nbsp;(distribution difference may cause poor generalization)
        </div>

        <p><strong>With TCA:</strong></p>
        <div class="formula">
        X<sub>S</sub>, X<sub>T</sub> →<sup>TCA</sup>→ Z<sub>S</sub>, Z<sub>T</sub> → Model trained on Z<sub>S</sub> → Predict Z<sub>T</sub>
        </div>

        <div class="card">
        The classifier works in the <strong>TCA feature space</strong> rather than directly in the original feature space.
        </div>

        <!-- ── When to Use TCA ───────────────────────────────────── -->
        <h2>When to Use TCA</h2>
        <ul>
            <li>Source and target domains are related.</li>
            <li>The source domain has labeled data.</li>
            <li>The target domain has unlabeled data.</li>
            <li>The feature distributions differ between domains.</li>
            <li>You want to learn a common representation before training a model.</li>
        </ul>

        <p>The original paper demonstrates TCA on <strong>cross-domain WiFi localization</strong> and <strong>cross-domain text classification</strong>.</p>

        <!-- ── TCA vs. PCA ───────────────────────────────────────── -->
        <h2>TCA vs. Ordinary PCA</h2>
        <p>TCA can look similar to PCA because both produce a lower-dimensional representation, but their goals are different.</p>
        <div class="formula">
        <strong>PCA:</strong> Find directions that preserve maximum variance.<br/>
        <strong>TCA:</strong> Find components that reduce source-target distribution difference while preserving useful structure.
        </div>
        <p>TCA is specifically designed for <strong>domain adaptation</strong>, rather than ordinary dimensionality reduction.</p>

        <!-- ── TCA vs. Instance Reweighting ──────────────────────── -->
        <h2>TCA vs. Instance Reweighting</h2>
        <p>Methods such as KMM and KLIEP address domain adaptation by giving different weights to training instances.
        TCA takes a different approach — instead of changing the importance of individual samples, it learns a <strong>new feature representation</strong> in which the source and target distributions become closer.</p>

        <!-- ── Important Parameters ──────────────────────────────── -->
        <h2>Important Parameters</h2>

        <p><strong>Number of components (m):</strong> Determines the dimensionality of the new feature space (e.g. m = 5 means five transfer components).</p>

        <p><strong>Kernel:</strong> Determines how relationships between samples are represented. The paper experiments with <strong>linear</strong> and <strong>RBF</strong> kernels.</p>

        <p><strong>μ (Regularization trade-off):</strong> Controls the trade-off between regularization and domain discrepancy. The paper uses μ = 0.1 in several settings.</p>

        <!-- ── Advantages of TCA ─────────────────────────────────── -->
        <h2>Advantages of TCA</h2>
        <ul>
            <li><strong>Directly addresses domain difference</strong> — explicitly reduces the difference between source and target distributions.</li>
            <li><strong>Learns a latent representation</strong> — learns transfer components automatically instead of manual feature selection.</li>
            <li><strong>Captures nonlinear relationships</strong> — uses kernel methods for nonlinear feature mappings.</li>
            <li><strong>Supports out-of-sample data</strong> — can generalize its learned kernel representation to unseen patterns.</li>
            <li><strong>Computationally efficient</strong> — uses eigendecomposition instead of expensive semi-definite programming. Complexity: O(m(n₁ + n₂)²).</li>
        </ul>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary</h2>

        <div class="card">
        <strong>Complete TCA Pipeline:</strong><br/><br/>
        Source + Target data → Build kernel K → Build MMD matrix L → Build centering matrix H
        → Solve eigenproblem → Select m leading eigenvectors → Transfer components
        → Transform source and target → Train classifier → Predict target
        </div>

        <div class="formula">
        K → L, H → solve (I + μ KLK)<sup>−1</sup> KHK → W = [w₁, w₂, …, w<sub>m</sub>]
        </div>

        <div class="card">
        <strong>One-line definition:</strong> TCA is a domain adaptation technique that learns transfer components to minimize the distribution difference between source and target domains in a new feature space.
        </div>
        """,
),
]

__all__ =["TCA_CONTENT"]
