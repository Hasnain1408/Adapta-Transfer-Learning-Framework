"""Complete learning material for Joint Distribution Adaptation (JDA).

This is a self-contained module that provides the full JDA educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog as one scrollable article under the sidebar entry
"JDA (Joint Domain Adaptation)".

The content is adapted from the extended guide originally in
Learning_materials_content/jda.txt.
"""

from __future__ import annotations 

JDA_CONTENT :list [tuple [str ,str ]]=[
(
"JDA (Joint Domain Adaptation)",
"""
        <span class="badge">Joint Marginal & Conditional Adaptation</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In machine learning, a model is often trained using labeled data from one domain and then applied to another related domain.</p>

        <p>For example, suppose a software defect prediction model is trained using one software project and then applied to another project. Both projects may contain similar features, but the distributions of those features may be different.</p>

        <p>The difference between the source and target domains can make it difficult for a model trained on the source domain to perform well on the target domain. This problem is known as <strong>domain adaptation</strong>.</p>

        <p>One method designed to address this problem is <strong>Joint Distribution Adaptation (JDA)</strong>. JDA learns a new feature representation in which the <strong>joint distributions</strong> of the source and target domains become similar.</p>

        <div class="card">
        <strong>Central Idea:</strong> Reduce the difference between the source and target marginal distributions and conditional distributions at the same time.
        </div>

        <p>This is an important difference from methods that only try to align the overall feature distributions.</p>

        <!-- ── The Domain Adaptation Problem ─────────────────────── -->
        <h2>The Domain Adaptation Problem</h2>
        <p>Consider two domains:</p>
        <ul>
            <li><strong>Source domain:</strong> labeled data is available.</li>
            <li><strong>Target domain:</strong> data is available, but labels are initially unavailable.</li>
        </ul>

        <p>The source domain can be represented as:</p>
        <div class="formula">
        D<sub>S</sub> = {(x<sub>i</sub><sup>s</sup>, y<sub>i</sub><sup>s</sup>)}<sub>i=1</sub><sup>n<sub>s</sub></sup>
        </div>
        <p>where x<sub>i</sub><sup>s</sup> represents a source sample, y<sub>i</sub><sup>s</sup> represents its label, and n<sub>s</sub> is the number of source samples.</p>

        <p>The target domain is:</p>
        <div class="formula">
        D<sub>T</sub> = {x<sub>j</sub><sup>t</sup>}<sub>j=1</sub><sup>n<sub>t</sub></sup>
        </div>
        <p>where x<sub>j</sub><sup>t</sup> represents a target sample, and n<sub>t</sub> is the number of target samples.</p>

        <p>The target labels are unknown. The goal is to learn from the labeled source domain and use that knowledge to make predictions for the target domain.</p>

        <p>The difficulty is that the source and target distributions may be different:</p>
        <div class="formula">
        P<sub>S</sub>(X) ≠ P<sub>T</sub>(X)
        </div>
        <p>There may also be a difference in the relationship between features and labels:</p>
        <div class="formula">
        P<sub>S</sub>(Y | X) ≠ P<sub>T</sub>(Y | X)
        </div>

        <p>Therefore, simply training a model on the source domain may not be sufficient.</p>

        <!-- ── What Is JDA? ──────────────────────────────────────── -->
        <h2>What Is JDA?</h2>
        <p><strong>Joint Distribution Adaptation (JDA)</strong> is a domain adaptation method that learns a common feature representation for source and target domains. The word <strong>joint</strong> is important.</p>

        <p>JDA tries to align two types of distributions:</p>
        <ol>
            <li><strong>Marginal distributions</strong>: P(X)</li>
            <li><strong>Conditional distributions</strong>: P(Y | X)</li>
        </ol>

        <p>Instead of considering only P(X), JDA considers the joint distribution:</p>
        <div class="formula">
        P(X, Y)
        </div>

        <p>Since the target labels are unknown, JDA estimates target labels during the adaptation process.</p>

        <div class="card">
        <strong>Workflow Overview:</strong><br/><br/>
        Source data ───────┐<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JDA<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Common feature space<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑<br/>
        Target data ───────┘
        </div>

        <!-- ── Why Does JDA Consider Joint Distributions? ────────── -->
        <h2>Why Does JDA Consider Joint Distributions?</h2>
        <p>To understand JDA, first consider two different types of distribution differences.</p>

        <h3>Marginal Distribution</h3>
        <p>The <strong>marginal distribution</strong> describes how the input features are distributed: <strong>P(X)</strong>.</p>

        <div class="formula">
        Source: 10, 11, 12, 13, 14<br/>
        Target: 30, 31, 32, 33, 34
        </div>
        <p>Here, the source and target feature distributions are different. JDA reduces this difference.</p>

        <h3>Conditional Distribution</h3>
        <p>The <strong>conditional distribution</strong> describes the relationship between the features and their class labels: <strong>P(Y | X)</strong>.</p>
        <p>For example, suppose two domains have similar feature distributions but the relationship between features and classes is different. Aligning only P(X) may not be sufficient — the relationship between features and labels must also be considered.</p>

        <!-- ── The Joint Distribution ────────────────────────────── -->
        <h2>The Joint Distribution</h2>
        <p>The joint distribution combines both the feature distribution and the label relationship: <strong>P(X, Y)</strong>.</p>

        <p>Using the probability chain rule:</p>
        <div class="formula">
        P(X, Y) = P(Y | X) · P(X)
        </div>

        <p>This equation is fundamental to JDA. It shows that the joint distribution depends on:</p>
        <ul>
            <li>the marginal distribution P(X) — feature distribution;</li>
            <li>the conditional distribution P(Y | X) — label relationship.</li>
        </ul>

        <div class="card">
        <strong>Decomposition:</strong><br/>
        Joint Distribution P(X, Y)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├── Marginal P(X) (Feature distribution)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;└── Conditional P(Y | X) (Label relationship)
        </div>

        <!-- ── JDA vs. Basic Distribution Matching ───────────────── -->
        <h2>JDA vs. the Basic Distribution-Matching Idea</h2>
        <p>Suppose we only try to minimize the feature distribution distance:</p>
        <div class="formula">
        Dist(P<sub>S</sub>(X), P<sub>T</sub>(X))
        </div>
        <p>This aligns feature distributions, but does not explicitly consider class information.</p>

        <p>JDA extends this by considering both:</p>
        <div class="formula">
        Dist(P<sub>S</sub>(X), P<sub>T</sub>(X)) &nbsp;&nbsp; and &nbsp;&nbsp; Dist(P<sub>S</sub>(Y | X), P<sub>T</sub>(Y | X))
        </div>

        <p>Therefore, JDA aligns P<sub>S</sub>(X) with P<sub>T</sub>(X) and P<sub>S</sub>(Y | X) with P<sub>T</sub>(Y | X). This is why the method is called <strong>Joint Distribution Adaptation</strong>.</p>

        <!-- ── Maximum Mean Discrepancy (MMD) ────────────────────── -->
        <h2>Maximum Mean Discrepancy (MMD)</h2>
        <p>JDA uses <strong>Maximum Mean Discrepancy (MMD)</strong> to measure distribution differences in a kernel-induced feature space:</p>
        <ul>
            <li><strong>Large MMD</strong> ⇒ large distribution difference</li>
            <li><strong>Small MMD</strong> ⇒ similar distributions</li>
        </ul>

        <p>For two distributions P and Q, MMD is defined as:</p>
        <div class="formula">
        D(P, Q) = ‖ E<sub>x~P</sub>[φ(x)] − E<sub>y~Q</sub>[φ(y)] ‖<sub>ℋ</sub>
        </div>

        <p>In simple terms: map samples into a feature space, calculate the mean representation for each domain, and measure the distance between the two mean representations.</p>

        <!-- ── Marginal Distribution Adaptation ──────────────────── -->
        <h2>Marginal Distribution Adaptation</h2>
        <p>The first part of JDA is marginal distribution adaptation, reducing D(P<sub>S</sub>(X), P<sub>T</sub>(X)).</p>

        <div class="card">
        <strong>Before adaptation:</strong><br/>
        Source: ● ● ● ● &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Target: ○ ○ ○ ○<br/><br/>
        <strong>After adaptation:</strong><br/>
        ● ○ ● ○ ● ○ ● ○
        </div>

        <p>The exact representation is learned mathematically through the kernel formulation used by JDA.</p>

        <!-- ── Conditional Distribution Adaptation ────────────────── -->
        <h2>Conditional Distribution Adaptation</h2>
        <p>The second part of JDA is conditional distribution adaptation, reducing D(P<sub>S</sub>(Y | X), P<sub>T</sub>(Y | X)).</p>
        <p>This requires information about the target labels. However, target labels are initially unknown. JDA addresses this using <strong>pseudo-labels</strong> in an iterative process:</p>

        <div class="card">
        <strong>Iterative refinement:</strong><br/>
        Initial target data → Initial prediction → Pseudo-label target samples → Estimate conditional distributions → Adapt source & target → New feature representation → New target predictions → Updated pseudo-labels → Repeat
        </div>

        <!-- ── What Are Pseudo-Labels? ───────────────────────────── -->
        <h2>What Are Pseudo-Labels?</h2>
        <p>A <strong>pseudo-label</strong> is an estimated label assigned to an unlabeled target sample.</p>
        <p>If target sample x<sub>1</sub> has unknown true label, a classifier trained on the adapted space may predict ŷ<sub>1</sub> = 1. JDA uses ŷ<sub>1</sub> as a temporary label to estimate P<sub>T</sub>(Y | X). As the representation improves, predictions are updated.</p>

        <!-- ── The Core Idea of JDA ──────────────────────────────── -->
        <h2>The Core Idea of JDA</h2>
        <div class="card">
        <strong>Core Principle:</strong> Align the source and target marginal distributions and conditional distributions in a common feature space.
        </div>

        <div class="formula">
        Joint Adaptation = Marginal Adaptation + Conditional Adaptation
        </div>

        <!-- ── JDA Objective Function ────────────────────────────── -->
        <h2>JDA Objective Function</h2>
        <p>The JDA objective minimizes the combined marginal and conditional discrepancies:</p>

        <div class="formula">
        min<sub>A</sub> &nbsp; tr(A<sup>T</sup> K M K A)
        </div>
        <p>subject to:</p>
        <div class="formula">
        A<sup>T</sup> K H K A = I
        </div>

        <p>Here:</p>
        <ul>
            <li><strong>A</strong> is the learned transformation matrix;</li>
            <li><strong>K</strong> is the kernel matrix;</li>
            <li><strong>M</strong> represents the joint distribution discrepancy;</li>
            <li><strong>H</strong> is the centering matrix.</li>
        </ul>

        <!-- ── Understanding the Matrix M ────────────────────────── -->
        <h2>Understanding the Matrix M</h2>
        <p>The matrix <strong>M</strong> is the central component of JDA. It represents the total discrepancy to minimize:</p>

        <div class="formula">
        M = M<sub>0</sub> + Σ<sub>c=1</sub><sup>C</sup> M<sub>c</sub>
        </div>

        <div class="card">
        <strong>M = Marginal Discrepancy (M<sub>0</sub>) + Conditional Discrepancy (Σ M<sub>c</sub>)</strong>
        </div>

        <!-- ── Marginal Matrix M0 & Conditional Matrices Mc ──────── -->
        <h2>Marginal Matrix M<sub>0</sub> and Conditional Matrices M<sub>c</sub></h2>

        <h3>Marginal Matrix M<sub>0</sub></h3>
        <p>M<sub>0</sub> measures the difference between P<sub>S</sub>(X) and P<sub>T</sub>(X). It assigns values according to domain membership:</p>
        <ul>
            <li>Source ↔ Source → positive relationship</li>
            <li>Target ↔ Target → positive relationship</li>
            <li>Source ↔ Target → negative relationship</li>
        </ul>

        <h3>Conditional Matrices M<sub>c</sub></h3>
        <p>For each class c, M<sub>c</sub> measures the distribution difference between source and target samples belonging to class c (using known source labels and current target pseudo-labels).</p>
        <p>Summing across all classes gives the total conditional discrepancy Σ<sub>c</sub> M<sub>c</sub>.</p>

        <!-- ── Understanding Transformation A & Kernel Matrix K ──── -->
        <h2>Transformation Matrix A and Kernel Matrix K</h2>

        <h3>Transformation Matrix A</h3>
        <p>Matrix A maps the original kernel-based representation into the new feature space: X → K → A → Z. If m components are selected, A = [a<sub>1</sub>, a<sub>2</sub>, …, a<sub>m</sub>].</p>

        <h3>Kernel Matrix K</h3>
        <p>K represents relationships among all source and target samples:</p>
        <div class="formula">
        K = [ K<sub>SS</sub> &nbsp; K<sub>ST</sub> ]<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;[ K<sub>TS</sub> &nbsp; K<sub>TT</sub> ]
        </div>
        <p>Using a kernel function allows JDA to model non-linear relationships without explicitly computing high-dimensional mappings.</p>

        <!-- ── Understanding the Centering Matrix H ──────────────── -->
        <h2>Centering Matrix H</h2>
        <p>The centering matrix is:</p>
        <div class="formula">
        H = I − (1 / (n<sub>s</sub> + n<sub>t</sub>)) · 𝟏𝟏<sup>T</sup>
        </div>
        <p>The constraint <strong>A<sup>T</sup> K H K A = I</strong> prevents the learned representation from collapsing into a trivial all-zero solution.</p>

        <!-- ── Objective in Simple Words & Trade-Off Parameter μ ─── -->
        <h2>Trade-Off Parameter μ</h2>
        <p>A trade-off parameter μ can control the relative contribution of marginal and conditional adaptation:</p>
        <div class="formula">
        M = M<sub>0</sub> + μ · Σ<sub>c=1</sub><sup>C</sup> M<sub>c</sub>
        </div>
        <p>When μ = 0, JDA reduces to marginal-only adaptation (similar to TCA). As μ increases, conditional class alignment receives more emphasis.</p>

        <!-- ── Eigenvalue Decomposition ──────────────────────────── -->
        <h2>From Objective to Eigenvalue Problem</h2>
        <p>The JDA optimization problem can be converted into a generalized eigenvalue problem:</p>
        <div class="formula">
        (K M K + λ I)<sup>−1</sup> K H K
        </div>
        <p>where λ I is a regularization term that ensures numerical stability. The leading m eigenvectors are chosen to form A = [a<sub>1</sub>, a<sub>2</sub>, …, a<sub>m</sub>].</p>

        <!-- ── Complete JDA Workflow ─────────────────────────────── -->
        <h2>Complete JDA Workflow</h2>
        <ol>
            <li><strong>Step 1:</strong> Prepare source data D<sub>S</sub> (with labels) and target data D<sub>T</sub> (unlabeled).</li>
            <li><strong>Step 2:</strong> Combine source and target feature samples.</li>
            <li><strong>Step 3:</strong> Construct kernel matrix K.</li>
            <li><strong>Step 4:</strong> Construct marginal discrepancy matrix M<sub>0</sub>.</li>
            <li><strong>Step 5:</strong> Obtain initial target pseudo-labels using a base classifier.</li>
            <li><strong>Step 6:</strong> Construct conditional discrepancy matrices M<sub>c</sub> for each class c.</li>
            <li><strong>Step 7:</strong> Construct joint discrepancy matrix M = M<sub>0</sub> + Σ<sub>c</sub> M<sub>c</sub>.</li>
            <li><strong>Step 8:</strong> Solve generalized eigenvalue problem: (K M K + λ I)<sup>−1</sup> K H K.</li>
            <li><strong>Step 9:</strong> Select leading m eigenvectors to form transformation matrix A.</li>
            <li><strong>Step 10:</strong> Project both domains: X<sub>S</sub> → Z<sub>S</sub>, X<sub>T</sub> → Z<sub>T</sub>.</li>
            <li><strong>Step 11:</strong> Train classifier on adapted source data (Z<sub>S</sub>, Y<sub>S</sub>).</li>
            <li><strong>Step 12:</strong> Predict target labels Ŷ<sub>T</sub> (updated pseudo-labels).</li>
            <li><strong>Step 13:</strong> Repeat Steps 6–12 for a set number of iterations or until convergence.</li>
        </ol>

        <!-- ── JDA Workflow at a Glance ──────────────────────────── -->
        <h2>JDA Workflow at a Glance</h2>
        <div class="card">
        Source + Target data → Kernel K → Marginal M<sub>0</sub> → Initial target labels → Conditional M<sub>c</sub> → Joint M = M<sub>0</sub> + ΣM<sub>c</sub> → Eigenproblem → Eigenvectors A → Transform data → Train classifier → Predict target pseudo-labels → Update labels → Repeat
        </div>

        <div class="formula">
        Feature adaptation ↔ Target pseudo-labeling
        </div>

        <!-- ── A Simple Example ──────────────────────────────────── -->
        <h2>A Simple Example</h2>
        <p>Consider a software defect prediction problem with source project (100 labeled instances, 20 features) and target project (200 unlabeled instances, 20 features):</p>
        <ul>
            <li><strong>Stage 1:</strong> Align marginal distributions P<sub>S</sub>(X) and P<sub>T</sub>(X).</li>
            <li><strong>Stage 2:</strong> Base classifier generates initial pseudo-labels for target samples (e.g. x<sub>1</sub> → 0, x<sub>2</sub> → 1).</li>
            <li><strong>Stage 3:</strong> Align class-conditional distributions P<sub>S</sub>(Y | X) and P<sub>T</sub>(Y | X) using the pseudo-labels.</li>
            <li><strong>Stage 4:</strong> Learn improved representation, retrain classifier, update pseudo-labels, and iterate.</li>
        </ul>

        <!-- ── TCA vs. JDA ───────────────────────────────────────── -->
        <h2>TCA vs. JDA in Conceptual Terms</h2>
        <div class="formula">
        <strong>TCA:</strong> Align Marginal Distribution P(X)<br/><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/><br/>
        <strong>JDA:</strong> Align Marginal P(X) + Conditional P(Y | X) → Joint Distribution P(X, Y)
        </div>

        <p>TCA focuses on reducing difference between feature distributions. JDA extends this by also considering class-conditional structure, making both overall domains and class boundaries compatible.</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>If you remember only one thing about JDA:</strong><br/><br/>
        JDA learns a common feature space by aligning both the <strong>marginal distribution of the features</strong> and the <strong>conditional distribution of the classes</strong> between the source and target domains.
        </div>

        <div class="formula">
        Marginal + Conditional = Joint Distribution Adaptation
        </div>

        <div class="card">
        <strong>One-line definition:</strong> JDA extends distribution adaptation by aligning not only where the source and target data are located, but also how their samples are associated with different classes.
        </div>
        """,
),
]

__all__ =["JDA_CONTENT"]
