"""Complete learning material for k-Nearest Neighbors (k-NN) classifier.

This is a self-contained module that provides the full k-NN educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "k-NN (k-Nearest Neighbors)".
"""

from __future__ import annotations 

KNN_CONTENT :list [tuple [str ,str ]]=[
(
"k-NN (k-Nearest Neighbors)",
"""
        <span class="badge">Instance-Based Non-Parametric Classifier</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>The <strong>k-Nearest Neighbors (k-NN)</strong> algorithm is one of the most fundamental and intuitive supervised machine learning algorithms. Classified as an <strong>instance-based</strong> or <strong>lazy learning</strong> method, k-NN does not construct an explicit internal mathematical model or learn weights during training. Instead, it stores the entire training dataset and postpones all generalization computation until a query sample needs to be classified.</p>

        <p>In transfer learning and domain adaptation, k-NN plays a pivotal role. Many landmark domain adaptation methods—including TCA, JDA, BDA, and DALI—rely on 1-NN or 5-NN classifiers as their primary benchmark evaluation classifier and as the core engine for generating iterative target domain pseudo-labels.</p>

        <div class="card">
        <strong>Central Principle:</strong> Samples that are close to each other in feature space share the same class label. The prediction for an unannotated query sample is determined by the majority vote of its <em>k</em> nearest neighbors in the feature space.
        </div>

        <!-- ── How k-NN Works ────────────────────────────────────── -->
        <h2>How k-NN Works: The Core Mechanism</h2>
        <p>Given a labeled source dataset D<sub>S</sub> = {(x<sub>1</sub>, y<sub>1</sub>), …, (x<sub>n</sub>, y<sub>n</sub>)} where x<sub>i</sub> ∈ ℝ<sup>d</sup> and y<sub>i</sub> ∈ {1, …, C}, and an unannotated query target point <strong>x*</strong>:</p>

        <ol>
            <li><strong>Distance Calculation:</strong> Compute the distance between query point x* and every training instance x<sub>i</sub> in the dataset.</li>
            <li><strong>Neighbor Identification:</strong> Sort all calculated distances in ascending order and select the <em>k</em> training instances closest to x*. Let this set of neighbors be 𝒩<sub>k</sub>(x*).</li>
            <li><strong>Majority Voting:</strong> Tally the class labels of the <em>k</em> neighbors. The class with the highest frequency is assigned as the predicted label:
                <div class="formula">
                y* = argmax<sub>c ∈ {1, …, C}</sub> Σ<sub>x<sub>i</sub> ∈ 𝒩<sub>k</sub>(x*)</sub> 𝕀(y<sub>i</sub> = c)
                </div>
                where 𝕀(·) is the indicator function returning 1 if the condition is true and 0 otherwise.
            </li>
        </ol>

        <!-- ── Distance Metrics ──────────────────────────────────── -->
        <h2>Distance Metrics</h2>
        <p>The choice of distance metric defines what "similarity" means geometrically in the feature space:</p>

        <h3>1. Euclidean Distance (L2 Norm)</h3>
        <p>The standard straight-line distance, used as the default in most domain adaptation implementations:</p>
        <div class="formula">
        d<sub>Euclidean</sub>(x, z) = ‖x − z‖<sub>2</sub> = √( Σ<sub>j=1</sub><sup>d</sup> (x<sub>j</sub> − z<sub>j</sub>)<sup>2</sup> )
        </div>

        <h3>2. Manhattan Distance (L1 Norm)</h3>
        <p>The sum of absolute coordinate differences (grid or taxicab distance), preferred in high-dimensional or sparse feature vectors:</p>
        <div class="formula">
        d<sub>Manhattan</sub>(x, z) = ‖x − z‖<sub>1</sub> = Σ<sub>j=1</sub><sup>d</sup> |x<sub>j</sub> − z<sub>j</sub>|
        </div>

        <h3>3. Minkowski Distance (L<sub>p</sub> Norm)</h3>
        <p>The generalized metric family encompassing both L1 (p = 1) and L2 (p = 2):</p>
        <div class="formula">
        d<sub>Minkowski</sub>(x, z) = ( Σ<sub>j=1</sub><sup>d</sup> |x<sub>j</sub> − z<sub>j</sub>|<sup>p</sup> )<sup>1/p</sup>
        </div>

        <h3>4. Cosine Distance</h3>
        <p>Measures the angle between feature vectors rather than their magnitude, ideal for text document classification and normalized embeddings:</p>
        <div class="formula">
        d<sub>Cosine</sub>(x, z) = 1 − ( (x · z) / (‖x‖<sub>2</sub> · ‖z‖<sub>2</sub>) )
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Why k-NN is Central to Transfer Learning</h2>
        <p>k-NN occupies a unique, indispensable role across transfer learning research for three key reasons:</p>

        <ul>
            <li><strong>Direct Test of Manifold Alignment:</strong> Because k-NN relies purely on local metric neighborhood geometry without fitting parametric hyperplanes, its accuracy directly reflects how well the domain adaptation method aligned the source and target data clusters. If domain alignment succeeded, target points land in the neighborhood of corresponding source class points.</li>
            <li><strong>Pseudo-Label Generation:</strong> In methods like JDA, BDA, and DALI, conditional MMD adaptation requires initial target labels that are updated iteratively. Standard transfer learning protocols utilize a 1-NN classifier trained on source data to generate these pseudo-labels.</li>
            <li><strong>Non-Parametric Flexibility:</strong> It can model highly irregular, non-linear decision boundaries of arbitrary topological complexity without assumptions of Gaussian distributions or linear separability.</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters & Considerations</h2>

        <h3>1. The Value of k</h3>
        <ul>
            <li><strong>k = 1:</strong> Highly flexible, zero training error, but sensitive to label noise, outliers, and boundary overlap.</li>
            <li><strong>Small k (e.g. 3 or 5):</strong> Captures fine-grained local decision boundaries while smoothing out single-point noise. In this application, <strong>k = 5</strong> is standard.</li>
            <li><strong>Large k:</strong> Creates smoother, simpler decision boundaries, but risks diluting minority classes with majority class neighbors.</li>
        </ul>

        <h3>2. Feature Scaling & Preprocessing (Crucial!)</h3>
        <p>Because distance metrics treat coordinate differences equally, features with large numeric scales (e.g. salary in thousands) completely dominate features with small scales (e.g. age). <strong>Z-score standardization or Min-Max normalization is mandatory</strong> before running k-NN.</p>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>No assumptions about underlying data distributions (completely non-parametric).</li>
            <li>Naturally supports multi-class classification without one-vs-rest workarounds.</li>
            <li>Zero training time (trivial fit step).</li>
            <li>Effective in low-dimensional adapted subspaces produced by TCA, JDA, or BDA.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Inference time complexity O(n · d): computing distances to all training points is computationally demanding on large test datasets.</li>
            <li>Curse of dimensionality: in very high dimensional unadapted spaces, Euclidean distances concentrate and lose discriminatory power.</li>
            <li>Sensitive to irrelevant and uninformative feature noise.</li>
        </ul>
        </div>

        <!-- ── Implementation in This Application ────────────────── -->
        <h2>Implementation in This Studio</h2>
        <p>In this application's engine (<code>src/application/classification.py</code>), the k-NN classifier is initialized with:</p>
        <div class="formula">
        KNeighborsClassifier(n_neighbors=5)
        </div>
        <p>It is trained on the adapted source features Z<sub>S</sub> and used to predict target instances Z<sub>T</sub> across accuracy, precision, recall, F1, and multi-class macro metrics.</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> k-NN is the quintessential instance-based classifier. In domain adaptation, it serves as the ultimate litmus test: when cross-domain distributions are successfully aligned into a shared subspace, k-NN effortlessly classifies target samples by voting among their nearest source domain neighbors.
        </div>
        """,
),
]

__all__ =["KNN_CONTENT"]
