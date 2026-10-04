"""Complete learning material for Decision Tree classifier.

This is a self-contained module that provides the full Decision Tree
educational content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "Decision Tree".
"""

from __future__ import annotations 

DECISION_TREE_CONTENT :list [tuple [str ,str ]]=[
(
"Decision Tree",
"""
        <span class="badge">Hierarchical Rule-Based Classifier</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>A <strong>Decision Tree</strong> is a non-parametric supervised learning algorithm that performs classification by recursively partitioning the feature space into distinct, non-overlapping axis-aligned rectangular regions. Structured like an inverted tree, it starts at a single <strong>root node</strong>, branches through <strong>internal decision nodes</strong> that test individual feature thresholds, and concludes at <strong>leaf nodes</strong> that assign final class labels.</p>

        <p>In transfer learning research, Decision Trees serve as an intuitive "white-box" baseline. While individual trees are prone to high variance on raw unadapted data, testing a decision tree on adapted feature representations (from methods like TCA, BDA, or OCMTL) offers transparent diagnostic insight into whether domain adaptation has clustered classes into cleanly separable regions.</p>

        <div class="card">
        <strong>Central Principle:</strong> Recursively split data using simple if-then-else feature decision rules that maximize class purity at each step, forming a hierarchical tree of transparent logical decisions.
        </div>

        <!-- ── How Decision Trees Work ───────────────────────────── -->
        <h2>How a Decision Tree Works: Recursive Binary Splitting</h2>
        <p>Most modern implementations (including scikit-learn's CART algorithm) build binary trees through greedy top-down recursive partitioning:</p>

        <ol>
            <li><strong>Evaluate All Candidate Splits:</strong> For every feature <em>j</em> and every possible threshold value <em>t</em>, evaluate dividing the current node dataset D into two child subsets:
                <div class="formula">
                D<sub>left</sub> = { x ∈ D | x<sub>j</sub> ≤ t } &nbsp;&nbsp; and &nbsp;&nbsp; D<sub>right</sub> = { x ∈ D | x<sub>j</sub> &gt; t }
                </div>
            </li>
            <li><strong>Select the Best Split:</strong> Choose the feature <em>j*</em> and threshold <em>t*</em> that produces the greatest reduction in node impurity (maximum Information Gain).</li>
            <li><strong>Recurse:</strong> Apply the same splitting process independently to D<sub>left</sub> and D<sub>right</sub>.</li>
            <li><strong>Terminate:</strong> Stop splitting a node when a stopping criterion is met (e.g. node is pure, max depth reached, or sample count falls below minimum). Designate the terminal node as a leaf.</li>
        </ol>

        <!-- ── Splitting Criteria: Gini Impurity & Entropy ───────── -->
        <h2>Node Impurity Criteria</h2>
        <p>The quality of a split is quantified by how "pure" the resulting child subsets are (i.e. whether they contain samples from only one class):</p>

        <h3>1. Gini Impurity (Default CART Metric)</h3>
        <p>Measures the probability that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the class distribution in the subset:</p>

        <div class="formula">
        Gini(D) = 1 − Σ<sub>c=1</sub><sup>C</sup> p<sub>c</sub><sup>2</sup>
        </div>
        <p>where p<sub>c</sub> is the proportion of instances in node D belonging to class <em>c</em>. Gini = 0 represents a perfectly pure node (all samples belong to a single class); maximum impurity occurs when classes are equally distributed.</p>

        <h3>2. Shannon Entropy & Information Gain</h3>
        <p>Derived from Information Theory, Entropy measures the disorder or unpredictability in a node:</p>

        <div class="formula">
        H(D) = − Σ<sub>c=1</sub><sup>C</sup> p<sub>c</sub> log<sub>2</sub>(p<sub>c</sub>)
        </div>

        <p>The <strong>Information Gain (IG)</strong> of a split on feature A is the reduction in entropy from parent to children:</p>
        <div class="formula">
        IG(D, A) = H(D) − [ (|D<sub>left</sub>| / |D|) H(D<sub>left</sub>) + (|D<sub>right</sub>| / |D|) H(D<sub>right</sub>) ]
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Domain Adaptation</h2>
        <ul>
            <li><strong>Evaluating Axis-Aligned Separability in Latent Spaces:</strong> Methods like TCA, JDA, and OCMTL produce orthogonal or decorrelated latent dimensions. Evaluating a Decision Tree on these adapted features demonstrates whether orthogonal latent axes correspond directly to clean class boundaries.</li>
            <li><strong>Sensitivity to Domain Shift:</strong> Decision trees split on exact absolute numerical thresholds (e.g. "feature<sub>3</sub> &le; 12.5"). If marginal feature distributions shift between source and target domains, a decision tree trained on source thresholds will misclassify target data severely. Hence, observing decision tree improvement after adaptation is strong evidence of successful distribution alignment.</li>
            <li><strong>Interpretability & Rule Extraction:</strong> Domain experts in defect prediction or medicine can directly inspect the extracted decision paths (e.g. "IF Halstead Volume &gt; 150 AND Cyclomatic Complexity &gt; 10 THEN Defective").</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters & Overfitting Control</h2>
        <p>An unconstrained decision tree will grow until every training point is memorized, resulting in severe overfitting. Common regularization constraints include:</p>

        <ul>
            <li><strong>max_depth:</strong> Maximum allowable depth from root to leaf. Restricting depth prevents the tree from capturing source-specific noise.</li>
            <li><strong>min_samples_split:</strong> Minimum number of samples required to split an internal node.</li>
            <li><strong>min_samples_leaf:</strong> Minimum number of samples required to be present at a leaf node (smooths decision boundaries).</li>
            <li><strong>criterion:</strong> Impurity measurement ('gini' or 'entropy').</li>
            <li><strong>random_state:</strong> Ensures deterministic, reproducible split selection when multiple splits achieve identical purity gain (set to 42 in this application).</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Completely white-box and interpretable: can be visualized as a flowchart of human-readable decision rules.</li>
            <li>Requires zero feature normalization or scaling (monotonic transformations do not affect split points).</li>
            <li>Handles mixed feature types (numerical and categorical) naturally.</li>
            <li>Fast training and logarithmic inference time O(log n).</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>High variance: small changes in source training data can alter the entire tree structure.</li>
            <li>Prone to overfitting if unpruned.</li>
            <li>Axis-aligned limitation: cannot easily represent diagonal decision boundaries without stepwise approximations (mitigated by ensemble methods like Random Forest).</li>
        </ul>
        </div>

        <!-- ── Implementation in This Application ────────────────── -->
        <h2>Implementation in This Studio</h2>
        <p>In <code>src/application/classification.py</code>, the decision tree is initialized as:</p>
        <div class="formula">
        DecisionTreeClassifier(random_state=42)
        </div>
        <p>It provides an unconstrained decision tree benchmark, comparing directly against regularized linear models (Logistic Regression, SVM) and ensemble models (Random Forest).</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> The Decision Tree is the foundational rule-based classifier. While sensitive to domain shift when trained on unadapted data, its transparent recursive splitting provides an invaluable diagnostic window into how cleanly domain adaptation methods have separated classes in the transferred subspace.
        </div>
        """,
),
]

__all__ =["DECISION_TREE_CONTENT"]
