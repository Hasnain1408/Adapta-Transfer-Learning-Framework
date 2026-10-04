"""Complete learning material for Random Forest classifier.

This is a self-contained module that provides the full Random Forest
educational content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "Random Forest".
"""

from __future__ import annotations 

RANDOM_FOREST_CONTENT :list [tuple [str ,str ]]=[
(
"Random Forest",
"""
        <span class="badge">Ensemble Learning (Bagging & Random Subspaces)</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>A single decision tree has low bias but suffers from notoriously high variance: small variations in training data can lead to completely different tree structures and poor generalization on unseen test domains. <strong>Random Forest</strong>, introduced by Leo Breiman in 2001, is a powerful ensemble learning algorithm that dramatically reduces variance without increasing bias by combining hundreds of decorrelated decision trees.</p>

        <p>In transfer learning and domain adaptation, Random Forest is widely considered one of the highest-performing general-purpose classifiers. When source data is adapted to a target domain, residual domain discrepancy or slight distribution noise may remain. Random Forest's ensemble aggregation acts as a powerful buffer against this noise, yielding remarkably robust cross-domain classification.</p>

        <div class="card">
        <strong>Central Principle:</strong> Combine the predictions of a large collection of diverse, decorrelated decision trees grown on bootstrap samples of the data with randomized feature subsets, taking the majority vote for classification.
        </div>

        <!-- ── The Two Engines of Random Forest ──────────────────── -->
        <h2>The Dual Engines: Bagging & Random Subspaces</h2>
        <p>Random Forest achieves tree decorrelation and variance reduction through two distinct randomized mechanisms:</p>

        <h3>1. Bootstrap Aggregating (Bagging)</h3>
        <p>Given a source training dataset of size <em>n</em>, Random Forest constructs <strong>B</strong> independent bootstrap datasets by sampling <em>n</em> instances uniformly <strong>with replacement</strong>:</p>
        <ul>
            <li>On average, each bootstrap sample contains approximately <strong>63.2%</strong> of the original unique instances.</li>
            <li>The remaining <strong>36.8%</strong> of unpicked instances are known as <strong>Out-Of-Bag (OOB)</strong> samples, which serve as an internal validation set.</li>
        </ul>

        <h3>2. Random Feature Subspaces</h3>
        <p>Standard decision trees evaluate all <em>d</em> features at every split. If one feature is overwhelmingly predictive, all trees would choose it at the root node, resulting in highly correlated trees.</p>
        <p>Random Forest avoids this by considering only a small random subset of <strong>m ≈ √d</strong> features at each split. This forces trees to explore alternative, complementary feature relationships and decorrelates the individual tree predictors.</p>

        <!-- ── Mathematical Variance Reduction ───────────────────── -->
        <h2>The Mathematics of Variance Reduction</h2>
        <p>Why does averaging decorrelated trees work so effectively? Consider <em>B</em> random variables (individual tree predictions), each with variance σ<sup>2</sup> and positive pairwise correlation ρ. The variance of their ensemble average is:</p>

        <div class="formula">
        Var(Ensemble) = ρ · σ<sup>2</sup> + ( (1 − ρ) / B ) · σ<sup>2</sup>
        </div>

        <p>As the number of trees B grows large (e.g. B = 100), the second term ((1 − ρ)/B) σ<sup>2</sup> approaches zero. The remaining variance is bounded strictly by <strong>ρ · σ<sup>2</sup></strong>.</p>

        <div class="card">
        <strong>Key Mathematical Insight:</strong> By randomly restricting candidate features at every split, Random Forest actively reduces the inter-tree correlation <strong>ρ</strong>, driving down the overall ensemble variance far below what standard bagging can achieve alone.
        </div>

        <!-- ── Prediction & Probability Estimation ───────────────── -->
        <h2>Ensemble Aggregation & Probability Estimation</h2>
        <p>Given a query target sample <strong>x*</strong>, every tree T<sub>b</sub> in the forest casts an independent vote:</p>

        <h3>1. Majority Voting (Hard Prediction)</h3>
        <div class="formula">
        y* = mode { T<sub>1</sub>(x*), T<sub>2</sub>(x*), …, T<sub>B</sub>(x*) }
        </div>

        <h3>2. Soft Probability Estimation (for ROC-AUC)</h3>
        <p>Class probabilities are estimated by averaging the class proportions across all trees in the ensemble:</p>
        <div class="formula">
        P(Y = c | X = x*) = (1 / B) Σ<sub>b=1</sub><sup>B</sup> P<sub>b</sub>(Y = c | X = x*)
        </div>
        <p>This produces smooth, reliable class probabilities that are directly leveraged by this studio to calculate ROC-AUC scores.</p>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Domain Adaptation</h2>
        <ul>
            <li><strong>Resilience to Negative Transfer:</strong> If a domain adaptation method introduces minor distortion or residual feature shift in certain dimensions, single hyperplanes (like linear SVM) can be skewed. Random Forest averages over feature subspaces, naturally dampening the impact of distorted dimensions.</li>
            <li><strong>Capturing Non-Linear Interaction:</strong> Random Forest naturally captures high-order, non-linear interactions among adapted latent components without requiring manual kernel trick configuration.</li>
            <li><strong>Handling Feature Redundancy:</strong> Highly effective on tabular and software engineering datasets where features contain complex dependencies.</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters & Best Practices</h2>
        <ul>
            <li><strong>n_estimators:</strong> The number of trees in the forest. More trees always reduce variance without causing overfitting, reaching diminishing returns around 100–300 trees. In this application, <strong>n_estimators = 100</strong> provides optimal balance between accuracy and computational speed.</li>
            <li><strong>max_features:</strong> Number of features to consider per split. The theoretical default for classification is <strong>sqrt(d)</strong>.</li>
            <li><strong>max_depth:</strong> Maximum depth of individual trees. Default is None (nodes expanded until all leaves are pure).</li>
            <li><strong>random_state:</strong> Ensures deterministic, reproducible forest construction (set to 42 in this application).</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>One of the most accurate, versatile off-the-shelf classifiers available.</li>
            <li>Virtually immune to overfitting as the number of trees increases.</li>
            <li>Naturally produces well-calibrated class probabilities.</li>
            <li>Inherently robust to outliers and noisy features.</li>
            <li>Requires minimal feature scaling or hyperparameter tuning.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Ensemble structure creates a "black-box" model that is harder to interpret than a single decision tree.</li>
            <li>Larger memory footprint and higher inference cost than single linear models.</li>
        </ul>
        </div>

        <!-- ── Implementation in This Application ────────────────── -->
        <h2>Implementation in This Studio</h2>
        <p>In <code>src/application/classification.py</code>, Random Forest is constructed as:</p>
        <div class="formula">
        RandomForestClassifier(n_estimators=100, random_state=42)
        </div>
        <p>It fits 100 decorrelated trees on the adapted source features and evaluates on target instances, outputting multi-class probabilities for ROC-AUC scoring and consensus predictions for accuracy, precision, recall, and F1.</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> Random Forest harnesses the wisdom of crowds in machine learning. By combining bagging with random feature selection, it suppresses the high variance of individual decision trees, offering unmatched robustness against cross-domain noise and residual distribution discrepancies in transfer learning experiments.
        </div>
        """,
),
]

__all__ =["RANDOM_FOREST_CONTENT"]
