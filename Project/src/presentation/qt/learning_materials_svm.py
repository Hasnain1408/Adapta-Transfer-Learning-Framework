"""Complete learning material for Support Vector Machine (SVM) classifier.

This is a self-contained module that provides the full SVM educational
content as a single (title, html_body) pair. It is imported by
learning_materials_content.py and rendered inside the Learning
Materials dialog under the sidebar entry "SVM (Support Vector Machine)".
"""

from __future__ import annotations 

SVM_CONTENT :list [tuple [str ,str ]]=[
(
"SVM (Support Vector Machine)",
"""
        <span class="badge">Maximum Margin Classifier & Kernel Machines</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Support Vector Machines (SVM)</strong>, developed by Vladimir Vapnik and colleagues within Statistical Learning Theory, represent one of the most mathematically rigorous and powerful classification frameworks in modern machine learning. Rather than simply finding <em>any</em> boundary that separates classes, SVM identifies the <strong>optimal maximum-margin hyperplane</strong> that leaves the widest possible safety buffer between distinct classes.</p>

        <p>In transfer learning and domain adaptation, SVM is exceptionally influential. The maximum-margin principle directly minimizes structural risk, providing built-in geometric resilience against cross-domain distribution shifts: target samples that drift slightly from source training points are far less likely to cross a wide-margin decision boundary than a narrow one.</p>

        <div class="card">
        <strong>Central Principle:</strong> Construct a decision hyperplane in a high-dimensional feature space that maximizes the geometric distance (margin) to the nearest training instances of any class (known as <em>support vectors</em>).
        </div>

        <!-- ── The Concept of Margin & Support Vectors ───────────── -->
        <h2>The Maximum Margin Principle</h2>
        <p>For a binary classification problem with training points (x<sub>i</sub>, y<sub>i</sub>) where y<sub>i</sub> ∈ {−1, +1}, a separating hyperplane is defined by:</p>
        <div class="formula">
        w<sup>T</sup> x + b = 0
        </div>
        <p>where <strong>w ∈ ℝ<sup>d</sup></strong> is the normal vector perpendicular to the hyperplane and <strong>b ∈ ℝ</strong> is the bias offset.</p>

        <p>The geometric distance from the hyperplane to any point x<sub>i</sub> is |w<sup>T</sup> x<sub>i</sub> + b| / ‖w‖. For linearly separable data with normalized boundaries w<sup>T</sup> x + b = ±1, the total margin width between the two bounding planes is:</p>

        <div class="formula">
        Margin Width = 2 / ‖w‖<sub>2</sub>
        </div>

        <div class="card">
        <strong>Optimization Insight:</strong> Maximizing the margin width (2 / ‖w‖) is mathematically equivalent to minimizing the squared Euclidean norm <strong>½ ‖w‖<sup>2</sup></strong> subject to correct classification constraints.
        </div>

        <p>The training points that lie exactly on the marginal hyperplanes (w<sup>T</sup> x<sub>i</sub> + b = ±1) are called <strong>Support Vectors</strong>. All other training points can be removed or moved without altering the final decision boundary.</p>

        <!-- ── Soft-Margin SVM & Regularization Parameter C ──────── -->
        <h2>Soft-Margin SVM & Slack Variables</h2>
        <p>Real-world datasets are rarely linearly separable. To tolerate noise, outliers, and overlapping class distributions, Cortes and Vapnik (1995) introduced <strong>slack variables ξ<sub>i</sub> ≥ 0</strong>, creating the <strong>Soft-Margin SVM</strong>:</p>

        <div class="formula">
        min<sub>w, b, ξ</sub> &nbsp; ½ ‖w‖<sup>2</sup> + C · Σ<sub>i=1</sub><sup>n</sup> ξ<sub>i</sub><br/><br/>
        subject to: &nbsp; y<sub>i</sub>(w<sup>T</sup> x<sub>i</sub> + b) ≥ 1 − ξ<sub>i</sub>, &nbsp;&nbsp; ξ<sub>i</sub> ≥ 0 &nbsp; ∀ i
        </div>

        <p>Where:</p>
        <ul>
            <li><strong>½ ‖w‖<sup>2</sup>:</strong> Margin maximization objective (promotes wider margins and simpler models).</li>
            <li><strong>Σ ξ<sub>i</sub>:</strong> Penalty for margin violations and classification errors.</li>
            <li><strong>C (Regularization Parameter):</strong> Governs the fundamental trade-off:
                <ul>
                    <li><strong>Large C:</strong> High penalty for errors; enforces a narrow margin to strictly separate every training point (risks overfitting).</li>
                    <li><strong>Small C:</strong> Tolerates more margin violations; prioritizes a wider margin and smoother decision boundary (better generalization across domains).</li>
                </ul>
            </li>
        </ul>

        <!-- ── The Dual Formulation & Support Vectors ────────────── -->
        <h2>The Dual Formulation</h2>
        <p>Applying the method of Lagrange multipliers transforms the primal optimization into its Wolfe Dual:</p>

        <div class="formula">
        max<sub>α</sub> &nbsp; Σ<sub>i=1</sub><sup>n</sup> α<sub>i</sub> − ½ Σ<sub>i=1</sub><sup>n</sup> Σ<sub>j=1</sub><sup>n</sup> α<sub>i</sub> α<sub>j</sub> y<sub>i</sub> y<sub>j</sub> (x<sub>i</sub><sup>T</sup> x<sub>j</sub>)<br/><br/>
        subject to: &nbsp; 0 ≤ α<sub>i</sub> ≤ C, &nbsp;&nbsp; Σ<sub>i=1</sub><sup>n</sup> α<sub>i</sub> y<sub>i</sub> = 0
        </div>

        <p>Here, α<sub>i</sub> are the Lagrange multipliers. Most α<sub>i</sub> evaluate to exactly zero. Points with <strong>α<sub>i</sub> &gt; 0</strong> are the critical <strong>Support Vectors</strong>.</p>

        <p>The final classification decision function depends solely on the support vectors and inner products:</p>
        <div class="formula">
        f(x) = sign( Σ<sub>i ∈ SV</sub> α<sub>i</sub> y<sub>i</sub> (x<sub>i</sub><sup>T</sup> x) + b )
        </div>

        <!-- ── The Kernel Trick: Non-Linear SVM ─────────────────── -->
        <h2>The Kernel Trick: Non-Linear SVM</h2>
        <p>Because the dual formulation depends exclusively on dot products x<sub>i</sub><sup>T</sup> x<sub>j</sub>, non-linear classification can be achieved without computing explicit high-dimensional feature mappings φ(x). By substituting a <strong>Kernel Function K(x<sub>i</sub>, x<sub>j</sub>) = φ(x<sub>i</sub>)<sup>T</sup> φ(x<sub>j</sub>)</strong> (Mercer's Theorem), data is implicitly mapped into an infinite-dimensional Hilbert space where complex non-linear boundaries become linearly separable:</p>

        <h3>1. Radial Basis Function (RBF / Gaussian Kernel) — Default</h3>
        <div class="formula">
        K<sub>RBF</sub>(x<sub>i</sub>, x<sub>j</sub>) = exp( −γ · ‖x<sub>i</sub> − x<sub>j</sub>‖<sup>2</sup> )
        </div>
        <p>Where <strong>γ (gamma)</strong> defines the influence radius of individual support vectors. Default γ = "scale" sets γ = 1 / (d · Var(X)).</p>

        <h3>2. Linear Kernel</h3>
        <div class="formula">
        K<sub>Linear</sub>(x<sub>i</sub>, x<sub>j</sub>) = x<sub>i</sub><sup>T</sup> x<sub>j</sub>
        </div>
        <p>Extremely fast; ideal when feature dimensionality is high or after manifold dimensionality alignment.</p>

        <h3>3. Polynomial Kernel</h3>
        <div class="formula">
        K<sub>Poly</sub>(x<sub>i</sub>, x<sub>j</sub>) = ( x<sub>i</sub><sup>T</sup> x<sub>j</sub> + c )<sup>d</sup>
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Domain Adaptation</h2>
        <ul>
            <li><strong>Geometric Margin Buffer:</strong> Transfer learning aims to align domains, but perfect alignment is rarely achieved. A wide-margin SVM decision boundary provides a geometric buffer that tolerates residual target domain shifts without misclassifying samples.</li>
            <li><strong>Dual Representation Compatibility:</strong> Kernel SVM uses the exact same RKHS principles as kernel-based transfer methods (such as TCA, JDA, and MEDA), providing theoretical cohesion.</li>
            <li><strong>Continuous Decision Scores for ROC-AUC:</strong> The unthresholded distance to the hyperplane f(x) = w<sup>T</sup> φ(x) + b serves as an exact continuous ranking score, enabling precise ROC-AUC calculation without probability calibration overhead.</li>
        </ul>

        <!-- ── Key Hyperparameters ───────────────────────────────── -->
        <h2>Key Hyperparameters & Optimization</h2>
        <ul>
            <li><strong>C (Regularization):</strong> Default C = 1.0. Balance between margin width and training error tolerance.</li>
            <li><strong>kernel:</strong> 'rbf' is standard for non-linear feature structures; 'linear' is used for high-dimensional or pre-aligned spaces.</li>
            <li><strong>gamma:</strong> RBF bandwidth. "scale" automatically adapts to feature variance.</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Exceptional generalization capability rooted in Structural Risk Minimization.</li>
            <li>Effective in high-dimensional spaces, even where feature count exceeds sample count.</li>
            <li>Memory efficient: decision boundary depends strictly on a subset of training points (support vectors).</li>
            <li>Versatile: custom kernel functions can be plugged in for specific data structures.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Training scales between O(n<sup>2</sup>) and O(n<sup>3</sup>) with sample count, making unaccelerated exact SVM slow on very massive datasets (millions of rows).</li>
            <li>Sensitive to feature scale: requires feature standardization or normalization.</li>
        </ul>
        </div>

        <!-- ── Implementation in This Application ────────────────── -->
        <h2>Implementation in This Studio</h2>
        <p>In <code>src/application/classification.py</code>, SVM is configured as:</p>
        <div class="formula">
        SVC(kernel="rbf", C=1.0, gamma="scale")
        </div>
        <p>It utilizes the decision function directly for ROC-AUC scoring, avoiding expensive probability calibration while delivering maximum-margin classification on the adapted target features.</p>

        <!-- ── Summary ───────────────────────────────────────────── -->
        <h2>Summary & Key Takeaway</h2>
        <div class="card">
        <strong>Key Takeaway:</strong> Support Vector Machines optimize the structural margin of separation rather than just empirical training error. In transfer learning studio pipelines, SVM's wide decision margins and versatile RBF kernel provide an exceptional defense against residual cross-domain distribution shifts.
        </div>
        """,
),
]

__all__ =["SVM_CONTENT"]
