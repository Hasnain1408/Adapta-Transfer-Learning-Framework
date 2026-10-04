"""Complete learning materials for Feature Normalization & Standardization.

This is a self-contained module that provides in-depth educational content
for every normalization and scaling transform available in the application.
It is imported by learning_materials_content.py and rendered inside the
Learning Materials dialog under the "Normalization & Scaling" tab.
"""

from __future__ import annotations 

NORMALIZATION_CONTENT :list [tuple [str ,str ]]=[
# ── 1. Min-Max Normalization ──────────────────────────────────────────
(
"Min-Max Normalization",
"""
        <span class="badge">Bounded Feature Rescaling [0, 1]</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Min-Max Normalization</strong> (also known as feature scaling or unity-based normalization) is a linear transformation technique that rescales features from their original numeric domain into a strictly bounded interval, typically <strong>[0, 1]</strong> or <strong>[−1, 1]</strong>.</p>

        <p>In machine learning and transfer learning pipelines, features often have vastly different measurement units (e.g. lines of code measured in thousands vs cyclomatic complexity measured in tens). Without normalization, gradient-based optimization and distance metrics are overwhelmingly dominated by features with large numerical magnitudes.</p>

        <div class="card">
        <strong>Central Principle:</strong> Linearly compress or stretch feature values so that the minimum value maps to 0 and the maximum value maps to 1, preserving exact relative distances between samples while bounding dynamic range.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>For a feature vector <em>x</em> with observed minimum x<sub>min</sub> and maximum x<sub>max</sub>:</p>

        <div class="formula">
        x' = (x − x<sub>min</sub>) / (x<sub>max</sub> − x<sub>min</sub>)
        </div>

        <p>To rescale to an arbitrary custom range [a, b] (such as [−1, 1] for neural networks with tanh activations):</p>
        <div class="formula">
        x' = a + [ (x − x<sub>min</sub>) · (b − a) ] / (x<sub>max</sub> − x<sub>min</sub>)
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Domain Adaptation</h2>
        <ul>
            <li><strong>Equalizing Cross-Domain Scales:</strong> When source and target domains have different physical sensor sensitivities or measurement protocols, Min-Max normalization aligns the numerical dynamic ranges into a common bounded [0, 1] spectrum.</li>
            <li><strong>Distance Metric Compatibility:</strong> Bounded features ensure that Euclidean distances in k-NN and kernel inner products in SVM/TCA receive proportional contributions from all dimensions.</li>
            <li><strong>Preserving Zero Entries:</strong> If the data is naturally non-negative with x<sub>min</sub> = 0, Min-Max normalization preserves sparsity (zero values remain zero).</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Guarantees strictly bounded values with fixed upper and lower limits.</li>
            <li>Preserves the exact relative spacing and linear relationships between data points.</li>
            <li>Intuitive, non-destructive, and computationally efficient (O(n) pass).</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li><strong>Highly Sensitive to Outliers:</strong> If a single extreme outlier exists (e.g. x<sub>max</sub> = 10,000 while all normal values are between 1 and 10), the normal data points will be compressed into a microscopic interval near 0 (e.g. [0.0001, 0.001]), destroying discriminatory variance.</li>
        </ul>
        </div>
        """,
),

# ── 2. Z-score Standardization ────────────────────────────────────────
(
"Z-score Standardization",
"""
        <span class="badge">Zero Mean, Unit Variance (StandardScaler)</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Z-score Standardization</strong> (implemented via scikit-learn's <code>StandardScaler</code>) is the most widely adopted preprocessing technique in scientific computing and statistical machine learning. It transforms a feature so that its empirical distribution has a <strong>mean of 0 (μ = 0)</strong> and a <strong>standard deviation of 1 (σ = 1)</strong>.</p>

        <p>Unlike Min-Max normalization, Z-score standardization does not bind data to a fixed minimum and maximum. Instead, it measures how many standard deviations each data point lies above or below the mean.</p>

        <div class="card">
        <strong>Central Principle:</strong> Center the data cloud at the origin (zero mean) and scale variance along all axes to unity (unit variance), ensuring scale-invariant optimization across all features.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>For a feature vector <em>x</em> with sample mean μ and standard deviation σ:</p>

        <div class="formula">
        μ = (1 / n) Σ<sub>i=1</sub><sup>n</sup> x<sub>i</sub><br/><br/>
        σ = √( (1 / n) Σ<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> − μ)<sup>2</sup> )<br/><br/>
        x' = (x − μ) / σ
        </div>

        <p>After transformation: E[x'] = 0 &nbsp; and &nbsp; Var(x') = 1.</p>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Standard Practice in Transfer Learning (TCA, JDA, BDA)</h2>
        <p>In classical domain adaptation research, Z-score standardization is universally applied as a standard best practice:</p>
        <ul>
            <li><strong>Kernel Matrix Conditioning:</strong> Methods like TCA, JDA, BDA, and MEDA compute kernel matrices K (e.g. RBF kernel exp(−γ ‖x<sub>i</sub> − x<sub>j</sub>‖<sup>2</sup>)). If features have uneven variances, the RBF kernel exponent is dominated by high-variance features. Standardizing variance ensures isotropic kernel coverage.</li>
            <li><strong>Combined Joint Scaling:</strong> In standard transfer learning protocols, source and target data are stacked [X<sub>S</sub>; X<sub>T</sub>] and standardized jointly so both domains share a unified coordinate reference frame before distribution alignment.</li>
            <li><strong>Hyperplane Optimization:</strong> SVM and Logistic Regression rely on L2 regularization penalties ½ ‖w‖<sup>2</sup>. If features are unstandardized, the penalty disproportionately shrinks weights of smaller-scale features regardless of their true predictive importance.</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Much less sensitive to moderate outliers than Min-Max scaling.</li>
            <li>Converts features into standard normal score units (z-scores).</li>
            <li>Accelerates gradient descent and improves numerical conditioning of covariance and Hessian matrices.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Does not produce bounded intervals (values can range from −∞ to +∞).</li>
            <li>Destroys sparsity in sparse matrices by subtracting the non-zero mean μ.</li>
        </ul>
        </div>
        """,
),

# ── 3. Robust Standardization ─────────────────────────────────────────
(
"Robust Standardization",
"""
        <span class="badge">Outlier-Resistant Scaling (RobustScaler)</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>Standard Z-score standardization relies heavily on the sample mean μ and standard deviation σ. Unfortunately, both mean and variance have a breakdown point of 0%: a single severe outlier or corrupted measurement can arbitrarily pull the mean and inflate the standard deviation, distorting the entire standardized distribution.</p>

        <p><strong>Robust Standardization</strong> (implemented via scikit-learn's <code>RobustScaler</code>) eliminates this vulnerability by replacing parametric moments (mean and variance) with robust order statistics: the <strong>Median</strong> and the <strong>Interquartile Range (IQR)</strong>.</p>

        <div class="card">
        <strong>Central Principle:</strong> Center data using the median (50th percentile) and scale using the interquartile range (75th − 25th percentile), making the transformation virtually immune to extreme outliers and heavy-tailed distributions.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>Given a feature vector <em>x</em>, compute its quartiles:</p>
        <ul>
            <li><strong>Q<sub>1</sub>(x):</strong> 25th percentile (first quartile).</li>
            <li><strong>Q<sub>2</sub>(x):</strong> 50th percentile (the Median).</li>
            <li><strong>Q<sub>3</sub>(x):</strong> 75th percentile (third quartile).</li>
            <li><strong>IQR(x):</strong> Interquartile Range = Q<sub>3</sub>(x) − Q<sub>1</sub>(x) (capturing the middle 50% of the distribution).</li>
        </ul>

        <div class="formula">
        x' = ( x − Q<sub>2</sub>(x) ) / ( Q<sub>3</sub>(x) − Q<sub>1</sub>(x) ) = ( x − Median ) / IQR
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning & Defect Prediction</h2>
        <ul>
            <li><strong>Software Engineering Metric Datasets:</strong> Software metrics (such as lines of code, cyclomatic complexity, Halstead effort in NASA, PROMISE, or AEEEM datasets) exhibit extreme right-skewness and heavy tails, with rare massive files having thousands of lines of code. RobustScaler scales normal files accurately without letting mega-files skew the entire feature space.</li>
            <li><strong>Cross-Domain Sensor Spikes:</strong> In IoT and WiFi localization transfer learning, sensor anomalies produce sporadic extreme readings. Robust standardization ensures these anomalies do not corrupt the domain adaptation projection matrix.</li>
        </ul>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Exceptionally resistant to outliers and heavy-tailed skewness.</li>
            <li>Middle 50% of the data is scaled to a standard unit interval [−0.5, +0.5].</li>
            <li>Outliers retain their relative extremity without distorting inliers.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Does not bound data to a fixed range.</li>
            <li>If the middle 50% of data has identical discrete values (IQR = 0), division by zero must be guarded against.</li>
        </ul>
        </div>
        """,
),

# ── 4. L1 Normalization ───────────────────────────────────────────────
(
"L1 Normalization",
"""
        <span class="badge">Sample-wise Manhattan Vector Scaling</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>While standardization scales each feature column across samples, <strong>L1 Normalization</strong> operates <strong>sample-wise (row-wise)</strong> across features. It scales each individual sample vector independently so that the sum of the absolute values of its features equals <strong>1.0</strong>.</p>

        <p>This transforms each instance vector into a unit vector under the L1 norm (Manhattan / Taxicab distance).</p>

        <div class="card">
        <strong>Central Principle:</strong> Rescale individual sample rows so that ‖x‖<sub>1</sub> = Σ |x<sub>j</sub>| = 1.0. For non-negative features, this converts feature values into a discrete probability distribution (simplex).
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>For a sample row vector x = [x<sub>1</sub>, x<sub>2</sub>, …, x<sub>d</sub>]:</p>

        <div class="formula">
        ‖x‖<sub>1</sub> = Σ<sub>j=1</sub><sup>d</sup> |x<sub>j</sub>|<br/><br/>
        x' = x / ‖x‖<sub>1</sub> = [ x<sub>1</sub> / ‖x‖<sub>1</sub>, &nbsp; x<sub>2</sub> / ‖x‖<sub>1</sub>, &nbsp; …, &nbsp; x<sub>d</sub> / ‖x‖<sub>1</sub> ]
        </div>

        <!-- ── Role in Transfer Learning ──────────────────────────── -->
        <h2>Role in Transfer Learning</h2>
        <ul>
            <li><strong>Text Classification & Bag-of-Words:</strong> In cross-domain NLP (e.g. Amazon reviews, 20 Newsgroups), document lengths vary widely. L1 normalization converts raw term frequency counts into Term Frequencies (TF), turning documents into comparable probability distributions over vocabulary words regardless of document length.</li>
            <li><strong>Compositional & Histogram Data:</strong> When features represent color histograms or frequency spectra across domains, L1 normalization ensures that total histogram volume is identical (area = 1.0), isolating shape discrepancies from intensity discrepancies.</li>
        </ul>
        """,
),

# ── 5. L2 (Unit Vector) Normalization ─────────────────────────────────
(
"L2 (Unit Vector) Normalization",
"""
        <span class="badge">Row-wise Euclidean Hypersphere Projection</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>L2 Normalization</strong> (also called Euclidean unit vector normalization or spherical projection) rescales each sample vector independently so that its Euclidean length (L2 norm) equals <strong>1.0</strong>. Geometrically, this projects every data point onto the surface of a unit hypersphere in ℝ<sup>d</sup>.</p>

        <p>In visual domain adaptation, L2 row normalization is one of the most critical preprocessing steps. State-of-the-art algorithms—including <strong>JGSA, DALI, and MEDA</strong>—explicitly require row-wise L2 normalization of feature vectors before computing domain adaptation matrices.</p>

        <div class="card">
        <strong>Central Principle:</strong> Project every sample vector onto a unit sphere (‖x‖<sub>2</sub> = 1.0), eliminating sample magnitude differences and focusing exclusively on feature vector orientation and directional angles.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>For a sample row vector x = [x<sub>1</sub>, x<sub>2</sub>, …, x<sub>d</sub>]:</p>

        <div class="formula">
        ‖x‖<sub>2</sub> = √( Σ<sub>j=1</sub><sup>d</sup> x<sub>j</sub><sup>2</sup> )<br/><br/>
        x' = x / ‖x‖<sub>2</sub> = [ x<sub>1</sub> / ‖x‖<sub>2</sub>, &nbsp; x<sub>2</sub> / ‖x‖<sub>2</sub>, &nbsp; …, &nbsp; x<sub>d</sub> / ‖x‖<sub>2</sub> ]
        </div>

        <!-- ── Equivalence of Euclidean Distance and Cosine Similarity -->
        <h2>Cosine Similarity Equivalence</h2>
        <p>For any two L2-normalized unit vectors <em>u</em> and <em>v</em> (where ‖u‖<sub>2</sub> = ‖v‖<sub>2</sub> = 1):</p>

        <div class="formula">
        ‖u − v‖<sub>2</sub><sup>2</sup> = ‖u‖<sup>2</sup> + ‖v‖<sup>2</sup> − 2 (u · v) = 2 − 2 · cos(θ)
        </div>

        <p>This remarkable property means that on unit-normalized data, <strong>Euclidean distance is strictly monotonic to Cosine distance</strong>. Classifiers relying on Euclidean distance (like k-NN or RBF SVM) automatically measure angular similarity, eliminating distortions caused by sample brightness or document length.</p>

        <!-- ── Application in DALI & JGSA ────────────────────────── -->
        <h2>Mandatory Role in DALI & JGSA</h2>
        <p>In visual domain adaptation (e.g. Office-Caltech, Office-Home, PIE datasets), Deep CNN embeddings (ResNet, DeCAF, SURF) have varying vector magnitudes depending on image contrast and illumination. Applying <code>_l2_normalize_rows(X)</code> standardizes all visual samples to unit length, enabling MMD distribution alignment to operate purely on semantic direction.</p>
        """,
),

# ── 6. Mean Centering ─────────────────────────────────────────────────
(
"Mean Centering",
"""
        <span class="badge">Origin Translation (Zero Mean)</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Mean Centering</strong> subtracts the average value from data points without scaling their variance. It translates the center of mass of the data cloud to the origin (zero vector) while preserving the original units, variances, and relative distances between all samples.</p>

        <p>It can be applied in two directions:</p>
        <ul>
            <li><strong>Column Mean Centering (Feature Centering):</strong> Subtracts each feature's mean across all samples: x'<sub>ij</sub> = x<sub>ij</sub> − μ<sub>j</sub>.</li>
            <li><strong>Row Mean Centering (Sample Centering):</strong> Subtracts each sample's mean across all features: x'<sub>ij</sub> = x<sub>ij</sub> − μ<sub>i</sub>.</li>
        </ul>

        <!-- ── Role in Transfer Learning & Linear Algebra ────────── -->
        <h2>Role in PCA, Covariance & CORAL</h2>
        <ul>
            <li><strong>Unbiased Covariance Computation:</strong> The sample covariance matrix is defined as C = (1/(n−1)) (X − μ)<sup>T</sup> (X − μ). Only after column mean centering can covariance be computed as the simple matrix product C = (1/(n−1)) X<sup>T</sup> X.</li>
            <li><strong>CORAL Foundation:</strong> CORAL strictly requires source and target features to be zero-centered before computing and aligning covariance matrices C<sub>S</sub> and C<sub>T</sub>.</li>
            <li><strong>Preserves Natural Variance:</strong> Unlike Z-score standardization, mean centering does not divide by σ, making it suitable when features with high natural variance should legitimately carry more weight in subsequent analysis.</li>
        </ul>
        """,
),

# ── 7. Column-wise & Row-wise Unit Norm ────────────────────────────────
(
"Column-wise & Row-wise Unit Norm",
"""
        <span class="badge">Directional Axis Normalization</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>In matrix-based machine learning, data can be normalized along either dimension of the data matrix X ∈ ℝ<sup>n × d</sup>:</p>

        <h3>1. Column-wise Unit Norm (Feature-wise Normalization)</h3>
        <p>Normalizes each feature column independently across all <em>n</em> samples so that the Euclidean length of the column equals 1.0:</p>
        <div class="formula">
        X'<sub>:, j</sub> = X<sub>:, j</sub> / ‖X<sub>:, j</sub>‖<sub>2</sub> = X<sub>:, j</sub> / √( Σ<sub>i=1</sub><sup>n</sup> X<sub>ij</sub><sup>2</sup> )
        </div>
        <p>Ensures that no single feature column dominates matrix inner products due to sheer number of samples or large total energy.</p>

        <h3>2. Row-wise Unit Norm (Sample-wise Normalization)</h3>
        <p>Normalizes each sample row independently across all <em>d</em> features so that ‖X<sub>i, :</sub>‖<sub>2</sub> = 1.0, projecting individual instances to the unit hypersphere.</p>
        """,
),

# ── 8. Frobenius Normalization ────────────────────────────────────────
(
"Frobenius Normalization",
"""
        <span class="badge">Total Matrix Energy Scaling</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>The <strong>Frobenius Norm</strong> (also known as the Hilbert–Schmidt norm) is the matrix generalization of the vector Euclidean norm. It measures the total magnitude or "energy" contained across all entries of a matrix.</p>

        <p><strong>Frobenius Normalization</strong> divides an entire matrix X ∈ ℝ<sup>n × d</sup> by its Frobenius norm, scaling the overall matrix so that its total squared sum equals <strong>1.0</strong>.</p>

        <div class="card">
        <strong>Central Principle:</strong> Scale an entire matrix globally by ‖X‖<sub>F</sub> so that the total Frobenius norm of the normalized matrix is strictly 1.0, preserving all relative row and column relationships while bounding total matrix scale.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <div class="formula">
        ‖X‖<sub>F</sub> = √( Σ<sub>i=1</sub><sup>n</sup> Σ<sub>j=1</sub><sup>d</sup> X<sub>ij</sub><sup>2</sup> ) = √( tr(X<sup>T</sup> X) )<br/><br/>
        X' = X / ‖X‖<sub>F</sub>
        </div>

        <!-- ── Role in Transfer Learning Algorithms ──────────────── -->
        <h2>Critical Role in Transfer Learning (DALI, MEDA)</h2>
        <p>In modern multi-objective domain adaptation frameworks (such as <strong>DALI</strong> in <code>src/infrastructure/methods/dali_method.py</code>), the objective function sums multiple diverse matrix terms: MMD matrix M, within-class scatter S<sub>w</sub>, between-class scatter S<sub>b</sub>, and Graph Laplacian L.</p>

        <p>If one matrix has an unnormalized magnitude of 10,000 while another is 0.01, the trade-off hyperparameters (β, δ, ε) become unpredictable. DALI explicitly normalizes matrices by their Frobenius norm:</p>
        <div class="formula">
        M = M / ‖M‖<sub>F</sub> &nbsp;&nbsp; and &nbsp;&nbsp; S<sub>b</sub> = S<sub>b</sub> / ‖S<sub>b</sub>‖<sub>F</sub>
        </div>
        <p>This guarantees that every matrix term operates on a balanced, well-conditioned numerical scale.</p>
        """,
),

# ── 9. Spectral Normalization ─────────────────────────────────────────
(
"Spectral Normalization",
"""
        <span class="badge">Matrix Operator 2-Norm & Lipschitz Bounding</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Spectral Normalization</strong> scales a matrix by its <strong>spectral norm (matrix operator 2-norm)</strong>, which equals its largest singular value <strong>σ<sub>max</sub>(X)</strong>. It enforces a strict mathematical bound on the maximum factor by which the matrix can amplify any vector.</p>

        <div class="card">
        <strong>Central Principle:</strong> Enforce 1-Lipschitz continuity: scaling matrix X by σ<sub>max</sub>(X) guarantees that for any vector <em>v</em>, ‖X' v‖<sub>2</sub> ≤ ‖v‖<sub>2</sub>, preventing vector explosion during repeated matrix projections.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>The spectral norm of a matrix X ∈ ℝ<sup>n × d</sup> is the square root of the maximum eigenvalue of X<sup>T</sup> X:</p>

        <div class="formula">
        ‖X‖<sub>2</sub> = σ<sub>max</sub>(X) = max<sub>v ≠ 0</sub> ( ‖X v‖<sub>2</sub> / ‖v‖<sub>2</sub> )<br/><br/>
        X' = X / σ<sub>max</sub>(X)
        </div>

        <!-- ── Role in Machine Learning & Adaptation ─────────────── -->
        <h2>Role in Modern Learning & Transfer Stability</h2>
        <ul>
            <li><strong>Preventing Numerical Explosion:</strong> In iterative transfer learning algorithms (e.g. OCMTL's alternating SVD loops or deep domain adaptation networks), unnormalized matrix transformations can compound across iterations, causing singular values to explode or collapse.</li>
            <li><strong>Bounded Projection Operators:</strong> Spectral normalization ensures that feature representations remain strictly bounded on the unit Lipschitz ball.</li>
        </ul>
        """,
),

# ── 10. Mahalanobis Standardization & Whitening ───────────────────────
(
"Mahalanobis Standardization",
"""
        <span class="badge">Full Covariance Decorrelation & Whitening</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p>Standard Z-score standardization scales each feature independently, ignoring the off-diagonal correlations between features. If two features are 99% correlated, standard scaling preserves that collinearity.</p>

        <p><strong>Mahalanobis Standardization</strong> (also known as <strong>Multivariate Whitening</strong>) transforms the entire data cloud using the inverse square root of the empirical covariance matrix. It simultaneously centers the data, scales variances to 1, and <strong>completely eliminates all linear correlations between features</strong>.</p>

        <div class="card">
        <strong>Central Principle:</strong> Transform data into an uncorrelated, spherical coordinate space where the empirical covariance matrix equals the Identity Matrix (Cov(X') = I), transforming Mahalanobis distances into simple Euclidean distances.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <p>Given centered data X with empirical sample covariance matrix Σ ∈ ℝ<sup>d × d</sup>:</p>
        <div class="formula">
        Σ = (1 / (n − 1)) · X<sup>T</sup> X
        </div>

        <p>Perform Singular Value Decomposition (SVD) on Σ:</p>
        <div class="formula">
        Σ = U · Λ · U<sup>T</sup>
        </div>

        <p>The inverse square root whitening transformation matrix is:</p>
        <div class="formula">
        Σ<sup>−1/2</sup> = U · (Λ + ε I)<sup>−1/2</sup> · U<sup>T</sup><br/><br/>
        X<sub>whitened</sub> = (X − μ) · Σ<sup>−1/2</sup>
        </div>
        <p>where ε ≈ 10<sup>−10</sup> guards against singular or collinear dimensions.</p>

        <!-- ── The Connection to CORAL ───────────────────────────── -->
        <h2>Direct Connection to CORAL Domain Adaptation</h2>
        <p>Mahalanobis standardization is the exact theoretical engine of <strong>CORAL (Correlation Alignment)</strong>:</p>
        <div class="formula">
        X<sub>S</sub><sup>aligned</sup> = [ X<sub>S</sub> · C<sub>S</sub><sup>−1/2</sup> ] · C<sub>T</sub><sup>1/2</sup>
        </div>
        <p>The first bracket [ X<sub>S</sub> · C<sub>S</sub><sup>−1/2</sup> ] is precisely Mahalanobis standardization of the source domain! It whitens the source data into an isotropic spherical distribution before re-coloring it with target correlations.</p>

        <!-- ── Strengths & Limitations ───────────────────────────── -->
        <h2>Strengths & Limitations</h2>
        <div class="card">
        <strong>Strengths:</strong><br/>
        <ul>
            <li>Completely removes multicollinearity and redundant feature dependencies.</li>
            <li>Produces a spherical covariance matrix (Cov = I).</li>
            <li>Fundamental to statistical distance metrics and domain alignment.</li>
        </ul>
        <strong>Limitations:</strong><br/>
        <ul>
            <li>Requires computing and inverting a d × d covariance matrix (cost O(d<sup>3</sup>)).</li>
            <li>Requires regularized pseudo-inverses when feature count exceeds sample count (d &gt; n).</li>
        </ul>
        </div>
        """,
),

# ── 11. Variance Scaling ──────────────────────────────────────────────
(
"Variance Scaling",
"""
        <span class="badge">Dispersion Scaling without Centering</span>

        <!-- ── Introduction ──────────────────────────────────────── -->
        <h2>Introduction</h2>
        <p><strong>Variance Scaling</strong> (implemented via <code>StandardScaler(with_mean=False)</code>) divides each feature by its sample standard deviation σ without subtracting the mean μ.</p>

        <div class="card">
        <strong>Central Principle:</strong> Equalize the variance of all features to 1.0 while leaving the empirical mean untouched, preserving natural data sparsity and origin references.
        </div>

        <!-- ── Mathematical Formulation ──────────────────────────── -->
        <h2>Mathematical Formulation</h2>
        <div class="formula">
        σ<sub>j</sub> = √( (1 / n) Σ<sub>i=1</sub><sup>n</sup> x<sub>ij</sub><sup>2</sup> − μ<sub>j</sub><sup>2</sup> )<br/><br/>
        x'<sub>ij</sub> = x<sub>ij</sub> / σ<sub>j</sub>
        </div>

        <!-- ── When to Use ───────────────────────────────────────── -->
        <h2>When to Use Variance Scaling</h2>
        <ul>
            <li><strong>Sparse Data Matrices:</strong> In sparse text matrices (TF-IDF) or high-dimensional bag-of-words datasets where 99% of entries are zero, subtracting the mean μ fills all zeros with non-zero values, destroying sparsity and causing memory consumption to explode. Variance scaling preserves all zero entries while standardizing dispersion.</li>
            <li><strong>Non-Negative Representations:</strong> When physical features must remain strictly positive (e.g. pixel intensities, chemical concentrations).</li>
        </ul>
        """,
),
]

__all__ =["NORMALIZATION_CONTENT"]
