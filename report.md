# CSE 546 Machine Learning - Homework 6 Report
**Name:** Toni  
**Date:** November 2025

## 1. Introduction
This analysis examines a 10,000-sample, 128-dimensional dataset with five ground-truth classes supplied for Homework 6. The objectives follow the assignment brief: (i) visualize class structure via PCA and t-SNE (Problem 1), (ii) perform K-Means on the original feature space (Problem 2), (iii) repeat clustering after PCA reduction to 20 dimensions (Problem 3), and (iv) compare clustering quality using ARI, silhouette, and runtime (Problem 4).

## 2. Experimental Setup
### 2.1 Data Preprocessing
The 128-dimensional feature vectors were standardized using `StandardScaler`, enforcing zero mean and unit variance per lecture guidance to stabilize covariance estimates before eigenvalue decomposition. No missing values were detected (`Any NaN: False`), and labels remain in {0,…,4}.

### 2.2 Methods
**PCA:** Principal Component Analysis retained the leading eigenvectors of the covariance matrix, with `n_components=2` for visualization and `n_components=20` for dimensionality reduction (random_state=42).  
**t-SNE:** The stochastic neighbor embedding used `perplexity=30`, `learning_rate='auto'`, `max_iter=1000`, and PCA initialization to respect local manifold structure while mitigating random initialization variance.  
**K-Means:** Clustering employed `init='k-means++'`, `n_init=10`, `max_iter=300`, and `random_state=42`. Continuous monitoring of inertia ensured convergence in line with lecture discussions on centroid updates.

## 3. Results and Analysis
### 3.1 Dimensionality Reduction Visualization (Problem 1)
**PCA Results:** As shown in Figure 1, the first two principal components explain 10.61% and 7.33% of the variance respectively (17.94% cumulative), consistent with the eigenvalue spectrum predicted by lecture derivations. Classes 1 and 3 exhibit the largest centroid separation (distance 8.49), while classes 0 and 4 nearly overlap (distance 1.43), reflecting limited between-class variance along leading eigenvectors. One percent of samples exceed the 5.87 radius threshold, indicating mild outliers within this linear subspace.

**t-SNE Results:** Figure 2 demonstrates tighter class islands because perplexity 30 balances local neighborhood preservation with global layout. The algorithm converged in 29.87 seconds, a cost aligned with lecture warnings about O(n²) affinity computations. Class 2 separates cleanly, whereas classes 0 and 4 still intermix, suggesting intrinsic similarity beyond linear components.

**Comparison:** Comparing Figures 1 and 2 highlights the PCA trade-off: faster eigenvalue decomposition but reduced separation due to the curse of dimensionality compressing variance into many directions. Figure 3 confirms t-SNE’s superior local structure preservation but at substantially higher runtime, supporting use for interpretation rather than iterative model tuning.

### 3.2 K-Means on Original Data (Problem 2)
Figure 4 shows the elbow in SSE flattening beyond k=5, matching both the five ground-truth classes and the lecture emphasis on minimizing within-cluster variance without overfitting. The silhouette profile in Figure 5 peaks near k=5 with an average score of 0.036, reinforcing that high dimensional sparsity (curse of dimensionality) impedes tight clusters despite theoretical convergence guarantees. Core samples for each cluster (indices 3856, 4013, 3337, 2801, 1468) achieve silhouettes up to 0.184, while boundary samples 1611 and 8292 register -0.102 and -0.101, illustrating how between-cluster variance is only marginally greater than within-cluster variance along certain axes.

### 3.3 K-Means on PCA-Reduced Data (Problem 3)
Reducing to 20 components retained 50.31% of total variance (Figure 6), mitigating noise dimensions identified in lecture discussions on bias-variance trade-offs. The elbow curve in Figure 7 again favors k=5, but Figure 8 reports a higher average silhouette of 0.110, signifying improved within-cluster cohesion after suppressing low-eigenvalue directions. This dimensionality reduction lessens the curse of dimensionality by concentrating energy in informative components while preserving convergence behavior of Lloyd’s updates.

### 3.4 Method Comparison (Problem 4)
Table 1 consolidates quantitative metrics, and Figure 9 visualizes the trade-offs. The original 128D model attains the highest ARI (0.697) but suffers the lowest silhouette (0.036) and longer runtime (0.073 s). The PCA-20D model slightly lowers ARI to 0.689 yet raises silhouette to 0.110 and cuts runtime to 0.021 s (71% faster). This reflects the lecture-framed bias-variance compromise: dimensionality reduction increases bias marginally but decreases variance and computational cost.

Table 1: Clustering Performance Comparison
| Method | ARI | Silhouette | Time(s) |
|--------|-----|------------|---------|
| Original (128D) | 0.697 | 0.036 | 0.073 |
| PCA (20D) | 0.689 | 0.110 | 0.021 |

## 4. Discussion
### 4.1 Key Findings
Figure 1 confirms that only 17.94% of variance sits in the first two eigen-directions, validating the need for more than 2D projections. Figure 5 reveals how high-dimensional sparsity blurs cluster margins, explaining the negative silhouettes of boundary samples 1611 and 8292. Figure 8 demonstrates that projecting into 20D enhances cohesion enough to elevate the silhouette by 0.074 while maintaining the elbow at k=5. Figure 9 and Table 1 together show that the modest ARI loss (0.008) under PCA comes with a substantial 71% runtime reduction, critical for iterative model selection.

### 4.2 Theoretical Insights
The curse of dimensionality manifests in the original-space silhouette of 0.036, because Euclidean distances converge as predicted in lecture derivations, limiting between-cluster variance gains. PCA’s eigenvalue decomposition reorders axes by energy, and the 50.31% retained variance confirms that the leading eigenvectors capture dominant structure while discarding noisy components with small eigenvalues. K-Means convergence criteria rely on monotonically decreasing inertia; the elbow at Figure 4 shows the inertia plateau after k=5, confirming convergence without unnecessary splits. Finally, the bias-variance trade-off appears between the two clustering pipelines: PCA introduces bias by discarding 49.69% of variance yet reduces estimator variance, leading to steadier silhouettes and faster execution.

## 5. Conclusion
Based on the quantitative evidence, the original 128-dimensional K-Means remains the recommended method when fidelity to ground-truth labels (ARI 0.697) is paramount, despite its slightly higher runtime. For exploratory workflows requiring faster iteration, the PCA-20D model offers a competitive ARI (within 0.008) and superior silhouette (0.110) while cutting computation time by 71%. Practitioners can therefore choose between higher accuracy or lower latency depending on downstream needs, but the original-space clustering serves as the benchmark for final reporting.

## Figure Captions
Figure 1: PCA projection of the standardized 128D data to PC1–PC2 showing 17.94% variance explained and highlighting class overlap between labels 0 and 4.  
Figure 2: t-SNE visualization (perplexity 30) illustrating improved local separation at the cost of a 29.87 s runtime.  
Figure 3: Side-by-side comparison of PCA and t-SNE embeddings emphasizing global vs. local structure trade-offs.  
Figure 4: Elbow curve for original-space K-Means with inertia leveling after k=5.  
Figure 5: Silhouette plot for original-space K-Means with average score 0.036 and negative boundary silhouettes.  
Figure 6: Cumulative variance retained by the first 20 principal components reaching 50.31%.  
Figure 7: Elbow curve on 20D PCA features confirming k=5 as the efficient cluster count.  
Figure 8: Silhouette plot on 20D PCA features with average score 0.110, evidencing tighter clusters.  
Figure 9: Bar charts comparing ARI, silhouette, and runtime between original and PCA pipelines.  
Figure 10: Combined visualization overlaying ground-truth colors and cluster markers (best method: original 128D) in PCA and t-SNE spaces.
