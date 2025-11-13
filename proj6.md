# Project 6: ML Clustering and Dimensionality Reduction

## Project Overview

This project implements and compares clustering algorithms (K-Means) on high-dimensional data (128D) with dimensionality reduction techniques (PCA, t-SNE) for visualization and performance analysis.

## Data Structure

```
Data Files:
├── X.csv          # Feature matrix (n_samples × 128)
├── y.csv          # Ground truth labels (n_samples × 1, values: 0-4)
├── images.csv     # Original images (n_samples × image_pixels)
└── readme         # Data description
```

## Project Structure

```
homework6/
├── homework6.ipynb           # Main Jupyter notebook
├── figures/                  # Output directory for all plots
│   ├── problem1/            # PCA and t-SNE visualizations
│   ├── problem2/            # Original data clustering
│   ├── problem3/            # PCA-reduced clustering
│   └── extra_credit/        # Combined visualizations
├── results/                 # Numerical results and tables
└── report/                  # Final submission files
```

## Implementation Architecture

### Core Components

#### 1. Data Loader Module

```python
class DataLoader:
    - load_data(): Load X, y, images from CSV
    - preprocess(): Standardize features
    - train_test_split(): Optional splitting
```

#### 2. Dimensionality Reduction Module

```python
class DimensionalityReducer:
    - pca_reduce(n_components): PCA reduction
    - tsne_reduce(n_components): t-SNE reduction
    - plot_2d(): Visualization helper
```

#### 3. Clustering Module

```python
class ClusteringAnalysis:
    - kmeans_range(k_range): Test multiple k values
    - calculate_sse(): Sum of squared errors
    - silhouette_analysis(): Silhouette coefficients
    - find_optimal_k(): Elbow method
```

#### 4. Evaluation Module

```python
class ClusterEvaluator:
    - adjusted_rand_index(): ARI calculation
    - silhouette_score(): Average silhouette
    - sample_extraction(): Core/boundary samples
```

#### 5. Visualization Module

```python
class Visualizer:
    - plot_elbow(): SSE vs k plot
    - plot_silhouette(): Silhouette plot
    - plot_samples(): Image grid display
    - combined_plot(): Ground truth + clusters
```

## Key Implementation Details

### Problem 1: Dimensionality Reduction

```python
# PCA Implementation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE Implementation
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Visualization
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10')
```

### Problem 2: K-Means Clustering

```python
# Elbow Method
sse_scores = []
k_range = range(2, 21)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse_scores.append(kmeans.inertia_)

# Silhouette Analysis
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
```

### Problem 3: PCA + K-Means

```python
# PCA to 20D
pca_20 = PCA(n_components=20)
X_pca_20 = pca_20.fit_transform(X_scaled)

# Apply K-Means to reduced data
kmeans_pca = KMeans(n_clusters=optimal_k)
labels_pca = kmeans_pca.fit_predict(X_pca_20)
```

### Problem 4: Comparison Metrics

```python
# Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
ari_original = adjusted_rand_score(y_true, labels_original)
ari_pca = adjusted_rand_score(y_true, labels_pca)

# Comparison table
comparison_df = pd.DataFrame({
    'Method': ['Original (128D)', 'PCA (20D)'],
    'ARI': [ari_original, ari_pca],
    'Silhouette': [sil_original, sil_pca],
    'Time (s)': [time_original, time_pca]
})
```

## Algorithm Parameters

### PCA Settings

* **2D Visualization** : n_components=2
* **Dimension Reduction** : n_components=20
* **Scaling** : StandardScaler() before PCA

### t-SNE Settings

* **n_components** : 2
* **perplexity** : 30 (adjust based on dataset size)
* **learning_rate** : 'auto'
* **n_iter** : 1000
* **random_state** : 42 (for reproducibility)

### K-Means Settings

* **n_clusters** : 2-20 (range for testing)
* **init** : 'k-means++'
* **n_init** : 10
* **max_iter** : 300
* **random_state** : 42

## Expected Results

### Problem 1

* PCA: Linear projection, faster, preserves global structure
* t-SNE: Non-linear, better cluster separation, local structure

### Problem 2

* Optimal k likely between 4-7 (close to 5 ground truth classes)
* Silhouette score > 0.3 indicates reasonable clustering

### Problem 3

* 20D PCA should retain >80% variance
* Clustering might be faster but potentially less accurate

### Problem 4

* ARI closer to 1.0 is better
* Best method depends on ARI, silhouette, and computational efficiency

## Performance Considerations

### Memory Usage

* Original data: ~8.5 MB
* Images data: ~312 MB (load only when needed)
* Use sparse matrices if applicable

### Computation Time

* t-SNE: O(n²) - may be slow for large datasets
* K-Means: O(n·k·i·d) where i=iterations, d=dimensions
* PCA: O(min(n², d²))

### Optimization Tips

1. Use `n_jobs=-1` for parallel processing where available
2. Cache intermediate results
3. Use vectorized operations
4. Consider mini-batch K-Means for large datasets

## Visualization Best Practices

### Figure Requirements

1. **Resolution** : 300 DPI for publication quality
2. **Size** : Figure size (10, 6) for single plots
3. **Labels** : Clear axis labels and titles
4. **Legend** : Always include for colored plots
5. **Color** : Use colorblind-friendly palettes

### Plot Styling

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
```

## Common Issues and Solutions

### Issue 1: Memory Error with Images

 **Solution** : Load images in batches or only load selected samples

### Issue 2: t-SNE Takes Too Long

 **Solution** :

* Reduce perplexity
* Use PCA to reduce to 50D first
* Use subset for initial testing

### Issue 3: Poor Clustering Results

 **Solution** :

* Ensure data is scaled
* Try different random seeds
* Increase n_init for K-Means

### Issue 4: Silhouette Plot Issues

 **Solution** :

* Check for correct cluster labels
* Ensure enough samples per cluster
* Handle edge cases (k=2)

## Code Quality Guidelines

### Documentation

* Docstrings for all functions
* Inline comments for complex logic
* Markdown cells explaining approach

### Error Handling

```python
try:
    result = complex_operation()
except Exception as e:
    print(f"Error in operation: {e}")
    # Fallback or alternative approach
```

### Reproducibility

* Set random seeds
* Document library versions
* Save intermediate results

## Testing Strategy

### Unit Tests

1. Test data loading
2. Test dimension reduction
3. Test clustering with known data
4. Test evaluation metrics

### Integration Tests

1. Full pipeline execution
2. Figure generation
3. Report compilation

### Validation

1. Check ARI bounds [−1, 1]
2. Verify silhouette scores [−1, 1]
3. Ensure SSE decreases with k

## Deliverables Checklist

### Code

* [X] Clean, commented notebook
* [X] All cells execute without errors
* [X] Reproducible results

### Figures (numbered)

* [ ] Figure 1: PCA 2D visualization
* [ ] Figure 2: t-SNE 2D visualization
* [ ] Figure 3: PCA vs t-SNE comparison
* [ ] Figure 4: Elbow plot (original)
* [ ] Figure 5: Silhouette plot (original)
* [ ] Figure 6: Sample images grid
* [ ] Figure 7: PCA variance explained
* [ ] Figure 8: Elbow plot (PCA-20D)
* [ ] Figure 9: Silhouette plot (PCA-20D)
* [ ] Figure 10: Method comparison table

### Report Sections

* [ ] Introduction (0.5 page)
* [ ] Methods (1-2 pages)
* [ ] Results (4-5 pages with figures)
* [ ] Discussion (1-2 pages)
* [ ] Conclusion (0.5 page)

### Recording Components

* [ ] Code execution demo
* [ ] Results explanation
* [ ] Method comparison
* [ ] Best approach justification

## Resources and References

### Key sklearn Documentation

* [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
* [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [Silhouette](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
* [ARI](https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index)

### Concepts to Reference from Lectures

1. Curse of dimensionality
2. PCA eigenvalue decomposition
3. t-SNE perplexity parameter
4. K-Means convergence criteria
5. Silhouette coefficient interpretation
6. Cluster validation metrics

## Session Log

### 2025-11-11

* Updated notebook data loader to use `pathlib.Path`, auto-detect `dataset_hwk5` layout, and validate dtype/NaN status.
* Centralized figure/result directory creation for all problems to maintain consistent output structure.
* Implemented PCA 2D projection workflow with reproducible settings and high-DPI saving to `figures/problem1/pca_2d_visualization.png`.
* Added automated PCA analysis summary that quantifies variance coverage, class separability, compactness, and potential outliers based on computed metrics.
* Hardened data loading to search common relative and absolute directories (including Downloads) for the HWK5 dataset to prevent `FileNotFoundError` during notebook execution.
* Added support for `dataset_hwk6` folder naming and optional `HW6_DATA_DIR` environment override so the loader adapts to alternate course distributions.
* Executed notebook end-to-end after updating t-SNE (`max_iter`) for scikit-learn ≥1.5 compatibility and confirmed outputs saved to `figures/` hierarchy.