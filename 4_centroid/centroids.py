"""
K-means clustering on UMAP coordinates
Groups papers into semantic clusters based on their 2D positions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_optimal_k(X, k_range=(10, 50), step=2):
    """
    Find optimal number of clusters using elbow method and silhouette score
    
    Args:
        X: Coordinate array (n_samples, 2)
        k_range: Tuple of (min_k, max_k)
        step: Step size for k values
        
    Returns:
        Optimal k value
    """
    k_values = range(k_range[0], k_range[1] + 1, step)
    inertias = []
    silhouettes = []
    
    print(f"\n🔍 Testing k values from {k_range[0]} to {k_range[1]}...")
    
    for k in tqdm(k_values, desc="Finding optimal k"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(k_values, silhouettes, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_optimization.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved optimization plots to cluster_optimization.png")
    
    # Find optimal k (highest silhouette score)
    optimal_k = list(k_values)[np.argmax(silhouettes)]
    print(f"\n📊 Optimal k based on silhouette score: {optimal_k}")
    print(f"   Silhouette score: {max(silhouettes):.3f}")
    
    return optimal_k


def cluster_papers(
    input_file: str = "4_centroid/data/neurips_2025_papers_with_summaries.parquet",
    output_file: str = "4_centroid/data/neurips_2025_papers_with_clusters.parquet",
    n_clusters: int = None,
    auto_k: bool = True
):
    """
    Cluster papers using K-means on UMAP coordinates
    
    Args:
        input_file: Input parquet file with umap_x and umap_y columns
        output_file: Output parquet file with cluster assignments
        n_clusters: Number of clusters (if None, auto-detect)
        auto_k: If True, automatically find optimal k
    """
    print("="*70)
    print("K-means Clustering on UMAP Coordinates")
    print("="*70)
    
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"\n📖 Loading papers from {input_file}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df)} papers")
    
    # Check for UMAP coordinates
    if 'umap_x' not in df.columns or 'umap_y' not in df.columns:
        print("❌ Error: No UMAP coordinates found. Run UMAP first.")
        return
    
    # Extract coordinates
    valid_mask = df['umap_x'].notna() & df['umap_y'].notna()
    X = df.loc[valid_mask, ['umap_x', 'umap_y']].values
    
    print(f"📊 Papers with UMAP coordinates: {len(X)}/{len(df)}")
    print(f"   Coordinate range: X=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
          f"Y=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # Determine number of clusters
    if auto_k and n_clusters is None:
        n_clusters = find_optimal_k(X, k_range=(10, 50), step=2)
    elif n_clusters is None:
        # Default based on dataset size
        n_clusters = max(15, min(30, len(df) // 200))
        print(f"\n🎯 Using default k={n_clusters} (based on dataset size)")
    else:
        print(f"\n🎯 Using specified k={n_clusters}")
    
    # Perform K-means clustering
    print(f"\n🚀 Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,  # More initializations for stability
        max_iter=300,
        verbose=0
    )
    
    labels = kmeans.fit_predict(X)
    
    # Calculate quality metrics
    print("\n📊 Clustering Quality Metrics:")
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    print(f"   Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    print(f"   Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
    
    # Add cluster assignments to dataframe
    df['cluster'] = -1  # Default for papers without UMAP
    df.loc[valid_mask, 'cluster'] = labels
    
    # Calculate cluster statistics
    print(f"\n📈 Cluster Statistics:")
    cluster_sizes = df[df['cluster'] >= 0]['cluster'].value_counts().sort_index()
    print(f"   Mean size: {cluster_sizes.mean():.1f} papers")
    print(f"   Median size: {cluster_sizes.median():.1f} papers")
    print(f"   Min size: {cluster_sizes.min()} papers (Cluster {cluster_sizes.idxmin()})")
    print(f"   Max size: {cluster_sizes.max()} papers (Cluster {cluster_sizes.idxmax()})")
    
    # Show cluster size distribution
    print(f"\n📊 Cluster Size Distribution:")
    for cluster_id in sorted(cluster_sizes.index):
        size = cluster_sizes[cluster_id]
        bar = '█' * int(size / cluster_sizes.max() * 40)
        print(f"   Cluster {cluster_id:2d}: {size:4d} papers {bar}")
    
    # Store cluster centers
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['center_x', 'center_y']
    )
    cluster_centers['cluster'] = range(n_clusters)
    cluster_centers['size'] = [cluster_sizes.get(i, 0) for i in range(n_clusters)]
    
    # Save cluster centers separately
    centers_file = Path(output_file).with_name('cluster_centers.csv')
    cluster_centers.to_csv(centers_file, index=False)
    print(f"\n💾 Saved cluster centers to {centers_file}")
    
    # Save main output
    output_path = Path(output_file)
    print(f"\n💾 Saving to {output_file}...")
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    # File size info
    import os
    file_size_mb = os.path.getsize(output_path) / 1024**2
    print(f"✅ Saved {len(df)} papers with cluster assignments")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Visualization
    print(f"\n📊 Generating cluster visualization...")
    create_cluster_visualization(df, kmeans.cluster_centers_, n_clusters)
    
    print("\n" + "="*70)
    print("✅ Clustering complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(df)}")
    print(f"   - Papers with clusters: {(df['cluster'] >= 0).sum()}")
    print(f"   - Number of clusters: {n_clusters}")
    print(f"\n💡 Next: Run name_clusters.py to assign names to each cluster")


def create_cluster_visualization(df, centers, n_clusters):
    """Create visualization of clusters"""
    import matplotlib.pyplot as plt
    
    # Filter valid papers
    valid_df = df[df['cluster'] >= 0].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_data = valid_df[valid_df['cluster'] == cluster_id]
        ax.scatter(
            cluster_data['umap_x'],
            cluster_data['umap_y'],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=20
        )
    
    # Plot cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='black',
        marker='X',
        s=200,
        edgecolors='white',
        linewidths=2,
        label='Centers',
        zorder=1000
    )
    
    # Add cluster labels at centers
    for i, (cx, cy) in enumerate(centers):
        ax.annotate(
            str(i),
            (cx, cy),
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(f'NeurIPS 2025 Papers - K-means Clustering (k={n_clusters})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Legend (only show first 10 clusters to avoid clutter)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:min(11, len(handles))], labels[:min(11, len(labels))], 
             loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved cluster visualization to cluster_visualization.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cluster papers using K-means on UMAP coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal k
  python cluster_papers.py --auto-k
  
  # Specify number of clusters
  python cluster_papers.py --n-clusters 20
  
  # Custom input/output
  python cluster_papers.py --input data.parquet --output clustered.parquet --n-clusters 25

Recommended k values:
  - Small dataset (<1000): k=10-15
  - Medium dataset (1000-5000): k=15-25
  - Large dataset (5000+): k=20-40
        """
    )
    
    parser.add_argument('--input', default='4_centroid/data/neurips_2025_papers_with_summaries.parquet',
                       help='Input parquet file with UMAP coordinates')
    parser.add_argument('--output', default='4_centroid/data/neurips_2025_papers_with_clusters.parquet',
                       help='Output parquet file')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters (if not auto-detecting)')
    parser.add_argument('--auto-k', action='store_true',
                       help='Automatically find optimal k using silhouette score')
    parser.add_argument('--no-auto-k', dest='auto_k', action='store_false',
                       help='Disable automatic k detection')
    parser.set_defaults(auto_k=True)
    
    args = parser.parse_args()
    
    cluster_papers(
        input_file=args.input,
        output_file=args.output,
        n_clusters=args.n_clusters,
        auto_k=args.auto_k
    )