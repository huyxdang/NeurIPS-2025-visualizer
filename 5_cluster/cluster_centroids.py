"""
Meta-clustering: Group centroids into larger clusters
Takes the cluster centers from the first k-means and groups them into bigger meta-clusters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import argparse


def find_optimal_meta_k(centers_df, k_range=(3, 10), step=1):
    """
    Find optimal number of meta-clusters using silhouette score
    
    Args:
        centers_df: DataFrame with center_x and center_y columns
        k_range: Tuple of (min_k, max_k)
        step: Step size for k values
        
    Returns:
        Optimal k value
    """
    X = centers_df[['center_x', 'center_y']].values
    k_values = range(k_range[0], k_range[1] + 1, step)
    silhouettes = []
    
    print(f"\n🔍 Testing meta-cluster k values from {k_range[0]} to {k_range[1]}...")
    
    for k in k_values:
        if k >= len(centers_df):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
        print(f"   k={k}: silhouette={silhouettes[-1]:.3f}")
    
    if not silhouettes:
        return k_range[0]
    
    optimal_k = list(k_values)[np.argmax(silhouettes)]
    print(f"\n📊 Optimal meta-cluster k: {optimal_k}")
    print(f"   Silhouette score: {max(silhouettes):.3f}")
    
    return optimal_k


def cluster_centroids(
    centers_file: str = "5_cluster/data/cluster_centers.csv",
    papers_file: str = "5_cluster/data/neurips_2025_papers_with_clusters.parquet",
    output_file: str = "5_cluster/data/neurips_2025_papers_with_meta_clusters.parquet",
    n_meta_clusters: int = None,
    auto_k: bool = True
):
    """
    Cluster centroids into larger meta-clusters
    
    Args:
        centers_file: CSV file with cluster centers (center_x, center_y, cluster, size)
        papers_file: Parquet file with papers and cluster assignments
        output_file: Output parquet file with meta-cluster assignments
        n_meta_clusters: Number of meta-clusters (if None, auto-detect)
        auto_k: If True, automatically find optimal k
    """
    print("="*70)
    print("Meta-clustering: Grouping Centroids into Larger Clusters")
    print("="*70)
    
    # Load cluster centers
    centers_path = Path(centers_file)
    if not centers_path.exists():
        print(f"❌ File not found: {centers_file}")
        return
    
    print(f"\n📖 Loading cluster centers from {centers_file}...")
    centers_df = pd.read_csv(centers_path)
    print(f"✅ Loaded {len(centers_df)} cluster centers")
    print(f"   Columns: {list(centers_df.columns)}")
    
    # Check for required columns
    if 'center_x' not in centers_df.columns or 'center_y' not in centers_df.columns:
        print("❌ Error: cluster centers must have 'center_x' and 'center_y' columns")
        return
    
    # Load papers
    papers_path = Path(papers_file)
    if not papers_path.exists():
        print(f"❌ File not found: {papers_file}")
        return
    
    print(f"\n📖 Loading papers from {papers_file}...")
    papers_df = pd.read_parquet(papers_path)
    print(f"✅ Loaded {len(papers_df)} papers")
    
    # Check for cluster column
    if 'cluster' not in papers_df.columns:
        print("❌ Error: papers file must have 'cluster' column")
        return
    
    # Extract centroid coordinates
    X = centers_df[['center_x', 'center_y']].values
    print(f"\n📊 Centroid coordinate range:")
    print(f"   X: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"   Y: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # Determine number of meta-clusters
    if auto_k and n_meta_clusters is None:
        n_meta_clusters = find_optimal_meta_k(centers_df, k_range=(3, min(10, len(centers_df) - 1)), step=1)
    elif n_meta_clusters is None:
        # Default: roughly sqrt of number of clusters
        n_meta_clusters = max(3, int(np.sqrt(len(centers_df))))
        print(f"\n🎯 Using default meta-cluster k={n_meta_clusters}")
    else:
        print(f"\n🎯 Using specified meta-cluster k={n_meta_clusters}")
    
    # Perform K-means on centroids
    print(f"\n🚀 Running K-means on {len(centers_df)} centroids with {n_meta_clusters} meta-clusters...")
    kmeans = KMeans(
        n_clusters=n_meta_clusters,
        random_state=42,
        n_init=20,
        max_iter=300,
        verbose=0
    )
    
    meta_labels = kmeans.fit_predict(X)
    
    # Calculate quality metrics
    print("\n📊 Meta-clustering Quality Metrics:")
    silhouette = silhouette_score(X, meta_labels)
    print(f"   Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    
    # Add meta-cluster assignments to centers dataframe
    centers_df['meta_cluster'] = meta_labels
    
    # Create mapping from original cluster to meta-cluster
    cluster_to_meta = dict(zip(centers_df['cluster'], centers_df['meta_cluster']))
    
    # Add meta-cluster to papers dataframe
    papers_df['meta_cluster'] = papers_df['cluster'].map(cluster_to_meta)
    # Papers without clusters get -1
    papers_df['meta_cluster'] = papers_df['meta_cluster'].fillna(-1).astype(int)
    
    # Calculate statistics
    print(f"\n📈 Meta-cluster Statistics:")
    meta_cluster_sizes = papers_df[papers_df['meta_cluster'] >= 0]['meta_cluster'].value_counts().sort_index()
    print(f"   Number of meta-clusters: {n_meta_clusters}")
    print(f"   Mean size: {meta_cluster_sizes.mean():.1f} papers")
    print(f"   Median size: {meta_cluster_sizes.median():.1f} papers")
    print(f"   Min size: {meta_cluster_sizes.min()} papers (Meta-cluster {meta_cluster_sizes.idxmin()})")
    print(f"   Max size: {meta_cluster_sizes.max()} papers (Meta-cluster {meta_cluster_sizes.idxmax()})")
    
    # Show meta-cluster composition
    print(f"\n📊 Meta-cluster Composition:")
    print(f"{'Meta-Cluster':<15} {'Size':<8} {'Original Clusters':<50}")
    print("="*80)
    
    for meta_id in sorted(centers_df['meta_cluster'].unique()):
        meta_centers = centers_df[centers_df['meta_cluster'] == meta_id]
        original_clusters = sorted(meta_centers['cluster'].tolist())
        total_papers = meta_centers['size'].sum()
        clusters_str = ', '.join(map(str, original_clusters))
        if len(clusters_str) > 45:
            clusters_str = clusters_str[:42] + "..."
        print(f"{meta_id:<15} {total_papers:<8} {clusters_str}")
    
    # Save meta-cluster centers
    meta_centers_file = Path(output_file).with_name('meta_cluster_centers.csv')
    meta_centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['meta_center_x', 'meta_center_y']
    )
    meta_centers_df['meta_cluster'] = range(n_meta_clusters)
    meta_centers_df['size'] = [meta_cluster_sizes.get(i, 0) for i in range(n_meta_clusters)]
    meta_centers_df.to_csv(meta_centers_file, index=False)
    print(f"\n💾 Saved meta-cluster centers to {meta_centers_file}")
    
    # Save updated centers with meta-cluster assignments
    updated_centers_file = Path(output_file).with_name('cluster_centers_with_meta.csv')
    centers_df.to_csv(updated_centers_file, index=False)
    print(f"💾 Saved updated cluster centers to {updated_centers_file}")
    
    # Save main output
    output_path = Path(output_file)
    print(f"\n💾 Saving papers with meta-clusters to {output_file}...")
    papers_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    # File size info
    import os
    file_size_mb = os.path.getsize(output_path) / 1024**2
    print(f"✅ Saved {len(papers_df)} papers with meta-cluster assignments")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Visualization
    print(f"\n📊 Generating meta-cluster visualization...")
    create_meta_cluster_visualization(centers_df, kmeans.cluster_centers_, n_meta_clusters)
    
    print("\n" + "="*70)
    print("✅ Meta-clustering complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(papers_df)}")
    print(f"   - Papers with meta-clusters: {(papers_df['meta_cluster'] >= 0).sum()}")
    print(f"   - Number of original clusters: {len(centers_df)}")
    print(f"   - Number of meta-clusters: {n_meta_clusters}")


def create_meta_cluster_visualization(centers_df, meta_centers, n_meta_clusters):
    """Create visualization of meta-clusters"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for meta-clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_meta_clusters))
    
    # Plot each meta-cluster
    for meta_id in range(n_meta_clusters):
        meta_centers_data = centers_df[centers_df['meta_cluster'] == meta_id]
        
        # Plot original cluster centers
        ax.scatter(
            meta_centers_data['center_x'],
            meta_centers_data['center_y'],
            c=[colors[meta_id]],
            label=f'Meta-cluster {meta_id}',
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidths=1.5
        )
        
        # Add cluster ID labels
        for _, row in meta_centers_data.iterrows():
            ax.annotate(
                str(int(row['cluster'])),
                (row['center_x'], row['center_y']),
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
    
    # Plot meta-cluster centers
    ax.scatter(
        meta_centers[:, 0],
        meta_centers[:, 1],
        c='red',
        marker='X',
        s=300,
        edgecolors='white',
        linewidths=3,
        label='Meta-cluster Centers',
        zorder=1000
    )
    
    # Add meta-cluster labels at centers
    for i, (cx, cy) in enumerate(meta_centers):
        ax.annotate(
            f'M{i}',
            (cx, cy),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            bbox=dict(boxstyle='circle,pad=0.4', facecolor='red', alpha=0.8)
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Meta-clustering: Grouping Centroids into Larger Clusters', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    output_file = '5_cluster/meta_cluster_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved meta-cluster visualization to {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Meta-cluster centroids into larger groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal number of meta-clusters
  python cluster_centroids.py --auto-k
  
  # Specify number of meta-clusters
  python cluster_centroids.py --n-meta-clusters 5
  
  # Custom input/output
  python cluster_centroids.py \\
    --centers 5_cluster/data/cluster_centers.csv \\
    --papers 5_cluster/data/papers.parquet \\
    --output 5_cluster/data/output.parquet \\
    --n-meta-clusters 6

Recommended meta-cluster k values:
  - 20 original clusters: k=4-6 meta-clusters
  - 30 original clusters: k=5-7 meta-clusters
  - 50 original clusters: k=7-10 meta-clusters
        """
    )
    
    parser.add_argument('--centers', 
                       default='5_cluster/data/cluster_centers.csv',
                       help='Input CSV file with cluster centers')
    parser.add_argument('--papers',
                       default='5_cluster/data/neurips_2025_papers_with_clusters.parquet',
                       help='Input parquet file with papers and cluster assignments')
    parser.add_argument('--output', '-o',
                       default='5_cluster/data/neurips_2025_papers_with_meta_clusters.parquet',
                       help='Output parquet file with meta-cluster assignments')
    parser.add_argument('--n-meta-clusters', type=int, default=None,
                       help='Number of meta-clusters (if not auto-detecting)')
    parser.add_argument('--auto-k', action='store_true',
                       help='Automatically find optimal k using silhouette score')
    parser.add_argument('--no-auto-k', dest='auto_k', action='store_false',
                       help='Disable automatic k detection')
    parser.set_defaults(auto_k=True)
    
    args = parser.parse_args()
    
    cluster_centroids(
        centers_file=args.centers,
        papers_file=args.papers,
        output_file=args.output,
        n_meta_clusters=args.n_meta_clusters,
        auto_k=args.auto_k
    )

