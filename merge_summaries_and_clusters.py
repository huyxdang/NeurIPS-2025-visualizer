"""
Merge Summaries and Cluster Names
Combines:
- Summaries (problem, solution, eli5) from 3_features/data/neurips_2025_papers_with_summaries.parquet
- Cluster info (cluster, cluster_name, cluster_description) from 4_cluster/data/neurips_2025_papers_with_cluster_names.parquet

Output: A combined parquet file with all columns
"""

import pandas as pd
from pathlib import Path
import argparse

def merge_summaries_and_clusters(
    summaries_file: str = "3_features/data/neurips_2025_papers_with_summaries.parquet",
    clusters_file: str = "4_cluster/data/neurips_2025_papers_with_cluster_names.parquet",
    output_file: str = "neurips_2025_papers_complete.parquet"
):
    """
    Merge summaries and cluster information into a single parquet file
    
    Args:
        summaries_file: Path to parquet file with summaries (problem, solution, eli5)
        clusters_file: Path to parquet file with cluster assignments and names
        output_file: Path for the merged output file
    """
    print("="*70)
    print("Merging Summaries and Cluster Information")
    print("="*70)
    
    # Load summaries
    print(f"\n📖 Loading summaries from {summaries_file}...")
    summaries_path = Path(summaries_file)
    if not summaries_path.exists():
        print(f"❌ File not found: {summaries_file}")
        return False
    
    df_summaries = pd.read_parquet(summaries_path)
    print(f"✅ Loaded {len(df_summaries)} papers with summaries")
    print(f"   Columns: {list(df_summaries.columns)}")
    
    # Load cluster names
    print(f"\n📖 Loading cluster information from {clusters_file}...")
    clusters_path = Path(clusters_file)
    if not clusters_path.exists():
        print(f"❌ File not found: {clusters_file}")
        return False
    
    df_clusters = pd.read_parquet(clusters_path)
    print(f"✅ Loaded {len(df_clusters)} papers with cluster information")
    print(f"   Columns: {list(df_clusters.columns)}")
    
    # Check for common key
    if 'paper_id' in df_summaries.columns and 'paper_id' in df_clusters.columns:
        merge_key = 'paper_id'
    elif 'paper' in df_summaries.columns and 'paper' in df_clusters.columns:
        merge_key = 'paper'
    else:
        print("❌ Error: No common key found for merging (need 'paper_id' or 'paper')")
        return False
    
    print(f"\n🔗 Merging on key: '{merge_key}'...")
    
    # Select only cluster-related columns from clusters file to avoid duplicates
    cluster_cols = ['cluster', 'cluster_name', 'cluster_description']
    if merge_key not in cluster_cols:
        cluster_cols = [merge_key] + cluster_cols
    
    # Merge
    df_merged = df_summaries.merge(
        df_clusters[cluster_cols],
        on=merge_key,
        how='outer',  # Keep all papers from both files
        suffixes=('', '_dup')
    )
    
    # Remove duplicate columns if any
    df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_dup')]
    
    print(f"✅ Merged {len(df_merged)} papers")
    
    # Show column summary
    print(f"\n📊 Final columns ({len(df_merged.columns)}):")
    for col in df_merged.columns:
        non_null = df_merged[col].notna().sum()
        print(f"   - {col:25s} ({non_null:,}/{len(df_merged):,} non-null)")
    
    # Check for missing data
    print(f"\n📈 Data Completeness:")
    if 'problem' in df_merged.columns:
        summaries_count = df_merged['problem'].notna().sum()
        print(f"   - Papers with summaries: {summaries_count:,}/{len(df_merged):,}")
    if 'cluster_name' in df_merged.columns:
        clusters_count = df_merged['cluster_name'].notna().sum()
        print(f"   - Papers with cluster names: {clusters_count:,}/{len(df_merged):,}")
    if 'cluster' in df_merged.columns:
        clustered_count = (df_merged['cluster'] >= 0).sum()
        print(f"   - Papers in clusters: {clustered_count:,}/{len(df_merged):,}")
    
    # Save output
    output_path = Path(output_file)
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving merged data to {output_file}...")
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    # File size info
    import os
    file_size_mb = os.path.getsize(output_path) / 1024**2
    print(f"✅ Saved {len(df_merged)} papers")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Show sample
    print(f"\n📋 Sample merged data:")
    sample_cols = ['paper', 'cluster', 'cluster_name', 'problem', 'solution', 'eli5']
    available_cols = [col for col in sample_cols if col in df_merged.columns]
    sample = df_merged[available_cols].head(3)
    for idx, row in sample.iterrows():
        print(f"\n   Paper: {row['paper'][:60]}..." if 'paper' in row and pd.notna(row['paper']) else "")
        if 'cluster' in row and pd.notna(row['cluster']):
            print(f"   Cluster: {int(row['cluster'])} - {row.get('cluster_name', 'N/A')}")
        if 'problem' in row and pd.notna(row['problem']):
            print(f"   Problem: {row['problem'][:80]}...")
    
    print("\n" + "="*70)
    print("✅ Merge complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(df_merged)}")
    print(f"   - Columns: {len(df_merged.columns)}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge summaries and cluster information into a single parquet file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python merge_summaries_and_clusters.py
  
  # Custom output file
  python merge_summaries_and_clusters.py --output data/complete_papers.parquet
  
  # Custom input files
  python merge_summaries_and_clusters.py \\
    --summaries 3_features/data/summaries.parquet \\
    --clusters 4_cluster/data/clusters.parquet \\
    --output complete.parquet
        """
    )
    
    parser.add_argument('--summaries', 
                       default='3_features/data/neurips_2025_papers_with_summaries.parquet',
                       help='Input parquet file with summaries (problem, solution, eli5)')
    parser.add_argument('--clusters',
                       default='4_cluster/data/neurips_2025_papers_with_cluster_names.parquet',
                       help='Input parquet file with cluster information (cluster, cluster_name, cluster_description)')
    parser.add_argument('--output', '-o',
                       default='neurips_2025_papers_complete.parquet',
                       help='Output parquet file path')
    
    args = parser.parse_args()
    
    success = merge_summaries_and_clusters(
        summaries_file=args.summaries,
        clusters_file=args.clusters,
        output_file=args.output
    )
    
    exit(0 if success else 1)

