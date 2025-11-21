"""
Utility to inspect Parquet files
Shows schema, sample data, and statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def inspect_parquet(file_path):
    """Inspect a Parquet file and display useful information"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print("="*70)
    print(f"Inspecting: {file_path.name}")
    print("="*70)
    
    # Load file
    print("\n📖 Loading file...")
    df = pd.read_parquet(file_path)
    
    # Basic info
    print(f"\n📊 Basic Information:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # File size
    import os
    file_size = os.path.getsize(file_path) / 1024**2
    print(f"   File size: {file_size:.2f} MB")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   Memory usage: {memory_mb:.2f} MB")
    
    # Schema
    print(f"\n📋 Schema:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"   - {col:20s} {str(dtype):15s} ({null_count:,} nulls, {null_pct:.1f}%)")
    
    # Special columns
    special_cols = ['embedding', 'umap_x', 'umap_y']
    for col in special_cols:
        if col in df.columns:
            if col == 'embedding':
                # Check embedding dimensions
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    sample_embedding = non_null.iloc[0]
                    if isinstance(sample_embedding, (list, np.ndarray)):
                        print(f"\n🔢 Embedding Info:")
                        print(f"   Dimensions: {len(sample_embedding)}")
                        print(f"   Papers with embeddings: {len(non_null):,}/{len(df):,}")
            
            elif col in ['umap_x', 'umap_y']:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    print(f"\n📍 {col.upper()} Statistics:")
                    print(f"   Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
                    print(f"   Mean: {df[col].mean():.2f}")
                    print(f"   Std: {df[col].std():.2f}")
    
    # Track and Award distribution
    if 'track' in df.columns:
        print(f"\n🎯 Track Distribution:")
        track_counts = df['track'].value_counts()
        for track, count in track_counts.items():
            print(f"   - {track}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if 'award' in df.columns:
        print(f"\n🏆 Award Distribution:")
        award_counts = df['award'].value_counts()
        for award, count in award_counts.items():
            print(f"   - {award}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Sample rows
    print(f"\n📄 Sample Rows:")
    sample_cols = [col for col in ['paper', 'track', 'award', 'umap_x', 'umap_y'] if col in df.columns]
    if sample_cols:
        print(df[sample_cols].head(3).to_string())
    
    # Authors info
    if 'authors' in df.columns:
        print(f"\n👥 Author Statistics:")
        author_counts = df['authors'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
        print(f"   Average authors per paper: {author_counts.mean():.1f}")
        print(f"   Max authors: {author_counts.max()}")
        print(f"   Min authors: {author_counts.min()}")
    
    print("\n" + "="*70)
    print("✅ Inspection complete!")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <file.parquet>")
        print("\nAvailable files in data/ directory:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.parquet"):
                print(f"  - data/{f.name}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    # If no directory specified, default to data/
    if "/" not in file_path and "\\" not in file_path:
        file_path = str(Path("data") / file_path)
    inspect_parquet(file_path)