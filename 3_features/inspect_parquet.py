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
    
    # Schema with sample values
    print(f"\n📋 All Columns with Sample Values:")
    print("="*70)
    
    # Exclude embedding column from samples
    columns_to_sample = [col for col in df.columns if col != 'embedding']
    
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        
        print(f"\n{col} ({dtype})")
        print(f"   Nulls: {null_count:,} ({null_pct:.1f}%)")
        
        # Show sample value (skip embedding column)
        if col != 'embedding':
            # Find first non-null value
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample_value = non_null.iloc[0]
                
                # Format based on type
                if isinstance(sample_value, (list, np.ndarray)):
                    if len(sample_value) > 0:
                        if isinstance(sample_value[0], (list, np.ndarray)):
                            print(f"   Sample: Array of arrays (first element: {type(sample_value[0])})")
                        else:
                            preview = str(sample_value[:5]) if len(sample_value) > 5 else str(sample_value)
                            print(f"   Sample: {preview}... (length: {len(sample_value)})")
                    else:
                        print(f"   Sample: Empty array")
                elif isinstance(sample_value, str):
                    # Truncate long strings
                    max_len = 200
                    if len(sample_value) > max_len:
                        print(f"   Sample: {sample_value[:max_len]}...")
                    else:
                        print(f"   Sample: {sample_value}")
                elif isinstance(sample_value, (int, float, np.integer, np.floating)):
                    # For numeric columns, show stats
                    if pd.api.types.is_numeric_dtype(df[col]):
                        print(f"   Sample: {sample_value}")
                        if null_count < len(df):
                            print(f"   Range: [{df[col].min()}, {df[col].max()}]")
                            print(f"   Mean: {df[col].mean():.2f}")
                    else:
                        print(f"   Sample: {sample_value}")
                else:
                    print(f"   Sample: {str(sample_value)[:200]}")
            else:
                print(f"   Sample: (all null)")
        else:
            # Special handling for embedding column
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample_embedding = non_null.iloc[0]
                if isinstance(sample_embedding, (list, np.ndarray)):
                    print(f"   Embedding dimensions: {len(sample_embedding)}")
                    print(f"   Papers with embeddings: {len(non_null):,}/{len(df):,}")
    
    # Distribution statistics for categorical columns
    print(f"\n📊 Column Distributions:")
    print("="*70)
    
    # Track distribution
    if 'track' in df.columns:
        print(f"\n🎯 Track Distribution:")
        track_counts = df['track'].value_counts()
        for track, count in track_counts.items():
            print(f"   - {track}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Award distribution
    if 'award' in df.columns:
        print(f"\n🏆 Award Distribution:")
        award_counts = df['award'].value_counts()
        for award, count in award_counts.items():
            print(f"   - {award}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Authors info
    if 'authors' in df.columns:
        print(f"\n👥 Author Statistics:")
        author_counts = df['authors'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
        print(f"   Average authors per paper: {author_counts.mean():.1f}")
        print(f"   Max authors: {author_counts.max()}")
        print(f"   Min authors: {author_counts.min()}")
    
    # UMAP statistics
    if 'umap_x' in df.columns and 'umap_y' in df.columns:
        print(f"\n📍 UMAP Coordinate Statistics:")
        umap_valid = df[['umap_x', 'umap_y']].dropna()
        if len(umap_valid) > 0:
            print(f"   Papers with UMAP coordinates: {len(umap_valid):,}/{len(df):,}")
            print(f"   X range: [{df['umap_x'].min():.2f}, {df['umap_x'].max():.2f}]")
            print(f"   Y range: [{df['umap_y'].min():.2f}, {df['umap_y'].max():.2f}]")
    
    # Summary statistics (if summaries exist)
    summary_cols = ['problem', 'solution', 'eli5']
    if any(col in df.columns for col in summary_cols):
        print(f"\n📝 Summary Statistics:")
        for col in summary_cols:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Count errors
                    errors = (df[col].str.contains('error|parse|failed', case=False, na=False)).sum()
                    avg_len = non_null.str.len().mean()
                    print(f"   {col}: {len(non_null):,} summaries, avg length: {avg_len:.0f} chars, errors: {errors}")
    
    print("\n" + "="*70)
    print("✅ Inspection complete!")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <file.parquet>")
        print("\nAvailable files in 3_features/data/ directory:")
        data_dir = Path(__file__).parent / "data"
        if data_dir.exists():
            for f in data_dir.glob("*.parquet"):
                print(f"  - data/{f.name}")
        # Also check for default summary file
        default_file = data_dir / "neurips_2025_papers_with_summaries.parquet"
        if default_file.exists():
            print(f"\nDefault file exists: {default_file}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    # If no directory specified, default to data/ in same directory as script
    if "/" not in file_path and "\\" not in file_path:
        data_dir = Path(__file__).parent / "data"
        file_path = str(data_dir / file_path)
    
    inspect_parquet(file_path)