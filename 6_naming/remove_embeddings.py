"""
Remove embedding columns from parquet file
Creates a new parquet file without embedding columns to reduce file size
"""

import pandas as pd
from pathlib import Path
import argparse


def remove_embeddings(
    input_file: str = "6_naming/data/neurips_2025_papers_with_all_names.parquet",
    output_file: str = "6_naming/data/neurips_2025_papers_with_all_names_no_embeddings.parquet",
    embedding_columns: list = None
):
    """
    Remove embedding columns from parquet file
    
    Args:
        input_file: Input parquet file
        output_file: Output parquet file without embeddings
        embedding_columns: List of column names to remove (if None, auto-detect)
    """
    print("="*70)
    print("Remove Embedding Columns")
    print("="*70)
    
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"\n📖 Loading file: {input_file}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Original file size
    import os
    original_size_mb = os.path.getsize(input_path) / 1024**2
    print(f"   Original file size: {original_size_mb:.2f} MB")
    
    # Detect embedding columns if not specified
    if embedding_columns is None:
        # Common embedding column names
        embedding_keywords = ['embedding', 'embed', 'vector', 'emb']
        embedding_columns = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in embedding_keywords)]
        
        # Also check for columns that are arrays/lists (likely embeddings)
        for col in df.columns:
            if col not in embedding_columns:
                try:
                    # Check if column contains arrays
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample is not None:
                        # Check if it's a numpy array or list-like
                        import numpy as np
                        if isinstance(sample, (np.ndarray, list)) and len(sample) > 10:
                            # Likely an embedding vector
                            embedding_columns.append(col)
                except:
                    pass
    
    if not embedding_columns:
        print("\n⚠️  No embedding columns detected. Nothing to remove.")
        print("   Available columns:", ", ".join(df.columns.tolist()))
        return
    
    print(f"\n📊 Found {len(embedding_columns)} embedding column(s) to remove:")
    for col in embedding_columns:
        try:
            col_size_mb = df[col].memory_usage(deep=True) / 1024**2
            print(f"   - {col} ({col_size_mb:.2f} MB)")
        except Exception:
            print(f"   - {col} (size unknown)")
    
    # Calculate total size of embedding columns
    try:
        total_embedding_size = sum(df[col].memory_usage(deep=True) for col in embedding_columns) / 1024**2
        print(f"\n   Total embedding size: {total_embedding_size:.2f} MB")
    except Exception:
        print(f"\n   Total embedding size: (cannot calculate)")
    
    # Remove embedding columns
    print(f"\n🗑️  Removing embedding columns...")
    df_cleaned = df.drop(columns=embedding_columns)
    
    print(f"✅ Removed {len(embedding_columns)} column(s)")
    print(f"   Remaining columns: {len(df_cleaned.columns)}")
    print(f"   Remaining columns: {', '.join(df_cleaned.columns.tolist())}")
    
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_file}...")
    df_cleaned.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    # Compare file sizes
    new_size_mb = os.path.getsize(output_path) / 1024**2
    size_reduction = original_size_mb - new_size_mb
    reduction_pct = (size_reduction / original_size_mb) * 100
    
    print(f"\n✅ File saved successfully!")
    print(f"\n📊 File Size Comparison:")
    print(f"   Original: {original_size_mb:.2f} MB")
    print(f"   New:      {new_size_mb:.2f} MB")
    print(f"   Reduced:  {size_reduction:.2f} MB ({reduction_pct:.1f}%)")
    
    print(f"\n{'='*70}")
    print("✅ Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove embedding columns from parquet file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove embeddings from default file
  python remove_embeddings.py
  
  # Custom input/output
  python remove_embeddings.py \\
    --input data/input.parquet \\
    --output data/output.parquet
  
  # Specify specific columns to remove
  python remove_embeddings.py \\
    --columns embedding embedding_vector
        """
    )
    
    parser.add_argument('--input', '-i',
                       default='6_naming/data/neurips_2025_papers_with_all_names.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', '-o',
                       default='6_naming/data/neurips_2025_papers_with_all_names_no_embeddings.parquet',
                       help='Output parquet file without embeddings')
    parser.add_argument('--columns', '-c', nargs='+',
                       default=None,
                       help='Specific column names to remove (if not specified, auto-detect)')
    
    args = parser.parse_args()
    
    remove_embeddings(
        input_file=args.input,
        output_file=args.output,
        embedding_columns=args.columns
    )

