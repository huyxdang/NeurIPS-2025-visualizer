"""
Apply UMAP dimensionality reduction to embeddings (Parquet version)
Reduces high-dimensional embeddings to 2D for visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from tqdm import tqdm
import argparse
import sys

# Default file paths
DATA_DIR = Path("2_embed/data")
DEFAULT_INPUT = DATA_DIR / "neurips_2025_papers_with_embeddings.parquet"
DEFAULT_OUTPUT = DATA_DIR / "neurips_2025_papers_with_umap.parquet"

def main(input_file=None, output_file=None):
    """Main function to apply UMAP reduction"""
    
    # Use provided files or defaults
    if input_file is None:
        input_file = DEFAULT_INPUT
    else:
        input_file = Path(input_file)
    
    if output_file is None:
        output_file = DEFAULT_OUTPUT
    else:
        output_file = Path(output_file)

    # Load papers with embeddings
    df = pd.read_parquet(input_file)
    print(f"✅ Loaded {len(df)} papers")

    # Check for embeddings
    if 'embedding' not in df.columns:
        print("❌ Error: No 'embedding' column found in dataset")
        sys.exit(1)

    embeddings_count = df['embedding'].notna().sum()
    print(f"📊 Papers with embeddings: {embeddings_count}/{len(df)}")

    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    valid_mask = df['embedding'].notna()
    valid_indices = df[valid_mask].index.tolist()

    # Convert embeddings to numpy array
    embeddings_list = df.loc[valid_mask, 'embedding'].tolist()
    X = np.array(embeddings_list)

    print(f"✅ Embedding matrix shape: {X.shape}")
    print(f"   Dimensions: {X.shape[1]}")
    print(f"   Valid papers: {X.shape[0]}")

    reducer = UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    X_umap = reducer.fit_transform(X)
    print(f"\n✅ UMAP complete! Output shape: {X_umap.shape}")

    # Statistics about the UMAP coordinates
    print(f"\n📊 UMAP Coordinate Statistics:")
    print(f"   X axis: [{X_umap[:, 0].min():.2f}, {X_umap[:, 0].max():.2f}]")
    print(f"   Y axis: [{X_umap[:, 1].min():.2f}, {X_umap[:, 1].max():.2f}]")
    print(f"   X mean: {X_umap[:, 0].mean():.2f}, std: {X_umap[:, 0].std():.2f}")
    print(f"   Y mean: {X_umap[:, 1].mean():.2f}, std: {X_umap[:, 1].std():.2f}")

    # Initialize columns with NaN
    df['umap_x'] = np.nan
    df['umap_y'] = np.nan

    # Assign UMAP coordinates to papers with embeddings
    for i, idx in enumerate(tqdm(valid_indices, desc="Adding UMAP coordinates")):
        df.at[idx, 'umap_x'] = float(X_umap[i, 0])
        df.at[idx, 'umap_y'] = float(X_umap[i, 1])

    # Check how many papers have coordinates
    coords_count = df['umap_x'].notna().sum()
    print(f"✅ Added coordinates to {coords_count} papers")

    # Save to Parquet
    print(f"\n💾 Saving to {output_file}...")
    df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)

    # File size info
    import os
    file_size_mb = os.path.getsize(output_file) / 1024**2
    print(f"✅ Saved {len(df)} papers with UMAP coordinates")
    print(f"   File size: {file_size_mb:.2f} MB")

    # Show sample
    print(f"\n📄 Sample paper with UMAP coordinates:")
    sample = df[df['umap_x'].notna()].iloc[0]
    print(f"   Title: {sample['paper'][:60]}...")
    print(f"   UMAP: ({sample['umap_x']:.2f}, {sample['umap_y']:.2f})")
    print(f"   Track: {sample['track']}")
    print(f"   Award: {sample['award']}")

    print("\n" + "="*70)
    print("✅ UMAP reduction complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(df)}")
    print(f"   - With UMAP coordinates: {coords_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply UMAP dimensionality reduction to embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default files (data/neurips_2025_papers_with_embeddings.parquet)
  python umap_reduce_parquet.py
  
  # Specify input file
  python umap_reduce_parquet.py data/neurips_2025_papers_with_embeddings.parquet
  
  # Specify both input and output
  python umap_reduce_parquet.py input.parquet -o output.parquet
        """
    )
    
    parser.add_argument('input', nargs='?', 
                       help='Input Parquet file with embeddings (default: 2_embed/data/neurips_2025_papers_with_embeddings.parquet)')
    parser.add_argument('-o', '--output', 
                       help='Output Parquet file (default: 2_embed/data/neurips_2025_papers_with_umap.parquet)')
    
    args = parser.parse_args()
    
    main(args.input, args.output)