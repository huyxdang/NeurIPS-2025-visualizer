"""
Apply UMAP dimensionality reduction to embeddings (Parquet version)
Reduces high-dimensional embeddings to 2D for visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
from tqdm import tqdm

# File paths
INPUT_FILE = Path("neurips_2025_papers_with_embeddings.parquet")
OUTPUT_FILE = Path("neurips_2025_papers_with_umap.parquet")

print("="*70)
print("UMAP Dimensionality Reduction (Parquet)")
print("="*70)

# Load papers with embeddings
print(f"\n📖 Loading embeddings from {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded {len(df)} papers")

# Check for embeddings
if 'embedding' not in df.columns:
    print("❌ Error: No 'embedding' column found in dataset")
    exit(1)

embeddings_count = df['embedding'].notna().sum()
print(f"📊 Papers with embeddings: {embeddings_count}/{len(df)}")

if embeddings_count == 0:
    print("❌ Error: No embeddings found. Run embed_papers_parquet.py first.")
    exit(1)

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

# Apply UMAP
print("\n🚀 Applying UMAP dimensionality reduction...")
print("   Parameters:")
print("   - n_neighbors: 75")
print("   - min_dist: 0.2")
print("   - metric: cosine")
print("   - n_components: 2")

reducer = UMAP(
    n_neighbors=75,
    min_dist=0.2,
    metric="cosine",
    n_components=2,
    random_state=42,
    verbose=True
)

X_umap = reducer.fit_transform(X)
print(f"\n✅ UMAP complete! Output shape: {X_umap.shape}")

# Statistics about the UMAP coordinates
print(f"\n📊 UMAP Coordinate Statistics:")
print(f"   X axis: [{X_umap[:, 0].min():.2f}, {X_umap[:, 0].max():.2f}]")
print(f"   Y axis: [{X_umap[:, 1].min():.2f}, {X_umap[:, 1].max():.2f}]")
print(f"   X mean: {X_umap[:, 0].mean():.2f}, std: {X_umap[:, 0].std():.2f}")
print(f"   Y mean: {X_umap[:, 1].mean():.2f}, std: {X_umap[:, 1].std():.2f}")

# Add UMAP coordinates to dataframe
print("\n📝 Adding UMAP coordinates to papers...")

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
print(f"\n💾 Saving to {OUTPUT_FILE}...")
df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy', index=False)

# File size info
import os
file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
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
print(f"\n📊 Output: {OUTPUT_FILE}")
print(f"   - Total papers: {len(df)}")
print(f"   - With UMAP coordinates: {coords_count}")
print(f"\n💡 Next: Use this file for visualization!")