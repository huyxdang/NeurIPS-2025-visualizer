"""
Convert NeurIPS 2025 papers from JSON to Parquet format
Parquet is much more efficient for large datasets (faster loading, smaller file size)
"""

import json
import pandas as pd
from pathlib import Path

# File paths
INPUT_JSON = Path("neurips_2025_papers.json")
OUTPUT_PARQUET = Path("neurips_2025_papers_full.parquet")

print("="*70)
print("Converting JSON to Parquet")
print("="*70)

# Load JSON
print(f"\n📖 Loading JSON from {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"✅ Loaded {len(papers)} papers")

# Convert to DataFrame
print("\n🔄 Converting to DataFrame...")
df = pd.DataFrame(papers)

# Display info
print(f"\n📊 Dataset Info:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Show sample
print(f"\n📄 Sample row:")
print(df.iloc[0].to_dict())

# Save to Parquet
print(f"\n💾 Saving to {OUTPUT_PARQUET}...")
df.to_parquet(OUTPUT_PARQUET, engine='pyarrow', compression='snappy', index=False)

# Compare file sizes
import os
json_size = os.path.getsize(INPUT_JSON) / 1024**2
parquet_size = os.path.getsize(OUTPUT_PARQUET) / 1024**2

print(f"\n✅ Conversion complete!")
print(f"\n📦 File Size Comparison:")
print(f"   JSON:    {json_size:.2f} MB")
print(f"   Parquet: {parquet_size:.2f} MB")
print(f"   Savings: {(1 - parquet_size/json_size)*100:.1f}%")

# Test loading speed
print(f"\n⚡ Testing load speed...")
import time

# Time JSON loading
start = time.time()
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    _ = json.load(f)
json_time = time.time() - start

# Time Parquet loading
start = time.time()
_ = pd.read_parquet(OUTPUT_PARQUET)
parquet_time = time.time() - start

print(f"   JSON load time:    {json_time:.3f}s")
print(f"   Parquet load time: {parquet_time:.3f}s")
print(f"   Speedup: {json_time/parquet_time:.1f}x faster")

print(f"\n{'='*70}")
print(f"✅ Done! Use {OUTPUT_PARQUET} for faster processing")
print(f"{'='*70}")