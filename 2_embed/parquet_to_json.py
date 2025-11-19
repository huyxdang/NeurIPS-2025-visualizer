"""
Convert Parquet file back to JSON
Useful for web visualizations or when JSON format is needed
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def parquet_to_json(input_file, output_file=None, exclude_embeddings=False):
    """
    Convert Parquet to JSON
    
    Args:
        input_file: Input Parquet file path
        output_file: Output JSON file path (optional)
        exclude_embeddings: If True, exclude large embedding arrays to reduce file size
    """
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    # Determine output file name
    if output_file is None:
        output_file = input_path.with_suffix('.json')
    
    output_path = Path(output_file)
    
    print("="*70)
    print("Converting Parquet to JSON")
    print("="*70)
    
    # Load Parquet
    print(f"\n📖 Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Optionally exclude embeddings to reduce file size
    if exclude_embeddings and 'embedding' in df.columns:
        print("📉 Excluding embeddings (will reduce file size)")
        df = df.drop(columns=['embedding'])
    
    # Convert to JSON
    print(f"\n🔄 Converting to JSON...")
    records = df.to_dict('records')
    
    # Convert numpy arrays to lists for JSON serialization
    print("🔧 Converting numpy arrays to lists...")
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                record[key] = value.item()
    
    # Show memory size before saving
    json_str = json.dumps(records)
    size_mb = sys.getsizeof(json_str) / 1024**2
    print(f"   JSON size in memory: {size_mb:.2f} MB")
    
    # Save to file
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    # Compare file sizes
    import os
    parquet_size = os.path.getsize(input_path) / 1024**2
    json_size = os.path.getsize(output_path) / 1024**2
    
    print(f"\n✅ Conversion complete!")
    print(f"\n📦 File Sizes:")
    print(f"   Parquet: {parquet_size:.2f} MB")
    print(f"   JSON:    {json_size:.2f} MB")
    print(f"   Ratio:   {json_size/parquet_size:.1f}x")
    
    # Show sample
    print(f"\n📄 Sample record:")
    sample = records[0]
    sample_str = json.dumps(sample, indent=2, ensure_ascii=False)
    print(sample_str[:500] + "..." if len(sample_str) > 500 else sample_str)
    
    print("\n" + "="*70)
    print(f"✅ Saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert Parquet to JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with embeddings
  python parquet_to_json.py neurips_2025_papers_with_umap.parquet
  
  # Convert without embeddings (smaller file)
  python parquet_to_json.py neurips_2025_papers_with_umap.parquet --no-embeddings
  
  # Specify output file
  python parquet_to_json.py input.parquet -o output.json
        """
    )
    
    parser.add_argument('input', help='Input Parquet file')
    parser.add_argument('-o', '--output', help='Output JSON file (optional)')
    parser.add_argument('--no-embeddings', action='store_true', 
                       help='Exclude embeddings to reduce file size')
    
    args = parser.parse_args()
    
    parquet_to_json(args.input, args.output, args.no_embeddings)