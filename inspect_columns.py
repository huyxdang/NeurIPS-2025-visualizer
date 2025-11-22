"""
Inspect all columns in the parquet file
Displays column names, data types, and sample information
"""

import pandas as pd
from pathlib import Path
import argparse


def inspect_columns(
    input_file: str = "6_naming/data/neurips_2025_papers_with_all_names.parquet",
    show_samples: bool = True,
    max_sample_length: int = 100
):
    """
    Display all columns and their information from the parquet file
    
    Args:
        input_file: Path to the parquet file
        show_samples: Whether to show sample values for each column
        max_sample_length: Maximum length of sample values to display
    """
    print("="*70)
    print("Parquet File Column Inspection")
    print("="*70)
    
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"\n📖 Loading file: {input_file}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df):,} rows")
    
    # File size
    import os
    file_size_mb = os.path.getsize(input_path) / 1024**2
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Column overview
    print(f"\n📊 Column Overview:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Total rows: {len(df):,}")
    
    # Display all columns
    print(f"\n{'='*70}")
    print("All Columns:")
    print(f"{'='*70}")
    
    for i, col in enumerate(df.columns, 1):
        print(f"\n{i}. {col}")
        print(f"   Type: {df[col].dtype}")
        
        # Non-null count
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"   Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")
        if null_count > 0:
            print(f"   Null: {null_count:,} ({null_count/len(df)*100:.1f}%)")
        
        # Unique values for categorical columns
        if df[col].dtype in ['object', 'string'] or df[col].dtype.name == 'category':
            try:
                unique_count = df[col].nunique()
                print(f"   Unique values: {unique_count:,}")
                if unique_count <= 20:
                    try:
                        unique_vals = df[col].dropna().unique().tolist()
                        # Try to sort, but handle unhashable types
                        try:
                            unique_vals = sorted(unique_vals)
                        except TypeError:
                            pass  # Can't sort unhashable types
                        print(f"   Values: {unique_vals}")
                    except (TypeError, ValueError):
                        print(f"   Values: (cannot display - contains unhashable types)")
            except (TypeError, ValueError):
                print(f"   Unique values: (cannot count - contains unhashable types)")
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.2f}" if df[col].dtype != 'int64' else f"   Mean: {df[col].mean():.1f}")
        
        # Sample values
        if show_samples:
            try:
                sample_values = df[col].dropna().head(3)
                if len(sample_values) > 0:
                    print(f"   Sample values:")
                    for idx, val in sample_values.items():
                        try:
                            val_str = str(val)
                            if len(val_str) > max_sample_length:
                                val_str = val_str[:max_sample_length] + "..."
                            print(f"      - {val_str}")
                        except Exception:
                            print(f"      - <unhashable/unprintable value>")
            except Exception:
                print(f"   Sample values: (cannot display)")
    
    # Summary table
    print(f"\n{'='*70}")
    print("Column Summary Table:")
    print(f"{'='*70}")
    print(f"{'Column Name':<30} {'Type':<15} {'Non-Null':<12} {'Unique':<10}")
    print("-"*70)
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if len(dtype_str) > 14:
            dtype_str = dtype_str[:11] + "..."
        
        non_null = df[col].notna().sum()
        try:
            unique = df[col].nunique()
            unique_str = f"{unique:,}"
        except (TypeError, ValueError):
            unique_str = "N/A"
        
        col_name = col[:29] if len(col) <= 29 else col[:26] + "..."
        print(f"{col_name:<30} {dtype_str:<15} {non_null:<12,} {unique_str:<10}")
    
    # List all column names
    print(f"\n{'='*70}")
    print("All Column Names (comma-separated):")
    print(f"{'='*70}")
    print(", ".join(df.columns.tolist()))
    
    print(f"\n{'='*70}")
    print("✅ Inspection complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inspect all columns in a parquet file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect default file
  python inspect_columns.py
  
  # Inspect custom file
  python inspect_columns.py --input data/my_file.parquet
  
  # Don't show sample values
  python inspect_columns.py --no-samples
        """
    )
    
    parser.add_argument('--input', '-i',
                       default='6_naming/data/neurips_2025_papers_with_all_names.parquet',
                       help='Input parquet file to inspect')
    parser.add_argument('--no-samples', dest='show_samples', action='store_false',
                       help='Do not show sample values')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of sample values to display')
    
    args = parser.parse_args()
    
    inspect_columns(
        input_file=args.input,
        show_samples=args.show_samples,
        max_sample_length=args.max_length
    )

