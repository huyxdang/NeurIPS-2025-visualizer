"""
Embed papers using Google Embedding Gemma 300M (local model)
Works with Parquet format for better performance

Features:
- Local inference (no API calls, no quotas!)
- Batch processing for efficiency
- Resume capability: Can continue from where it left off
- Parquet format: Efficient storage and loading
- Progress saving: Saves progress incrementally to Parquet
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from embedding_gemma_inference import get_embedding_model

# Configuration
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "neurips_2025_papers_full.parquet"
OUTPUT_FILE = DATA_DIR / "neurips_2025_papers_with_embeddings.parquet"
PROGRESS_FILE = DATA_DIR / "embedding_progress.parquet"
BATCH_SIZE = 32  # Process 32 papers at once
SAVE_INTERVAL = 100  # Save progress every 100 papers (since batching is fast)

def load_progress():
    """Load existing progress if available"""
    if PROGRESS_FILE.exists():
        print(f"📂 Found existing progress file: {PROGRESS_FILE}")
        df = pd.read_parquet(PROGRESS_FILE)
        print(f"   Loaded {len(df)} previously processed papers")
        return df
    return None

def save_progress(df):
    """Save progress to Parquet"""
    df.to_parquet(PROGRESS_FILE, engine='pyarrow', compression='snappy', index=False)

print("="*70)
print("NeurIPS 2025 Papers - Embedding Generation (Local Model)")
print("="*70)

# Load embedding model
print("\n🤖 Loading Embedding Gemma 300M model...")
embedding_model = get_embedding_model(batch_size=BATCH_SIZE)
embedding_dim = embedding_model.get_embedding_dim()
print(f"✅ Model ready! Embedding dimension: {embedding_dim}")

# Load input data
print(f"\n📖 Loading papers from {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded {len(df)} papers")

# Load progress if exists
progress_df = load_progress()

if progress_df is not None:
    # Merge with existing data to continue from where we left off
    processed_ids = set(progress_df[progress_df['embedding'].notna()]['paper_id'].tolist())
    print(f"📊 Progress: {len(processed_ids)}/{len(df)} papers already processed")
    
    # Start with progress data
    df = df.merge(
        progress_df[['paper_id', 'embedding']], 
        on='paper_id', 
        how='left',
        suffixes=('', '_progress')
    )
    
    # Use progress embeddings where available
    if 'embedding_progress' in df.columns:
        df['embedding'] = df['embedding_progress'].combine_first(df.get('embedding'))
        df = df.drop(columns=['embedding_progress'])
else:
    print("🆕 Starting fresh (no previous progress found)")
    df['embedding'] = None
    processed_ids = set()

# Generate embeddings
print("\n🚀 Generating embeddings...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Estimated time: ~{len(df) * 0.05 / 60:.1f} minutes (much faster than API!)")

errors = []
processed_count = len(processed_ids)

try:
    # Filter to only papers without embeddings
    mask_needs_embedding = df['embedding'].isna()
    indices_to_process = df[mask_needs_embedding].index.tolist()
    
    print(f"📝 {len(indices_to_process)} papers need embeddings\n")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(indices_to_process), BATCH_SIZE), 
                           desc="Processing batches"):
        batch_indices = indices_to_process[batch_start:batch_start + BATCH_SIZE]
        
        # Collect abstracts for this batch
        batch_abstracts = []
        valid_batch_indices = []
        
        for idx in batch_indices:
            abstract = df.at[idx, 'abstract']
            if abstract and abstract != "N/A" and pd.notna(abstract):
                batch_abstracts.append(abstract)
                valid_batch_indices.append(idx)
        
        if not batch_abstracts:
            continue
        
        try:
            # Generate embeddings for the batch
            batch_embeddings = embedding_model.encode(batch_abstracts, show_progress=False)
            
            # Assign embeddings to dataframe
            for idx, embedding in zip(valid_batch_indices, batch_embeddings):
                df.at[idx, 'embedding'] = embedding
                processed_count += 1
            
            # Save progress periodically
            if processed_count % SAVE_INTERVAL == 0:
                save_progress(df)
                
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error processing batch starting at {batch_start}: {error_msg}")
            for idx in valid_batch_indices:
                errors.append({
                    "index": idx, 
                    "paper_id": df.at[idx, 'paper_id'], 
                    "error": error_msg
                })
            continue
    
    # Final save
    save_progress(df)
    
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user. Saving progress...")
    save_progress(df)
    print(f"✅ Progress saved: {processed_count}/{len(df)} papers")
    exit(0)

# Count successful embeddings
embeddings_count = df['embedding'].notna().sum()
print(f"\n✅ Generated {embeddings_count} embeddings")

# Save final output
print(f"\n💾 Saving final output to {OUTPUT_FILE}...")
df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy', index=False)

print(f"✅ Saved {len(df)} papers with embeddings to {OUTPUT_FILE}")

# Clean up progress file if fully complete
if PROGRESS_FILE.exists() and embeddings_count == len(df):
    PROGRESS_FILE.unlink()
    print("✅ Cleaned up progress file (all papers processed)")

if errors:
    print(f"\n⚠️  {len(errors)} errors encountered")
    print("First few errors:")
    for error in errors[:5]:
        print(f"  - Paper {error['paper_id']}: {error['error']}")

print("\n" + "="*70)
print("✅ Embedding generation complete!")
print("="*70)
print(f"\n📊 Statistics:")
print(f"   Total papers: {len(df)}")
print(f"   With embeddings: {embeddings_count}")
print(f"   Embedding dimension: {embedding_dim}")
print(f"   Model: Embedding Gemma 300M (local)")