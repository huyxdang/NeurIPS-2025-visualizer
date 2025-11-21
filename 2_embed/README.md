# NeurIPS 2025 Papers - Parquet Pipeline 🚀

Complete pipeline for processing NeurIPS 2025 papers using **Parquet format** for better performance with large datasets.

## 🎯 Complete Workflow

```
JSON Papers (5k papers)
    ↓
[1] Convert to Parquet (fast storage)
    ↓
[2] Generate Embeddings (Local Model or Modal GPU)
    ↓
[3] Apply UMAP (2D reduction)
    ↓
[4] Ready for Visualization! 
    ↓ (optional)
[5] Convert to JSON (for web viz)
```

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run complete pipeline
python main.py --all
```

### Option 2: Step by Step

```bash
# Step 1: Convert JSON to Parquet
python json_to_parquet.py

# Step 2: Generate embeddings (local model or Modal GPU)
python embed_local.py
# Or use Modal GPU (faster):
# modal run embed.py

# Step 3: Apply UMAP dimensionality reduction
python umap_reduce_parquet.py

# Step 4 (Optional): Convert back to JSON for web viz
python parquet_to_json.py data/neurips_2025_papers_with_umap.parquet
```

## 📁 File Structure

All data files are stored in the `data/` subdirectory:

```
data/
  Input:
    neurips_2025_papers_full.json          # Your 5k papers from scraper

  Pipeline Output:
    neurips_2025_papers_full.parquet       # Step 1: Parquet format
    neurips_2025_papers_with_embeddings.parquet  # Step 2: With embeddings
    neurips_2025_papers_with_umap.parquet  # Step 3: With 2D coordinates
    neurips_2025_papers_final.json         # Step 4: Final JSON (optional)

  Progress Files:
    embedding_progress.parquet             # Resume embeddings if interrupted
```

## 🛠️ Individual Scripts

### 1. Convert JSON to Parquet

```bash
python json_to_parquet.py
```

**What it does:**
- Converts your JSON file to Parquet
- Shows file size comparison
- Demonstrates loading speed improvement

**Output:** `data/neurips_2025_papers_full.parquet`

### 2. Generate Embeddings

```bash
python embed_local.py
```

**What it does:**
- Reads papers from Parquet
- Generates embeddings using local Embedding Gemma 300M model
- Handles errors gracefully
- **Saves progress periodically** (can resume if interrupted!)

**Features:**
- ✅ Resume capability
- ✅ Progress saving
- ✅ Error handling
- ✅ Local inference (no API calls needed!)

**Output:** `data/neurips_2025_papers_with_embeddings.parquet`

### 3. Apply UMAP Reduction

```bash
python umap_reduce_parquet.py
```

**What it does:**
- Loads embeddings from Parquet
- Applies UMAP to reduce from high-dimensional embeddings → 2D
- Adds `umap_x` and `umap_y` coordinates to each paper

**Parameters:**
- n_neighbors: 75
- min_dist: 0.2
- metric: cosine

**Output:** `data/neurips_2025_papers_with_umap.parquet`

### 4. Convert to JSON (Optional)

```bash
# With embeddings (large file)
python parquet_to_json.py data/neurips_2025_papers_with_umap.parquet

# Without embeddings (smaller, recommended for web)
python parquet_to_json.py data/neurips_2025_papers_with_umap.parquet --no-embeddings
```

**What it does:**
- Converts final Parquet back to JSON
- Option to exclude embeddings for smaller file size

**Output:** `data/neurips_2025_papers_with_umap.json` (or specified output path)

## 🔍 Utility Scripts

### Inspect Parquet Files

```bash
# Specify full path or just filename (will look in data/ by default)
python inspect_parquet.py data/neurips_2025_papers_with_umap.parquet
# Or just:
python inspect_parquet.py neurips_2025_papers_with_umap.parquet
```

Shows:
- File size and memory usage
- Schema and data types
- Sample data
- Statistics (track distribution, award types, etc.)
- UMAP coordinate ranges

## 🎮 Using the Pipeline Script

The `main.py` script orchestrates everything:

```bash
# Run complete pipeline
python main.py --all

# Run complete pipeline and convert to JSON at the end
python main.py --all --to-json

# Run specific step
python main.py --step convert
python main.py --step embed
python main.py --step umap

# Force regeneration (skip existing files)
python main.py --all --force

# Convert to JSON without embeddings (smaller file)
python main.py --step to_json --no-embeddings

# Use Modal GPU for embeddings (much faster!)
python main.py --all --modal
```

## 💾 Data Format

### After UMAP (Final Format)

Each paper has:
```python
{
    "paper": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Abstract text...",
    "link": "https://openreview.net/forum?id=...",
    "track": "Main",
    "award": "Oral",
    "paper_id": "abc123",
    "embedding": [0.123, -0.456, ...],  # High-dimensional embedding vector
    "umap_x": 5.23,   # 2D coordinate
    "umap_y": -3.45   # 2D coordinate
}
```

## 🔄 Resume Capability

If embeddings get interrupted:

1. Progress is automatically saved every 10 papers
2. Progress file: `data/embedding_progress.parquet`
3. Just run `embed_local.py` again
4. It will continue from where it left off!


## 📊 Visualization Tips

For web visualization:
```bash
# Convert to JSON without embeddings (recommended)
python parquet_to_json.py data/neurips_2025_papers_with_umap.parquet --no-embeddings
```


## 📦 Dependencies

```bash
pip install -r ../requirements.txt
```



## 🎯 Next Steps

After running the pipeline:

1. **For web visualization:**
   - Use `data/neurips_2025_papers_final.json` (without embeddings)
   - Load in React/D3.js/your viz framework
   - Plot using `umap_x` and `umap_y` coordinates

2. **For Python analysis:**
   - Use `data/neurips_2025_papers_with_umap.parquet` directly
   - Much faster than JSON
   - Work with pandas DataFrames