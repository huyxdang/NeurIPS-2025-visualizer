# Clustering & Naming Pipeline

Complete pipeline for clustering papers and generating semantic cluster names.

## 🎯 What It Does

1. **K-means clustering** on UMAP coordinates
2. **LLM-based naming** using Qwen 2.5 14B
3. Adds `cluster`, `cluster_name`, `cluster_description` columns

## 🚀 Quick Start

```bash
# Run everything (auto-detect optimal k)
python orchestrate_clustering.py --all --auto-k

# Or specify number of clusters
python orchestrate_clustering.py --all --n-clusters 25
```

## 📦 Files

1. **cluster_papers.py** - K-means clustering
2. **name_clusters_modal.py** - LLM cluster naming
3. **orchestrate_clustering.py** - Runs both

## ⚙️ Options

```bash
# Clustering
--auto-k                # Auto-detect optimal k
--n-clusters 25         # Specify k manually

# Naming
--n-samples 8           # Papers to sample per cluster
--batch-size 5          # Clusters to name at once

# General
--force                 # Regenerate existing files
```

## 💡 Recommended Model

**Qwen 2.5 14B** (default) - Best balance:
- Speed: 1-2 minutes
- Quality: 9.5/10
- Cost: $0.03
- GPU: A10G

## 📊 Performance

- K-means: 3-5 min (CPU)
- Naming (14B): 1-2 min (GPU)
- **Total: ~5 min, $0.05**

## 🎨 Example Output

```
Cluster  Size   Name
0        234    Vision Transformers
1        198    Reinforcement Learning Theory
2        312    Large Language Models
3        421    Graph Neural Networks
4        187    Diffusion Models
...
```

## 📁 Output

- `neurips_2025_papers_with_cluster_names.parquet`
- `cluster_names.csv`
- `cluster_visualization.png`

Done! 🎉
