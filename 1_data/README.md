---
license: mit
task_categories:
- text-classification
- text-generation
tags:
- neurips
- machine-learning
- research-papers
- academic
datasets:
- neurips-2025
---

# NeurIPS 2025 Papers Dataset

This dataset contains all accepted papers from NeurIPS 2025, scraped from OpenReview.

## Dataset Description

- **Total Papers**: 5772
- **Tracks**: Main, Datasets and Benchmarks
- **Award Types**: Oral, Spotlight, Poster
- **Source**: OpenReview (https://openreview.net/)

## Dataset Statistics

### Overview
- **Total Papers**: 5772
- **Unique Paper IDs**: 5772
- ✅ No duplicate IDs

### Track Distribution
- **Main Track**: 5,275 papers (91.4%)
- **Datasets and Benchmarks Track**: 497 papers (8.6%)

### Award Distribution
- **Poster**: 4,949 papers (85.7%)
- **Oral**: 84 papers (1.5%)
- **Spotlight**: 739 papers (12.8%)

### Track × Award Combinations
- **Main - Poster**: 4,515 papers (78.2%)
- **Main - Spotlight**: 683 papers (11.8%)
- **Datasets and Benchmarks - Poster**: 434 papers (7.5%)
- **Main - Oral**: 77 papers (1.3%)
- **Datasets and Benchmarks - Spotlight**: 56 papers (1.0%)
- **Datasets and Benchmarks - Oral**: 7 papers (0.1%)

### Author Statistics
- **Total Authors** (across all papers): 33,878 if stats else 'N/A'
- **Unique Authors**: 23,704 if stats else 'N/A'
- **Average Authors per Paper**: 5.87 if stats else 'N/A'
- **Authors per Paper Range**: Min: 1 if stats else 'N/A', Max: 95 if stats else 'N/A', Avg: 5.87 if stats else 'N/A'
- **Papers with Authors**: 5,772 (100%) if stats else 'N/A'

### Abstract Statistics
- **Papers with Abstracts**: 5,772 (100%) if stats else 'N/A'
- **Average Abstract Length**: 1376 characters if stats else 'N/A'
- **Total Abstract Text**: 7,939,587 characters if stats else 'N/A'

## Dataset Structure

Each paper contains the following fields:
- `paper`: Title of the paper
- `authors`: List of author names
- `abstract`: Abstract text
- `link`: Direct link to OpenReview
- `track`: Track name (Main or Datasets and Benchmarks)
- `award`: Award type (Oral, Spotlight, or Poster)
- `paper_id`: Unique OpenReview paper ID

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("neurips-2025-papers", split="train")
print(dataset[0])
```

## Citation

If you use this dataset, please cite the original NeurIPS 2025 conference and OpenReview.

## License

This dataset is provided for research purposes. Please refer to OpenReview's terms of service.
