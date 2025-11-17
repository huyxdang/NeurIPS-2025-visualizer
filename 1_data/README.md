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
