"""
Upload NeurIPS 2025 papers dataset to Hugging Face Hub
"""

import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import HfHubHTTPError


def load_data(json_file):
    """Load papers from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_readme(dataset_name, stats=None):
    """Create README.md for the dataset"""
    # Build statistics sections
    if stats:
        total_papers = stats['total_papers']
        
        # Overview
        duplicate_status = '✅ No duplicate IDs' if not stats.get('has_duplicates', True) else '⚠️ Duplicate IDs found'
        
        # Track Distribution
        track_dist = '\n'.join([
            f"- **{track} Track**: {count:,} papers ({count/total_papers*100:.1f}%)"
            for track, count in stats.get('tracks', {}).items()
        ])
        
        # Award Distribution
        award_dist = '\n'.join([
            f"- **{award}**: {count:,} papers ({count/total_papers*100:.1f}%)"
            for award, count in stats.get('awards', {}).items()
        ])
        
        # Track × Award Combinations
        track_award_dist = '\n'.join([
            f"- **{combo}**: {count:,} papers ({count/total_papers*100:.1f}%)"
            for combo, count in sorted(stats.get('track_award_combos', {}).items(), key=lambda x: -x[1])
        ])
        
        # Author Statistics
        papers_with_authors_pct = (stats.get('papers_with_authors', 0) / total_papers * 100) if total_papers > 0 else 0
        
        # Abstract Statistics
        papers_with_abstracts_pct = (stats.get('papers_with_abstracts', 0) / total_papers * 100) if total_papers > 0 else 0
    else:
        total_papers = 'N/A'
        duplicate_status = 'N/A'
        track_dist = '- N/A'
        award_dist = '- N/A'
        track_award_dist = '- N/A'
        papers_with_authors_pct = 0
        papers_without_authors_pct = 0
        papers_with_abstracts_pct = 0
    
    readme_content = f"""---
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

- **Total Papers**: {total_papers if isinstance(total_papers, int) else total_papers}
- **Tracks**: Main, Datasets and Benchmarks
- **Award Types**: Oral, Spotlight, Poster
- **Source**: OpenReview (https://openreview.net/)

## Dataset Statistics

### Overview
- **Total Papers**: {total_papers if isinstance(total_papers, int) else total_papers}
- **Unique Paper IDs**: {stats.get('unique_paper_ids', 'N/A') if stats else 'N/A'}
- {duplicate_status}

### Track Distribution
{track_dist}

### Award Distribution
{award_dist}

### Track × Award Combinations
{track_award_dist}

### Author Statistics
- **Total Authors** (across all papers): {stats.get('total_authors', 0):,} if stats else 'N/A'
- **Unique Authors**: {stats.get('unique_authors', 0):,} if stats else 'N/A'
- **Average Authors per Paper**: {stats.get('avg_authors', 0):.2f} if stats else 'N/A'
- **Authors per Paper Range**: Min: {stats.get('min_authors', 0)} if stats else 'N/A', Max: {stats.get('max_authors', 0)} if stats else 'N/A', Avg: {stats.get('avg_authors', 0):.2f} if stats else 'N/A'
- **Papers with Authors**: {stats.get('papers_with_authors', 0):,} ({papers_with_authors_pct:.0f}%) if stats else 'N/A'

### Abstract Statistics
- **Papers with Abstracts**: {stats.get('papers_with_abstracts', 0):,} ({papers_with_abstracts_pct:.0f}%) if stats else 'N/A'
- **Average Abstract Length**: {stats.get('avg_abstract_length', 0):.0f} characters if stats else 'N/A'
- **Total Abstract Text**: {stats.get('total_abstract_length', 0):,} characters if stats else 'N/A'

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

dataset = load_dataset("{dataset_name}", split="train")
print(dataset[0])
```

## Citation

If you use this dataset, please cite the original NeurIPS 2025 conference and OpenReview.

## License

This dataset is provided for research purposes. Please refer to OpenReview's terms of service.
"""
    return readme_content


def upload_dataset(json_file, repo_id, username="huyxdang"):
    """Upload dataset to Hugging Face Hub"""
    full_repo_id = f"{username}/{repo_id}"
    
    print(f"📦 Preparing to upload dataset to: {full_repo_id}")
    print(f"   Source file: {json_file}")
    
    # Check if file exists
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"❌ Error: File not found: {json_file}")
        return False
    
    # Load data to get statistics
    print("📊 Loading data to calculate statistics...")
    papers = load_data(json_path)
    print(f"   Found {len(papers):,} papers")
    
    # Calculate comprehensive stats for README
    from collections import Counter
    tracks = Counter(p.get('track', 'Unknown') for p in papers)
    awards = Counter(p.get('award', 'Unknown') for p in papers)
    
    # Track × Award combinations
    track_award_combos = Counter(
        f"{p.get('track', 'Unknown')} - {p.get('award', 'Unknown')}" 
        for p in papers
    )
    
    # Author statistics
    total_authors = 0
    all_authors = []
    author_counts = []
    papers_with_authors = 0
    papers_without_authors = 0
    
    for paper in papers:
        authors = paper.get('authors', [])
        if isinstance(authors, list):
            author_count = len(authors)
            author_counts.append(author_count)
            all_authors.extend(authors)
            total_authors += author_count
            if author_count > 0:
                papers_with_authors += 1
            else:
                papers_without_authors += 1
        else:
            papers_without_authors += 1
    
    unique_authors = len(set(all_authors))
    avg_authors = total_authors / len(papers) if papers else 0
    min_authors = min(author_counts) if author_counts else 0
    max_authors = max(author_counts) if author_counts else 0
    
    # Abstract statistics
    total_abstract_length = 0
    papers_with_abstracts = 0
    papers_without_abstracts = 0
    
    for paper in papers:
        abstract = paper.get('abstract', '')
        if abstract and abstract != 'N/A':
            total_abstract_length += len(abstract)
            papers_with_abstracts += 1
        else:
            papers_without_abstracts += 1
    
    avg_abstract_length = (
        total_abstract_length / papers_with_abstracts 
        if papers_with_abstracts > 0 else 0
    )
    
    # Check for duplicate IDs
    paper_ids = [p.get('paper_id', '') for p in papers if p.get('paper_id')]
    unique_paper_ids = len(set(paper_ids))
    has_duplicates = len(paper_ids) != unique_paper_ids
    
    stats = {
        'total_papers': len(papers),
        'unique_paper_ids': unique_paper_ids,
        'has_duplicates': has_duplicates,
        'tracks': dict(tracks),
        'awards': dict(awards),
        'track_award_combos': dict(track_award_combos),
        'total_authors': total_authors,
        'unique_authors': unique_authors,
        'avg_authors': avg_authors,
        'min_authors': min_authors,
        'max_authors': max_authors,
        'papers_with_authors': papers_with_authors,
        'papers_without_authors': papers_without_authors,
        'total_abstract_length': total_abstract_length,
        'papers_with_abstracts': papers_with_abstracts,
        'papers_without_abstracts': papers_without_abstracts,
        'avg_abstract_length': avg_abstract_length
    }
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Check if repository exists, create if not
        print(f"\n🔍 Checking repository: {full_repo_id}")
        try:
            api.repo_info(repo_id=full_repo_id, repo_type="dataset")
            print(f"   ✅ Repository already exists")
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                print(f"   📝 Creating new repository...")
                create_repo(
                    repo_id=full_repo_id,
                    repo_type="dataset",
                    private=False,
                    exist_ok=False
                )
                print(f"   ✅ Repository created successfully")
            else:
                raise
        
        # Create README
        print(f"\n📝 Creating README.md...")
        readme_content = create_readme(repo_id, stats)
        readme_path = json_path.parent / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"   ✅ README created")
        
        # Upload JSON file
        print(f"\n⬆️  Uploading {json_path.name}...")
        upload_file(
            path_or_fileobj=str(json_path),
            path_in_repo=json_path.name,
            repo_id=full_repo_id,
            repo_type="dataset",
            commit_message=f"Upload NeurIPS 2025 papers dataset ({len(papers):,} papers)"
        )
        print(f"   ✅ JSON file uploaded")
        
        # Upload README
        print(f"\n⬆️  Uploading README.md...")
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="dataset",
            commit_message="Add dataset README"
        )
        print(f"   ✅ README uploaded")
        
        # Also upload CSV if it exists
        csv_path = json_path.with_suffix('.csv')
        if csv_path.exists():
            print(f"\n⬆️  Uploading {csv_path.name}...")
            upload_file(
                path_or_fileobj=str(csv_path),
                path_in_repo=csv_path.name,
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Upload CSV version of dataset"
            )
            print(f"   ✅ CSV file uploaded")
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Dataset uploaded to Hugging Face")
        print(f"   Repository: https://huggingface.co/datasets/{full_repo_id}")
        print(f"{'='*70}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error uploading dataset: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure you're logged in: huggingface-cli login")
        print(f"2. Check your internet connection")
        print(f"3. Verify the repository name is available")
        return False


def main():
    # Default file path
    default_file = Path(__file__).parent / 'neurips_2025_papers.json'
    default_repo_id = "neurips-2025-papers"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
    else:
        json_file = default_file
    
    if len(sys.argv) > 2:
        repo_id = sys.argv[2]
    else:
        repo_id = default_repo_id
    
    if not json_file.exists():
        print(f"❌ Error: File not found: {json_file}")
        print(f"\nUsage: python upload_to_huggingface.py [json_file] [repo_id]")
        print(f"  json_file: Path to JSON file (default: {default_file})")
        print(f"  repo_id: Repository ID (default: {default_repo_id})")
        sys.exit(1)
    
    success = upload_dataset(json_file, repo_id)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()