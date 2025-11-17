"""
Analyze and display summary statistics for NeurIPS 2025 papers dataset
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_data(json_file):
    """Load papers from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_statistics(papers):
    """Calculate comprehensive statistics"""
    stats = {
        'total_papers': len(papers),
        'tracks': Counter(),
        'awards': Counter(),
        'track_award_combinations': Counter(),
        'total_authors': 0,
        'papers_with_authors': 0,
        'papers_without_authors': 0,
        'total_abstract_length': 0,
        'papers_with_abstracts': 0,
        'papers_without_abstracts': 0,
        'unique_paper_ids': set(),
        'duplicate_ids': []
    }
    
    # Track author statistics
    author_counts = []
    all_authors = []
    
    for paper in papers:
        # Track distribution
        track = paper.get('track', 'Unknown')
        stats['tracks'][track] += 1
        
        # Award distribution
        award = paper.get('award', 'Unknown')
        stats['awards'][award] += 1
        
        # Track-Award combinations
        combo = f"{track} - {award}"
        stats['track_award_combinations'][combo] += 1
        
        # Authors
        authors = paper.get('authors', [])
        if isinstance(authors, list):
            author_count = len(authors)
            author_counts.append(author_count)
            all_authors.extend(authors)
            stats['total_authors'] += author_count
            if author_count > 0:
                stats['papers_with_authors'] += 1
            else:
                stats['papers_without_authors'] += 1
        else:
            stats['papers_without_authors'] += 1
        
        # Abstracts
        abstract = paper.get('abstract', '')
        if abstract and abstract != 'N/A':
            stats['total_abstract_length'] += len(abstract)
            stats['papers_with_abstracts'] += 1
        else:
            stats['papers_without_abstracts'] += 1
        
        # Check for duplicates
        paper_id = paper.get('paper_id', '')
        if paper_id:
            if paper_id in stats['unique_paper_ids']:
                stats['duplicate_ids'].append(paper_id)
            else:
                stats['unique_paper_ids'].add(paper_id)
    
    # Calculate averages
    stats['avg_authors_per_paper'] = (
        stats['total_authors'] / stats['total_papers'] 
        if stats['total_papers'] > 0 else 0
    )
    stats['avg_abstract_length'] = (
        stats['total_abstract_length'] / stats['papers_with_abstracts']
        if stats['papers_with_abstracts'] > 0 else 0
    )
    stats['unique_author_count'] = len(set(all_authors))
    stats['author_counts'] = {
        'min': min(author_counts) if author_counts else 0,
        'max': max(author_counts) if author_counts else 0,
        'avg': sum(author_counts) / len(author_counts) if author_counts else 0
    }
    stats['unique_paper_ids_count'] = len(stats['unique_paper_ids'])
    
    return stats


def print_statistics(stats):
    """Print formatted statistics"""
    print("=" * 70)
    print("NEURIPS 2025 PAPERS - SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 OVERVIEW")
    print(f"  Total Papers: {stats['total_papers']:,}")
    print(f"  Unique Paper IDs: {stats['unique_paper_ids_count']:,}")
    if stats['duplicate_ids']:
        print(f"  ⚠️  Duplicate IDs found: {len(stats['duplicate_ids'])}")
        print(f"     Duplicates: {stats['duplicate_ids'][:5]}")
    else:
        print(f"  ✅ No duplicate IDs")
    
    print(f"\n📚 TRACK DISTRIBUTION")
    for track, count in stats['tracks'].most_common():
        percentage = (count / stats['total_papers']) * 100
        print(f"  {track:30s}: {count:5,} papers ({percentage:5.1f}%)")
    
    print(f"\n🏆 AWARD DISTRIBUTION")
    for award, count in stats['awards'].most_common():
        percentage = (count / stats['total_papers']) * 100
        print(f"  {award:30s}: {count:5,} papers ({percentage:5.1f}%)")
    
    print(f"\n📋 TRACK × AWARD COMBINATIONS")
    for combo, count in stats['track_award_combinations'].most_common():
        percentage = (count / stats['total_papers']) * 100
        print(f"  {combo:50s}: {count:5,} papers ({percentage:5.1f}%)")
    
    print(f"\n👥 AUTHOR STATISTICS")
    print(f"  Total Authors (all papers): {stats['total_authors']:,}")
    print(f"  Unique Authors: {stats['unique_author_count']:,}")
    print(f"  Average Authors per Paper: {stats['avg_authors_per_paper']:.2f}")
    print(f"  Authors per Paper - Min: {stats['author_counts']['min']}, "
          f"Max: {stats['author_counts']['max']}, "
          f"Avg: {stats['author_counts']['avg']:.2f}")
    print(f"  Papers with Authors: {stats['papers_with_authors']:,}")
    print(f"  Papers without Authors: {stats['papers_without_authors']:,}")
    
    print(f"\n📝 ABSTRACT STATISTICS")
    print(f"  Papers with Abstracts: {stats['papers_with_abstracts']:,}")
    print(f"  Papers without Abstracts: {stats['papers_without_abstracts']:,}")
    if stats['papers_with_abstracts'] > 0:
        print(f"  Average Abstract Length: {stats['avg_abstract_length']:.0f} characters")
        print(f"  Total Abstract Text: {stats['total_abstract_length']:,} characters")
    
    print("\n" + "=" * 70)


def main():
    # Default file path
    default_file = Path(__file__).parent / 'neurips_2025_papers.json'
    
    # Allow command line argument for file path
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
    else:
        json_file = default_file
    
    if not json_file.exists():
        print(f"❌ Error: File not found: {json_file}")
        print(f"Usage: python analyze_data.py [path_to_json_file]")
        sys.exit(1)
    
    print(f"Loading data from: {json_file}")
    papers = load_data(json_file)
    
    print(f"Analyzing {len(papers):,} papers...")
    stats = calculate_statistics(papers)
    
    print_statistics(stats)


if __name__ == "__main__":
    main()