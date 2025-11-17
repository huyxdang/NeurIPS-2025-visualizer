"""
Enhanced NeurIPS 2025 Paper Scraper
Includes filtering by award type and additional features
"""

import openreview
import json
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import argparse
import time


class NeurIPSScraper:
    """Scraper for NeurIPS 2025 papers from OpenReview"""
    
    def __init__(self):
        """Initialize OpenReview client"""
        print("Initializing OpenReview client...")
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        
        # Track configurations
        self.tracks = {
            'Main': {
                'venue': 'NeurIPS.cc/2025/Conference',
                'name': 'Main'
            },
            'Datasets and Benchmarks': {
                'venue': 'NeurIPS.cc/2025/Datasets_and_Benchmarks_Track',
                'name': 'Datasets and Benchmarks'
            }
        }
    
    def get_paper_metadata(self, note, track: str) -> Dict:
        """
        Extract metadata from a paper Note object
        
        Args:
            note: OpenReview Note object
            track: Track name
        
        Returns:
            Dictionary with paper metadata
        """
        # Extract basic info with safe fallbacks
        paper_data = {
            'paper': note.content.get('title', {}).get('value', 'N/A'),
            'authors': note.content.get('authors', {}).get('value', []),
            'abstract': note.content.get('abstract', {}).get('value', 'N/A'),
            'link': f"https://openreview.net/forum?id={note.id}",
            'track': track,
            'award': 'Poster',  # Default
            'paper_id': note.id
        }
        
        # Determine award type from venue field
        venue = note.content.get('venue', {}).get('value', '').lower()
        
        if 'oral' in venue:
            paper_data['award'] = 'Oral'
        elif 'spotlight' in venue:
            paper_data['award'] = 'Spotlight'
        elif 'poster' in venue:
            paper_data['award'] = 'Poster'
        
        return paper_data
    
    def scrape_track(self, track_name: str, award_filter: Optional[str] = None) -> List[Dict]:
        """
        Scrape ACCEPTED papers from a specific track
        
        Args:
            track_name: Name of the track ('Main' or 'Datasets and Benchmarks')
            award_filter: Optional filter for award type ('Oral', 'Spotlight', 'Poster')
        
        Returns:
            List of paper dictionaries (only accepted papers)
        """
        track_info = self.tracks[track_name]
        papers = []
        
        print(f"\n{'='*60}")
        print(f"Scraping {track_name} Track (ACCEPTED PAPERS ONLY)...")
        if award_filter:
            print(f"Filtering for: {award_filter}")
        print(f"{'='*60}")
        
        try:
            # Fetch all submissions from this venue
            submissions = self.client.get_all_notes(
                content={'venueid': track_info['venue']},
                details='directReplies'
            )
            
            print(f"Found {len(submissions)} total submissions")
            print("Filtering for accepted papers only...")
            
            accepted_count = 0
            rejected_count = 0
            
            # Process each paper
            for note in tqdm(submissions, desc=f"Processing {track_name}"):
                try:
                    # Check if paper is accepted by looking at venue field
                    venue = note.content.get('venue', {}).get('value', '').lower()
                    
                    # Skip if no venue or venue doesn't indicate acceptance
                    # Accepted papers have venue like "NeurIPS 2025 oral", "NeurIPS 2025 spotlight", "NeurIPS 2025 poster"
                    if not venue or 'neurips' not in venue:
                        rejected_count += 1
                        continue
                    
                    # Additional check: accepted papers should have oral, spotlight, or poster in venue
                    if not any(keyword in venue for keyword in ['oral', 'spotlight', 'poster']):
                        rejected_count += 1
                        continue
                    
                    accepted_count += 1
                    paper_data = self.get_paper_metadata(note, track_name)
                    
                    # Apply award filter if specified
                    if award_filter and paper_data['award'] != award_filter:
                        continue
                    
                    papers.append(paper_data)
                    
                except Exception as e:
                    print(f"\nWarning: Error processing paper {note.id}: {e}")
                    continue
            
            print(f"\nAccepted papers found: {accepted_count}")
            print(f"Rejected/Unprocessed: {rejected_count}")
            
            if award_filter:
                print(f"After filtering for {award_filter}: {len(papers)} papers")
            else:
                print(f"Total scraped: {len(papers)} accepted papers")
            
        except Exception as e:
            print(f"Error scraping {track_name} track: {e}")
        
        return papers
    
    def scrape_all(self, 
                   tracks: Optional[List[str]] = None,
                   award_filter: Optional[str] = None) -> List[Dict]:
        """
        Scrape papers from specified tracks
        
        Args:
            tracks: List of track names to scrape (None = all tracks)
            award_filter: Optional filter for award type
        
        Returns:
            List of all paper dictionaries
        """
        if tracks is None:
            tracks = list(self.tracks.keys())
        
        all_papers = []
        
        for track_name in tracks:
            if track_name not in self.tracks:
                print(f"Warning: Unknown track '{track_name}', skipping...")
                continue
            
            papers = self.scrape_track(track_name, award_filter)
            all_papers.extend(papers)
            
            # Small delay to be nice to the API
            time.sleep(0.5)
        
        return all_papers
    
    def save_results(self, papers: List[Dict], output_file: str):
        """
        Save scraped papers to JSON and CSV files
        
        Args:
            papers: List of paper dictionaries
            output_file: Base output filename (without extension)
        """
        if not papers:
            print("No papers to save!")
            return
        
        # Prepare filenames
        json_file = output_file if output_file.endswith('.json') else f"{output_file}.json"
        csv_file = json_file.replace('.json', '.csv')
        
        # Save to JSON
        print(f"\n{'='*60}")
        print(f"Saving {len(papers)} papers to {json_file}...")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        # Save to CSV
        df = pd.DataFrame(papers)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Also saved to {csv_file}")
        
        # Print summary
        self.print_summary(papers)
        
        print(f"\n{'='*60}")
        print("Done! ✅")
    
    def print_summary(self, papers: List[Dict]):
        """Print summary statistics of scraped papers"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total papers scraped: {len(papers)}")
        
        # Group by track
        tracks = {}
        for paper in papers:
            track = paper['track']
            if track not in tracks:
                tracks[track] = []
            tracks[track].append(paper)
        
        for track_name, track_papers in sorted(tracks.items()):
            print(f"\n{track_name} Track: {len(track_papers)} papers")
            
            # Count by award type
            awards = {}
            for paper in track_papers:
                award = paper['award']
                awards[award] = awards.get(award, 0) + 1
            
            for award, count in sorted(awards.items()):
                print(f"  - {award}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape NeurIPS 2025 papers from OpenReview',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all papers from all tracks
  python scrape.py
  
  # Scrape only Main Track
  python scrape.py --tracks Main
  
  # Scrape only Oral papers
  python scrape.py --award Oral
  
  # Scrape Main Track Spotlights only
  python scrape.py --tracks Main --award Spotlight
  
  # Custom output filename
  python scrape.py --output my_data.json
        """
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='neurips_2025_papers.json',
        help='Output JSON file path (default: neurips_2025_papers.json)'
    )
    
    parser.add_argument(
        '--tracks', '-t',
        nargs='+',
        choices=['Main', 'Datasets and Benchmarks'],
        help='Specific tracks to scrape (default: all tracks)'
    )
    
    parser.add_argument(
        '--award', '-a',
        choices=['Oral', 'Spotlight', 'Poster'],
        help='Filter by award type (default: all types)'
    )
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = NeurIPSScraper()
    
    # Scrape papers
    papers = scraper.scrape_all(
        tracks=args.tracks,
        award_filter=args.award
    )
    
    # Save results
    scraper.save_results(papers, args.output)


if __name__ == "__main__":
    main()