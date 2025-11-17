"""
Quick test script to verify OpenReview API is working
"""

import openreview

def test_connection():
    """Test OpenReview API connection"""
    print("Testing OpenReview API connection...")
    print("-" * 50)
    
    try:
        # Initialize client
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        print("✅ Successfully connected to OpenReview API")
        
        # Test fetching a small sample from NeurIPS 2025
        print("\nFetching sample papers from NeurIPS 2025 Main Track...")
        
        # Fetch all submissions (API doesn't support limit parameter)
        # We'll just take the first 5 for testing
        all_submissions = client.get_all_notes(
            content={'venueid': 'NeurIPS.cc/2025/Conference'}
        )
        
        # Take first 5 for testing
        submissions = all_submissions[:5]
        
        print(f"✅ Successfully fetched {len(all_submissions)} total papers")
        print(f"   Showing first {len(submissions)} as sample:")
        
        if submissions:
            sample = submissions[0]
            authors = sample.content.get('authors', {}).get('value', [])
            authors_str = ', '.join(authors) if authors else 'N/A'
            # Truncate if too long
            if len(authors_str) > 100:
                authors_str = authors_str[:97] + '...'
            
            print(f"\nSample paper:")
            print(f"  Title: {sample.content.get('title', {}).get('value', 'N/A')[:80]}...")
            print(f"  Authors ({len(authors)}): {authors_str}")
            print(f"  Venue: {sample.content.get('venue', {}).get('value', 'N/A')}")
            print(f"  Link: https://openreview.net/forum?id={sample.id}")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! You're ready to scrape.")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify openreview-py is installed: pip install openreview-py")
        print("3. Try again in a few minutes (API might be rate limiting)")
        return False

if __name__ == "__main__":
    test_connection()