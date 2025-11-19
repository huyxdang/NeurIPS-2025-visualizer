"""
Generate problem/solution/ELI5 summaries for each paper using LLM
Adds three new columns to the parquet file: problem, solution, eli5
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI()

# Configuration
INPUT_FILE = Path("neurips_2025_papers_with_umap.parquet")
OUTPUT_FILE = Path("neurips_2025_papers_with_summaries.parquet")
PROGRESS_FILE = Path("summary_progress.parquet")
BATCH_DELAY = 0.2  # delay between API calls
SAVE_INTERVAL = 10  # save progress every N papers
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a research paper summarizer. Given an abstract, extract:
1. Problem: What problem does this paper address? (max 40 words)
2. Solution: What is their approach/solution? (max 40 words)  
3. ELI5: Explain like I'm 5 - make it simple and fun (max 40 words)

Respond ONLY with valid JSON in this exact format:
{
  "problem": "text here",
  "solution": "text here", 
  "eli5": "text here"
}

Keep each field under 40 words. Be concise and clear."""


def get_summary_with_retry(abstract, max_retries=MAX_RETRIES):
    """Get LLM summary with retry logic"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Faster and cheaper than gpt-4
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Abstract:\n\n{abstract}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
                
            result = json.loads(content)
            
            return {
                'problem': result.get('problem', 'N/A')[:300],  # Safety limit
                'solution': result.get('solution', 'N/A')[:300],
                'eli5': result.get('eli5', 'N/A')[:300]
            }
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)
                print(f"\n⚠️  Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Rate limit after {max_retries} attempts: {e}")
                
        except APIError as e:
            error_message = str(e)
            if 'quota' in error_message.lower():
                raise Exception(f"QUOTA_EXCEEDED: {error_message}")
            elif attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\n⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"API error after {max_retries} attempts: {e}")
                
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parse error. Response: {content[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                # Return fallback
                return {
                    'problem': 'Failed to parse',
                    'solution': 'Failed to parse',
                    'eli5': 'Failed to parse'
                }
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise Exception(f"Unexpected error: {e}")
    
    return None


def load_progress():
    """Load existing progress if available"""
    if PROGRESS_FILE.exists():
        print(f"📂 Found progress file: {PROGRESS_FILE}")
        df = pd.read_parquet(PROGRESS_FILE)
        print(f"   Loaded {len(df)} previously processed papers")
        return df
    return None


def save_progress(df):
    """Save progress to Parquet"""
    df.to_parquet(PROGRESS_FILE, engine='pyarrow', compression='snappy', index=False)


print("="*70)
print("NeurIPS 2025 Papers - Summary Generation")
print("="*70)

# Load input data
print(f"\n📖 Loading papers from {INPUT_FILE}...")
if not INPUT_FILE.exists():
    print(f"❌ File not found: {INPUT_FILE}")
    exit(1)

df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded {len(df)} papers")

# Load progress if exists
progress_df = load_progress()

if progress_df is not None:
    # Merge with existing progress
    processed_ids = set(progress_df[progress_df['problem'].notna()]['paper_id'].tolist())
    print(f"📊 Progress: {len(processed_ids)}/{len(df)} papers already processed")
    
    # Merge progress data
    df = df.merge(
        progress_df[['paper_id', 'problem', 'solution', 'eli5']], 
        on='paper_id', 
        how='left',
        suffixes=('', '_progress')
    )
    
    # Use progress data where available
    for col in ['problem', 'solution', 'eli5']:
        if f'{col}_progress' in df.columns:
            df[col] = df[f'{col}_progress'].combine_first(df.get(col))
            df = df.drop(columns=[f'{col}_progress'])
else:
    print("🆕 Starting fresh (no previous progress found)")
    df['problem'] = None
    df['solution'] = None
    df['eli5'] = None
    processed_ids = set()

# Generate summaries
print("\n🚀 Generating summaries...")
print(f"   Using model: gpt-4o-mini")
print(f"   Rate: ~{1/BATCH_DELAY:.0f} papers/second")
print(f"   Estimated time: ~{len(df) * BATCH_DELAY / 60:.0f} minutes")

errors = []
processed_count = len(processed_ids)

try:
    # Filter to only papers without summaries
    mask_needs_summary = df['problem'].isna()
    indices_to_process = df[mask_needs_summary].index.tolist()
    
    print(f"📝 {len(indices_to_process)} papers need summaries")
    
    for idx in tqdm(indices_to_process, desc="Generating summaries", initial=processed_count):
        abstract = df.at[idx, 'abstract']
        
        if abstract and abstract != "N/A" and pd.notna(abstract):
            try:
                # Generate summary
                summary = get_summary_with_retry(abstract)
                
                df.at[idx, 'problem'] = summary['problem']
                df.at[idx, 'solution'] = summary['solution']
                df.at[idx, 'eli5'] = summary['eli5']
                
                processed_count += 1
                
                # Save progress periodically
                if processed_count % SAVE_INTERVAL == 0:
                    save_progress(df)
                
                # Delay to avoid rate limits
                time.sleep(BATCH_DELAY)
                
            except Exception as e:
                error_msg = str(e)
                errors.append({
                    "index": idx, 
                    "paper_id": df.at[idx, 'paper_id'], 
                    "error": error_msg
                })
                print(f"\n❌ Error at index {idx}: {error_msg}")
                
                # Handle quota errors
                if 'QUOTA_EXCEEDED' in error_msg or 'quota' in error_msg.lower():
                    print("\n💾 Saving progress before exiting...")
                    save_progress(df)
                    print(f"✅ Progress saved: {processed_count}/{len(df)} papers")
                    print("\n❌ Quota exceeded. Run again later to resume.")
                    exit(1)
                continue
        else:
            # Mark as processed even if no abstract
            df.at[idx, 'problem'] = 'No abstract available'
            df.at[idx, 'solution'] = 'No abstract available'
            df.at[idx, 'eli5'] = 'No abstract available'
    
    # Final save
    save_progress(df)
    
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user. Saving progress...")
    save_progress(df)
    print(f"✅ Progress saved: {processed_count}/{len(df)} papers")
    exit(0)

# Count successful summaries
summaries_count = df['problem'].notna().sum()
print(f"\n✅ Generated {summaries_count} summaries")

# Save final output
print(f"\n💾 Saving final output to {OUTPUT_FILE}...")
df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy', index=False)

print(f"✅ Saved {len(df)} papers with summaries to {OUTPUT_FILE}")

# Clean up progress file if fully complete
if PROGRESS_FILE.exists() and summaries_count == len(df):
    PROGRESS_FILE.unlink()
    print("✅ Cleaned up progress file (all papers processed)")

if errors:
    print(f"\n⚠️  {len(errors)} errors encountered")
    print("First few errors:")
    for error in errors[:5]:
        print(f"  - Paper {error['paper_id']}: {error['error']}")

print("\n" + "="*70)
print("✅ Summary generation complete!")
print("="*70)
print(f"\n📊 Output: {OUTPUT_FILE}")
print(f"   - Total papers: {len(df)}")
print(f"   - With summaries: {summaries_count}")
print(f"   - Columns: {list(df.columns)}")