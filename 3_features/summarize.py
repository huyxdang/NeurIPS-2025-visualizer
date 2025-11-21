import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in project root or current directory
env_paths = [
    Path(__file__).parent.parent / ".env",  # Project root
    Path(__file__).parent / ".env",  # Current directory
    Path(".env")  # Current working directory
]

loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📝 Loaded .env from: {env_path}")
        loaded = True
        break

if not loaded:
    print("⚠️  No .env file found. Trying environment variables...")
    load_dotenv()  # Try default locations

class SummaryGenerator:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.3):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("\n❌ OPENAI_API_KEY not found!")
            print("\nPlease set your OpenAI API key in one of these ways:")
            print("1. Create a .env file in the project root with:")
            print("   OPENAI_API_KEY=sk-proj-your-key-here")
            print("2. Or set it as an environment variable:")
            print("   export OPENAI_API_KEY=sk-proj-your-key-here")
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Validate API key format
        if not self.api_key.startswith("sk-"):
            print("⚠️  Warning: API key doesn't start with 'sk-'. Please verify it's correct.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        print(f"✅ OpenAI client initialized using {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   API key: {self.api_key[:10]}...{self.api_key[-4:]}")

    def get_system_prompt(self):
        return """You are summarizing machine learning papers in three distinct ways. 
Follow these instructions EXACTLY:

GENERAL RULES:
- Do NOT reuse sentence templates across the three sections.
- Do NOT begin different answers with the same phrase ("This paper..." / "The authors...")
- Use *different vocabulary* in each section.
- Avoid filler like "aims to," "seeks to," or "in this work."

SECTIONS TO OUTPUT:
1. **Problem (30-40 words)**
   - State the central challenge.
   - Mention why previous methods fail.
   - Be concrete about the pain point.
   - Do not start with generic openings (e.g., “The problem is…”, “This paper addresses…”). Begin directly with the issue.

2. **Solution (30-40 words)**
   - Describe the key idea.
   - Include what makes it different from existing methods.
   - Highlight at least ONE technical innovation.
   - Begin directly with the solution.

3. **ELI5 (30-40 words)**
   - Explain like I'm 10 years old using simple, literal language.
   - Keep the explanation directly about the topic (e.g., if the paper is about robots, talk about robots — not unrelated metaphors).
   - Avoid analogies or stories.
   - No ML jargon.
   - Be vivid and engaging.

STYLE:
- Vary tone across the three sections.
- Use richer vocabulary, not generic textbook style.
- Never copy phrases between Problem/Solution/ELI5.

EXAMPLE (FORMAT ONLY — DO NOT COPY WORDING):
Problem: GRPO introduces question-level difficulty bias because its group-relative advantage depends heavily on how easy or hard sampled questions are. This bias, combined with binary rewards and clipping-based objectives, leads to unstable training, poor discrimination between answer types, and weak handling of data imbalance.

Solution: DisCO replaces the group-relative formulation with a discriminative scoring objective that separately increases scores for correct answers and suppresses incorrect ones. It removes clipping, uses non-clipping RL surrogates as scoring functions, and applies constrained optimization to maintain the KL limit, producing more stable, unbiased training.

ELI5: The model is taught to score good answers higher and bad answers lower without comparing questions of different difficulty. It learns directly which answers are correct, keeps training steady, and avoids being confused when some questions are much easier or harder than others.

RETURN FORMAT:
Return a JSON object with a single key "results" containing a list of objects.
Each object must contain:
- "id": (The same ID provided in the input)
- "problem": "text..."
- "solution": "text..."
- "eli5": "text..."

Ensure the output is valid JSON."""

    def generate_bulk_summaries(self, papers_batch: list[dict]) -> dict:
        """
        Sends a batch of papers (e.g., 10) to OpenAI in a single request.
        """
        # Prepare the input JSON string
        input_data = {
            "papers": [
                {"id": p["id"], "title": p["title"], "abstract": p["abstract"]} 
                for p in papers_batch
            ]
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": json.dumps(input_data)}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"} # Enforce JSON structure
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Expecting {"results": [...]}
            results = parsed.get("results", [])
            
            # Convert list to dict keyed by ID for easy lookup
            return {item['id']: item for item in results}

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Incorrect API key" in error_msg:
                print(f"\n❌ Authentication Error (401): Invalid API key")
                print(f"   Please check your OPENAI_API_KEY in .env file")
                print(f"   Current key (masked): {self.api_key[:10]}...{self.api_key[-4:]}")
            else:
                print(f"⚠️  Batch generation error: {e}")
            return {}

    def process_parquet(self, input_path: str, output_path: str, batch_size: int = 10, test_mode: int = 0):
        """
        Main processing loop with checkpoint/resume functionality.
        batch_size: How many abstracts to pack into ONE prompt (user requested 10).
        test_mode: If > 0, only process that many rows.
        """
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            return

        # Checkpoint file path
        checkpoint_path = Path(output_path).parent / f"{Path(output_path).stem}_checkpoint.parquet"
        
        print(f"📖 Loading papers from {input_path}...")
        df = pd.read_parquet(input_path)
        
        # --- TEST MODE LOGIC ---
        if test_mode > 0:
            print(f"🧪 TEST MODE: Limiting to first {test_mode} papers.")
            df = df.head(test_mode).copy() # Create a copy to avoid SettingWithCopy warnings
        else:
            print(f"✅ Loaded {len(df)} papers")

        # Initialize columns if they don't exist
        for col in ['problem', 'solution', 'eli5']:
            if col not in df.columns:
                df[col] = None
        
        # Check for existing checkpoint
        if checkpoint_path.exists():
            print(f"\n📂 Found checkpoint file: {checkpoint_path.name}")
            print("   Resuming from checkpoint...")
            try:
                checkpoint_df = pd.read_parquet(checkpoint_path)
                
                # Merge checkpoint data back into main dataframe
                # Only update rows that have summaries in checkpoint
                for col in ['problem', 'solution', 'eli5']:
                    if col in checkpoint_df.columns:
                        # Update only non-null values from checkpoint
                        mask = checkpoint_df[col].notna()
                        if mask.any():
                            # Align indices and update
                            for idx in checkpoint_df[mask].index:
                                if idx in df.index:
                                    df.at[idx, col] = checkpoint_df.at[idx, col]
                
                # Count how many are already done
                completed = df['problem'].notna().sum()
                print(f"   ✅ Resumed: {completed}/{len(df)} papers already have summaries")
            except Exception as e:
                print(f"   ⚠️  Error loading checkpoint: {e}")
                print("   Starting fresh...")
        else:
            print(f"\n🆕 No checkpoint found. Starting fresh...")

        # Calculate number of API calls needed
        # We iterate by batch_size (10), so 50 papers = 5 API calls
        total_rows = len(df)
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        # Using tqdm to show progress bar
        print(f"\n🚀 Generating summaries (Packing {batch_size} papers per prompt)...")
        print(f"   Total papers: {total_rows}")
        print(f"   Total batches: {total_batches} (each batch = {batch_size} papers)")
        print(f"   Checkpoint file: {checkpoint_path.name}")
        
        processed_papers = df['problem'].notna().sum()  # Count already completed
        
        # We process the dataframe in chunks
        for batch_num, i in enumerate(tqdm(range(0, total_rows, batch_size), 
                                          total=total_batches, 
                                          desc=f"Batches (10 papers each)"), 1):
            
            # 1. Slice the dataframe
            batch_df = df.iloc[i : i + batch_size]
            
            # 2. Check if this batch is already complete (skip if all have summaries)
            batch_complete = True
            for idx in batch_df.index:
                if pd.isna(df.at[idx, 'problem']):
                    batch_complete = False
                    break
            
            if batch_complete:
                # Skip this batch - already processed
                processed_papers += len(batch_df)
                continue
            
            # 3. Prepare data for API
            # We use the dataframe index as the unique 'id' to map back later
            papers_payload = []
            valid_indices = []

            for idx, row in batch_df.iterrows():
                # Skip if already has summary
                if pd.notna(df.at[idx, 'problem']):
                    continue
                    
                abstract = row.get('abstract', '')
                title = row.get('paper', 'Unknown Title') # Assuming column name is 'paper' or 'title'
                
                # Skip empty abstracts
                if pd.isna(abstract) or abstract == "N/A" or len(str(abstract)) < 10:
                    df.at[idx, 'problem'] = "No Abstract"
                    df.at[idx, 'solution'] = "No Abstract"
                    df.at[idx, 'eli5'] = "No Abstract"
                    processed_papers += 1
                    continue
                
                papers_payload.append({
                    "id": idx,
                    "title": title,
                    "abstract": str(abstract)[:4000] # Truncate overly long abstracts to be safe
                })
                valid_indices.append(idx)

            if not papers_payload:
                # All in this batch were either done or had no abstract
                continue

            # 4. Call OpenAI
            results_map = self.generate_bulk_summaries(papers_payload)

            # 5. Update DataFrame
            for idx in valid_indices:
                if idx in results_map:
                    res = results_map[idx]
                    df.at[idx, 'problem'] = res.get('problem', 'Error')
                    df.at[idx, 'solution'] = res.get('solution', 'Error')
                    df.at[idx, 'eli5'] = res.get('eli5', 'Error')
                else:
                    # The model missed this specific ID in the batch response
                    df.at[idx, 'problem'] = 'Generation Failed'
                    df.at[idx, 'solution'] = 'Generation Failed'
                    df.at[idx, 'eli5'] = 'Generation Failed'
            
            processed_papers += len(valid_indices)
            
            # Save checkpoint after each batch
            try:
                df.to_parquet(checkpoint_path, engine='pyarrow', compression='snappy', index=False)
                
                # Show sample from this batch
                if valid_indices:
                    # Find a good sample (not an error)
                    sample_idx = None
                    for idx in valid_indices:
                        if df.at[idx, 'problem'] not in ['Error', 'Generation Failed', 'No Abstract']:
                            sample_idx = idx
                            break
                    
                    if sample_idx is not None:
                        sample_title = df.at[sample_idx, 'paper']
                        print(f"\n💾 Checkpoint saved | Batch {batch_num}/{total_batches}")
                        print(f"📄 Sample from this batch:")
                        print(f"   Title: {sample_title[:80]}...")
                        print(f"   Problem: {df.at[sample_idx, 'problem']}")
                        print(f"   Solution: {df.at[sample_idx, 'solution']}")
                        print(f"   ELI5: {df.at[sample_idx, 'eli5']}")
                    else:
                        print(f"\n💾 Checkpoint saved | Batch {batch_num}/{total_batches}")
            except Exception as e:
                print(f"\n⚠️  Warning: Could not save checkpoint: {e}")
            
            # Show progress every 5 batches
            if batch_num % 5 == 0 or batch_num == total_batches:
                print(f"\n📊 Progress: {processed_papers}/{total_rows} papers processed ({processed_papers*100//total_rows}%) | Batch {batch_num}/{total_batches}")

        # Save final output
        print(f"\n💾 Saving final output to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        
        # Clean up checkpoint file if everything is complete
        if checkpoint_path.exists():
            completed = df['problem'].notna().sum()
            if completed == len(df):
                checkpoint_path.unlink()
                print(f"   ✅ Checkpoint file removed (all papers processed)")
            else:
                print(f"   📂 Checkpoint file kept ({completed}/{len(df)} papers complete)")
        
        # Statistics
        success_count = df['problem'].apply(lambda x: x not in [None, 'Generation Failed', 'Error', 'No Abstract']).sum()
        print(f"\n✅ Done! Processed {len(df)} papers.")
        print(f"   Successfully summarized: {success_count}")
        print(f"   Output file: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeurIPS Paper Summarizer (Local)")
    parser.add_argument("--input", default="3_features/data/neurips_2025_papers_with_umap.parquet", help="Input parquet file")
    parser.add_argument("--output", default="3_features/data/neurips_2025_papers_with_summaries.parquet", help="Output parquet file")
    parser.add_argument("--batch", type=int, default=10, help="How many papers to group in one OpenAI prompt (default 10)")
    parser.add_argument("--test", type=int, default=0, help="If set > 0, runs only on N papers (e.g., --test 50)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (0.0-2.0, default 0.7)")
    
    args = parser.parse_args()
    
    # Validate temperature
    if not 0.0 <= args.temperature <= 2.0:
        print("⚠️  Warning: Temperature should be between 0.0 and 2.0. Using default 0.3.")
        args.temperature = 0.3
    
    generator = SummaryGenerator(model_name="gpt-4o-mini", temperature=args.temperature)
    generator.process_parquet(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch,
        test_mode=args.test
    )