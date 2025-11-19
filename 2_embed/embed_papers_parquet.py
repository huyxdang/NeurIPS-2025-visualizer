"""
Modal deployment for Embedding Gemma 300M
Fast GPU-based embedding generation in the cloud
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("neurips-embedding-gemma")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers",
        "torch",
        "pandas",
        "pyarrow",
        "tqdm",
        "numpy",
    )
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview"
    )
)

# Create persistent volume for storing data
volume = modal.Volume.from_name("neurips-papers-vol", create_if_missing=True)

# HuggingFace secret for authentication
secrets = modal.Secret.from_name("huggingface-secret")


@app.cls(
    image=image,
    gpu="A10G",  # Free tier GPU
    secrets=[secrets],
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
class EmbeddingGenerator:
    """Modal class for generating embeddings"""
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts"""
        import torch
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import login
        import os
        
        # Login to HuggingFace
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        
        print("Loading Embedding Gemma 300M model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("google/embeddinggemma-300M").to(device=self.device)
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @modal.method()
    def embed_batch(self, texts: list[str]) -> list:
        """
        Embed a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings as lists (JSON serializable)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    @modal.method()
    def process_parquet(self, input_path: str = "/data/neurips_2025_papers_full.parquet"):
        """
        Process entire parquet file and generate embeddings
        
        Args:
            input_path: Path to input parquet file in volume
            
        Returns:
            Path to output parquet file
        """
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        
        print(f"📖 Loading papers from {input_path}...")
        df = pd.read_parquet(input_path)
        print(f"✅ Loaded {len(df)} papers")
        
        # Initialize embeddings column
        df['embedding'] = None
        
        # Process in batches
        batch_size = 64  # Larger batches on GPU
        print(f"\n🚀 Generating embeddings (batch_size={batch_size})...")
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            abstracts = batch_df['abstract'].tolist()
            
            # Filter valid abstracts
            valid_abstracts = [
                abstract for abstract in abstracts 
                if abstract and abstract != "N/A" and pd.notna(abstract)
            ]
            
            if valid_abstracts:
                # Generate embeddings
                embeddings = self.model.encode(
                    valid_abstracts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Assign back to dataframe
                valid_idx = 0
                for j, abstract in enumerate(abstracts):
                    if abstract and abstract != "N/A" and pd.notna(abstract):
                        df.at[i+j, 'embedding'] = embeddings[valid_idx]
                        valid_idx += 1
        
        # Save output
        output_path = "/data/neurips_2025_papers_with_embeddings.parquet"
        print(f"\n💾 Saving to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        
        embeddings_count = df['embedding'].notna().sum()
        print(f"✅ Generated {embeddings_count} embeddings")
        
        return output_path


@app.local_entrypoint()
def main(
    input_file: str = "neurips_2025_papers_full.parquet",
    output_file: str = "neurips_2025_papers_with_embeddings.parquet"
):
    """
    Main entrypoint for running embedding generation
    
    Usage:
        modal run modal_embedding_gemma.py --input-file path/to/input.parquet
    """
    from pathlib import Path
    
    print("="*70)
    print("NeurIPS 2025 - Modal Embedding Generation")
    print("="*70)
    
    # Upload input file to volume
    local_input = Path(input_file)
    if not local_input.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"\n📤 Uploading {input_file} to Modal volume...")
    volume.reload()
    
    # Copy file to volume
    import pandas as pd
    df = pd.read_parquet(local_input)
    remote_path = f"/data/{local_input.name}"
    
    # We'll process it directly in Modal
    print(f"✅ File ready ({len(df)} papers)")
    
    # Create generator and process
    print(f"\n🚀 Starting embedding generation on Modal GPU...")
    generator = EmbeddingGenerator()
    
    # Upload the dataframe
    print("📤 Uploading data to Modal...")
    with volume.batch_upload() as batch:
        df.to_parquet("/tmp/input.parquet")
        batch.put_file("/tmp/input.parquet", remote_path)
    
    # Process on Modal
    output_path = generator.process_parquet.remote(remote_path)
    print(f"✅ Processing complete: {output_path}")
    
    # Download result
    print(f"\n📥 Downloading result to {output_file}...")
    volume.reload()
    
    with open(output_file, "wb") as f:
        for chunk in volume.read_file(output_path):
            f.write(chunk)
    
    print(f"✅ Saved to {output_file}")
    
    # Verify
    df_output = pd.read_parquet(output_file)
    embeddings_count = df_output['embedding'].notna().sum()
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)
    print(f"Total papers: {len(df_output)}")
    print(f"With embeddings: {embeddings_count}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    # For testing locally
    pass