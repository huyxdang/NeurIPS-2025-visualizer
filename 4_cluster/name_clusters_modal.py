"""
Name clusters using LLM analysis
Samples abstracts from each cluster and generates descriptive names
Uses Modal + Qwen 2.5 14B for fast, high-quality naming
"""

import modal
import pandas as pd
import json
from pathlib import Path

# Create Modal app
app = modal.App("neurips-cluster-naming")

# vLLM image
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3.post1",
        "torch==2.4.0",
        "pandas",
        "pyarrow",
    )
)

# Volume for data
volume = modal.Volume.from_name("neurips-papers-vol", create_if_missing=True)

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # 14B is perfect for this task
GPU_CONFIG = "A100-80GB"  # 14B fits on A10G

SYSTEM_PROMPT = """You are a research paper topic analyzer. Given titles and abstracts from papers in a cluster, identify the common research theme.

Generate a concise, descriptive cluster name (2-5 words) that captures the main topic.

Respond ONLY with valid JSON in this exact format:
{
  "cluster_name": "Short Topic Name (2-5 words)",
  "description": "Brief 1-sentence description of what this cluster represents"
}

Be specific and use technical terms. Examples of good names:
- "Vision Transformers"
- "Reinforcement Learning Theory"
- "LLM Safety & Alignment"
- "Graph Neural Networks"
- "Diffusion Models"

Avoid generic names like "Machine Learning" or "AI Research"."""


@app.cls(
    image=vllm_image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=3600,
    scaledown_window=600,
)
class ClusterNamer:
    """Modal class for naming clusters with vLLM"""
    
    @modal.enter()
    def load_model(self):
        """Load vLLM model"""
        from vllm import LLM, SamplingParams
        
        print(f"Loading {MODEL_NAME} with vLLM...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.2,  # Low temperature for consistency
            top_p=0.9,
            max_tokens=200,
            stop=["```"]
        )
        
        print(f"✅ Model loaded successfully!")
    
    def format_prompt(self, titles: list[str], abstracts: list[str], cluster_id: int, size: int) -> str:
        """Format prompt for cluster naming"""
        # Create paper samples
        paper_samples = []
        for i, (title, abstract) in enumerate(zip(titles, abstracts), 1):
            # Truncate abstract to first 200 chars
            abstract_short = abstract[:200] + "..." if len(abstract) > 200 else abstract
            paper_samples.append(f"{i}. {title}\n   {abstract_short}")
        
        papers_text = "\n\n".join(paper_samples)
        
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Cluster #{cluster_id} contains {size} research papers. Here are {len(titles)} sample papers:

{papers_text}

What research topic do these papers share? Generate a concise cluster name (2-5 words) and brief description.<|im_end|>
<|im_start|>assistant
"""
    
    def parse_response(self, text: str, cluster_id: int) -> dict:
        """Parse JSON response from model"""
        try:
            # Remove markdown code blocks
            text = text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            result = json.loads(text)
            
            return {
                'cluster_name': result.get('cluster_name', f'Cluster {cluster_id}')[:100],
                'description': result.get('description', 'No description')[:300]
            }
        except json.JSONDecodeError:
            print(f"⚠️  JSON parse error for cluster {cluster_id}. Response: {text[:200]}")
            return {
                'cluster_name': f'Cluster {cluster_id}',
                'description': 'Failed to generate description'
            }
        except Exception as e:
            print(f"⚠️  Error parsing cluster {cluster_id}: {e}")
            return {
                'cluster_name': f'Cluster {cluster_id}',
                'description': 'Error'
            }
    
    @modal.method()
    def name_clusters_batch(self, cluster_data: list[dict]) -> list[dict]:
        """
        Name multiple clusters at once
        
        Args:
            cluster_data: List of dicts with keys: cluster_id, titles, abstracts, size
            
        Returns:
            List of dicts with cluster_name and description
        """
        # Format prompts
        prompts = []
        for data in cluster_data:
            prompt = self.format_prompt(
                data['titles'],
                data['abstracts'],
                data['cluster_id'],
                data['size']
            )
            prompts.append(prompt)
        
        # Generate with vLLM (batched)
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Parse responses
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            parsed = self.parse_response(generated_text, cluster_data[i]['cluster_id'])
            results.append(parsed)
        
        return results


def sample_papers_from_cluster(df: pd.DataFrame, cluster_id: int, n_samples: int = 8) -> dict:
    """
    Sample representative papers from a cluster
    
    Args:
        df: DataFrame with papers
        cluster_id: Cluster to sample from
        n_samples: Number of papers to sample
        
    Returns:
        Dict with titles, abstracts, size
    """
    cluster_papers = df[df['cluster'] == cluster_id]
    size = len(cluster_papers)
    
    # Sample papers (or all if cluster is small)
    n_samples = min(n_samples, size)
    sampled = cluster_papers.sample(n=n_samples, random_state=42)
    
    return {
        'cluster_id': cluster_id,
        'titles': sampled['paper'].tolist(),
        'abstracts': sampled['abstract'].tolist(),
        'size': size
    }


@app.function(
    image=vllm_image,
    volumes={"/data": volume},
    timeout=300,
)
def upload_file(file_bytes: bytes, remote_path: str):
    """Upload file to Modal volume"""
    with open(remote_path, "wb") as f:
        f.write(file_bytes)
    volume.commit()
    return remote_path


@app.function(
    image=vllm_image,
    volumes={"/data": volume},
    timeout=300,
)
def download_file(remote_path: str) -> bytes:
    """Download file from Modal volume"""
    with open(remote_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    input_file: str = "4_cluster/data/neurips_2025_papers_with_clusters.parquet",
    output_file: str = "4_cluster/data/neurips_2025_papers_with_cluster_names.parquet",
    n_samples: int = 8,
    batch_size: int = 5
):
    """
    Main entrypoint for cluster naming
    
    Args:
        input_file: Input parquet file with cluster assignments
        output_file: Output parquet file with cluster names
        n_samples: Number of papers to sample per cluster
        batch_size: Number of clusters to name at once
    """
    print("="*70)
    print("Cluster Naming with Qwen 2.5 14B")
    print("="*70)
    
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"\n📖 Loading papers from {input_file}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df)} papers")
    
    # Check for clusters
    if 'cluster' not in df.columns:
        print("❌ Error: No cluster assignments found. Run cluster_papers.py first.")
        return
    
    # Get unique clusters
    clusters = sorted(df[df['cluster'] >= 0]['cluster'].unique())
    n_clusters = len(clusters)
    
    print(f"📊 Found {n_clusters} clusters")
    print(f"   Sampling {n_samples} papers per cluster")
    
    # Sample papers from each cluster
    print(f"\n📝 Sampling papers from clusters...")
    cluster_samples = []
    for cluster_id in clusters:
        sample_data = sample_papers_from_cluster(df, cluster_id, n_samples)
        cluster_samples.append(sample_data)
    
    # Upload data to Modal
    print(f"\n📤 Uploading data to Modal...")
    with open(input_path, "rb") as f:
        file_bytes = f.read()
    remote_path = f"/data/{input_path.name}"
    upload_file.remote(file_bytes, remote_path)
    
    # Name clusters in batches
    print(f"\n🚀 Generating cluster names with Qwen 2.5 14B...")
    print(f"   Processing {n_clusters} clusters in batches of {batch_size}")
    
    namer = ClusterNamer()
    all_names = []
    
    for i in range(0, len(cluster_samples), batch_size):
        batch = cluster_samples[i:i+batch_size]
        batch_ids = [c['cluster_id'] for c in batch]
        
        print(f"   Naming clusters {batch_ids}...")
        names = namer.name_clusters_batch.remote(batch)
        all_names.extend(names)
    
    # Create cluster name mapping
    cluster_name_map = {}
    cluster_desc_map = {}
    
    print(f"\n📊 Generated Cluster Names:")
    print(f"{'Cluster':<8} {'Size':<6} {'Name':<40} Description")
    print("="*100)
    
    for cluster_id, name_data in zip(clusters, all_names):
        cluster_size = len(df[df['cluster'] == cluster_id])
        cluster_name_map[cluster_id] = name_data['cluster_name']
        cluster_desc_map[cluster_id] = name_data['description']
        
        print(f"{cluster_id:<8} {cluster_size:<6} {name_data['cluster_name']:<40} {name_data['description'][:50]}")
    
    # Add cluster names to dataframe
    df['cluster_name'] = df['cluster'].map(cluster_name_map).fillna('Unclustered')
    df['cluster_description'] = df['cluster'].map(cluster_desc_map).fillna('')
    
    # Save cluster names separately
    cluster_info = pd.DataFrame({
        'cluster': clusters,
        'cluster_name': [cluster_name_map[c] for c in clusters],
        'cluster_description': [cluster_desc_map[c] for c in clusters],
        'size': [len(df[df['cluster'] == c]) for c in clusters]
    })
    
    info_file = Path(output_file).with_name('cluster_names.csv')
    cluster_info.to_csv(info_file, index=False)
    print(f"\n💾 Saved cluster names to {info_file}")
    
    # Save main output
    print(f"\n💾 Saving to {output_file}...")
    output_path = Path(output_file)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    print(f"✅ Saved {len(df)} papers with cluster names")
    
    # Statistics
    named_count = (df['cluster'] >= 0).sum()
    
    print("\n" + "="*70)
    print("✅ Cluster naming complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(df)}")
    print(f"   - Papers with cluster names: {named_count}")
    print(f"   - Number of clusters: {n_clusters}")
    print(f"\n💡 Next: Use this file for visualization with cluster labels")


if __name__ == "__main__":
    # For testing
    pass
