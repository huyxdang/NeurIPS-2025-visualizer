"""
Name clusters and meta-clusters using LLM analysis
Groups all papers in each cluster/meta-cluster and generates descriptive names
Uses Modal + Qwen 2.5 14B for fast, high-quality naming

For centroids: Groups all papers in each original cluster together
For meta-clusters: Groups all papers in each meta-cluster together
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
GPU_CONFIG = "A100-80GB"  # 14B fits on A100

SYSTEM_PROMPT_CLUSTER = """You are a research paper topic analyzer. Given titles and abstracts from papers in a cluster, identify the common research theme.

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

SYSTEM_PROMPT_META = """You are a research paper topic analyzer. Given titles and abstracts from papers in a meta-cluster (a larger grouping of related clusters), identify the broad research theme.

Generate a concise, descriptive meta-cluster name (2-4 words) that captures the overarching topic.

Respond ONLY with valid JSON in this exact format:
{
  "cluster_name": "Broad Topic Name (2-4 words)",
  "description": "Brief 1-sentence description of what this meta-cluster represents"
}

Be specific but broader than individual clusters. Examples of good names:
- "Computer Vision"
- "Natural Language Processing"
- "Reinforcement Learning"
- "Graph & Network Methods"
- "Generative Models"

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
    
    def format_prompt(self, titles: list[str], abstracts: list[str], cluster_id: int, size: int, is_meta: bool = False) -> str:
        """Format prompt for cluster naming"""
        # Create paper samples
        paper_samples = []
        for i, (title, abstract) in enumerate(zip(titles, abstracts), 1):
            # Truncate abstract to first 200 chars
            abstract_short = abstract[:200] + "..." if len(abstract) > 200 else abstract
            paper_samples.append(f"{i}. {title}\n   {abstract_short}")
        
        papers_text = "\n\n".join(paper_samples)
        
        cluster_type = "Meta-cluster" if is_meta else "Cluster"
        system_prompt = SYSTEM_PROMPT_META if is_meta else SYSTEM_PROMPT_CLUSTER
        
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{cluster_type} #{cluster_id} contains {size} research papers. Here are {len(titles)} sample papers:

{papers_text}

What research topic do these papers share? Generate a concise {cluster_type.lower()} name (2-5 words) and brief description.<|im_end|>
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
            cluster_data: List of dicts with keys: cluster_id, titles, abstracts, size, is_meta
            
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
                data['size'],
                data.get('is_meta', False)
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


def sample_papers_from_group(df: pd.DataFrame, cluster_col: str, cluster_id: int, n_samples: int = 15) -> dict:
    """
    Sample representative papers from a cluster or meta-cluster
    
    Args:
        df: DataFrame with papers
        cluster_col: Column name ('cluster' or 'meta_cluster')
        cluster_id: Cluster/meta-cluster ID to sample from
        n_samples: Number of papers to sample (use more for meta-clusters)
        
    Returns:
        Dict with titles, abstracts, size
    """
    group_papers = df[df[cluster_col] == cluster_id]
    size = len(group_papers)
    
    # Sample papers (or all if group is small)
    n_samples = min(n_samples, size)
    sampled = group_papers.sample(n=n_samples, random_state=42)
    
    # Get titles and abstracts
    # Try different column names
    title_col = 'paper' if 'paper' in sampled.columns else 'title' if 'title' in sampled.columns else None
    abstract_col = 'abstract' if 'abstract' in sampled.columns else None
    
    if title_col is None or abstract_col is None:
        print(f"⚠️  Warning: Could not find title/abstract columns for {cluster_col}={cluster_id}")
        return {
            'cluster_id': cluster_id,
            'titles': [],
            'abstracts': [],
            'size': size
        }
    
    return {
        'cluster_id': cluster_id,
        'titles': sampled[title_col].tolist(),
        'abstracts': sampled[abstract_col].tolist(),
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
    input_file: str = "5_cluster/data/neurips_2025_papers_with_meta_clusters.parquet",
    output_file: str = "6_naming/data/neurips_2025_papers_with_all_names.parquet",
    name_centroids: bool = True,
    name_meta_clusters: bool = True,
    n_samples_centroid: int = 12,
    n_samples_meta: int = 20,
    batch_size: int = 5
):
    """
    Main entrypoint for naming clusters and meta-clusters
    
    Args:
        input_file: Input parquet file with cluster and meta_cluster assignments
        output_file: Output parquet file with cluster names
        name_centroids: Whether to name original clusters (centroids)
        name_meta_clusters: Whether to name meta-clusters
        n_samples_centroid: Number of papers to sample per centroid cluster
        n_samples_meta: Number of papers to sample per meta-cluster
        batch_size: Number of clusters to name at once
    """
    print("="*70)
    print("Cluster & Meta-cluster Naming with Qwen 2.5 14B")
    print("="*70)
    
    # Load data
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"\n📖 Loading papers from {input_file}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df)} papers")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for required columns
    if name_centroids and 'cluster' not in df.columns:
        print("❌ Error: No 'cluster' column found. Cannot name centroids.")
        name_centroids = False
    
    if name_meta_clusters and 'meta_cluster' not in df.columns:
        print("❌ Error: No 'meta_cluster' column found. Cannot name meta-clusters.")
        name_meta_clusters = False
    
    if not name_centroids and not name_meta_clusters:
        print("❌ Error: Nothing to name!")
        return
    
    # Upload data to Modal
    print(f"\n📤 Uploading data to Modal...")
    with open(input_path, "rb") as f:
        file_bytes = f.read()
    remote_path = f"/data/{input_path.name}"
    upload_file.remote(file_bytes, remote_path)
    
    namer = ClusterNamer()
    
    # Name centroids (original clusters)
    if name_centroids:
        print(f"\n{'='*70}")
        print("Naming Centroids (Original Clusters)")
        print(f"{'='*70}")
        
        clusters = sorted(df[df['cluster'] >= 0]['cluster'].unique())
        n_clusters = len(clusters)
        
        print(f"📊 Found {n_clusters} clusters")
        print(f"   Sampling {n_samples_centroid} papers per cluster")
        
        # Sample papers from each cluster
        print(f"\n📝 Sampling papers from clusters...")
        cluster_samples = []
        for cluster_id in clusters:
            sample_data = sample_papers_from_group(df, 'cluster', cluster_id, n_samples_centroid)
            sample_data['is_meta'] = False
            cluster_samples.append(sample_data)
        
        # Name clusters in batches
        print(f"\n🚀 Generating cluster names with Qwen 2.5 14B...")
        print(f"   Processing {n_clusters} clusters in batches of {batch_size}")
        
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
        
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        info_file = output_dir / 'cluster_names.csv'
        cluster_info.to_csv(info_file, index=False)
        print(f"\n💾 Saved cluster names to {info_file}")
    
    # Name meta-clusters
    if name_meta_clusters:
        print(f"\n{'='*70}")
        print("Naming Meta-clusters")
        print(f"{'='*70}")
        
        meta_clusters = sorted(df[df['meta_cluster'] >= 0]['meta_cluster'].unique())
        n_meta_clusters = len(meta_clusters)
        
        print(f"📊 Found {n_meta_clusters} meta-clusters")
        print(f"   Sampling {n_samples_meta} papers per meta-cluster")
        
        # Sample papers from each meta-cluster
        print(f"\n📝 Sampling papers from meta-clusters...")
        meta_cluster_samples = []
        for meta_id in meta_clusters:
            sample_data = sample_papers_from_group(df, 'meta_cluster', meta_id, n_samples_meta)
            sample_data['is_meta'] = True
            meta_cluster_samples.append(sample_data)
        
        # Name meta-clusters in batches
        print(f"\n🚀 Generating meta-cluster names with Qwen 2.5 14B...")
        print(f"   Processing {n_meta_clusters} meta-clusters in batches of {batch_size}")
        
        all_meta_names = []
        for i in range(0, len(meta_cluster_samples), batch_size):
            batch = meta_cluster_samples[i:i+batch_size]
            batch_ids = [c['cluster_id'] for c in batch]
            
            print(f"   Naming meta-clusters {batch_ids}...")
            names = namer.name_clusters_batch.remote(batch)
            all_meta_names.extend(names)
        
        # Create meta-cluster name mapping
        meta_cluster_name_map = {}
        meta_cluster_desc_map = {}
        
        print(f"\n📊 Generated Meta-cluster Names:")
        print(f"{'Meta-Cluster':<15} {'Size':<8} {'Name':<40} Description")
        print("="*100)
        
        for meta_id, name_data in zip(meta_clusters, all_meta_names):
            meta_size = len(df[df['meta_cluster'] == meta_id])
            meta_cluster_name_map[meta_id] = name_data['cluster_name']
            meta_cluster_desc_map[meta_id] = name_data['description']
            
            print(f"{meta_id:<15} {meta_size:<8} {name_data['cluster_name']:<40} {name_data['description'][:50]}")
        
        # Add meta-cluster names to dataframe
        df['meta_cluster_name'] = df['meta_cluster'].map(meta_cluster_name_map).fillna('Unclustered')
        df['meta_cluster_description'] = df['meta_cluster'].map(meta_cluster_desc_map).fillna('')
        
        # Save meta-cluster names separately
        meta_info = pd.DataFrame({
            'meta_cluster': meta_clusters,
            'meta_cluster_name': [meta_cluster_name_map[m] for m in meta_clusters],
            'meta_cluster_description': [meta_cluster_desc_map[m] for m in meta_clusters],
            'size': [len(df[df['meta_cluster'] == m]) for m in meta_clusters]
        })
        
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        meta_info_file = output_dir / 'meta_cluster_names.csv'
        meta_info.to_csv(meta_info_file, index=False)
        print(f"\n💾 Saved meta-cluster names to {meta_info_file}")
    
    # Save main output
    print(f"\n💾 Saving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    print(f"✅ Saved {len(df)} papers with cluster names")
    
    # Statistics
    print("\n" + "="*70)
    print("✅ Naming complete!")
    print("="*70)
    print(f"\n📊 Output: {output_file}")
    print(f"   - Total papers: {len(df)}")
    
    if name_centroids:
        named_count = (df['cluster'] >= 0).sum()
        print(f"   - Papers with cluster names: {named_count}")
        print(f"   - Number of clusters: {n_clusters}")
    
    if name_meta_clusters:
        meta_named_count = (df['meta_cluster'] >= 0).sum()
        print(f"   - Papers with meta-cluster names: {meta_named_count}")
        print(f"   - Number of meta-clusters: {n_meta_clusters}")
    
    print(f"\n💡 Next: Use this file for visualization with cluster labels")


if __name__ == "__main__":
    # For testing
    pass

