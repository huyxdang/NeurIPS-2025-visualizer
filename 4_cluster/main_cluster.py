"""
Orchestrator for Clustering Pipeline
Runs K-means clustering followed by LLM-based cluster naming
"""

import subprocess
import sys
from pathlib import Path
import argparse

class ClusteringPipeline:
    """Orchestrates the complete clustering workflow"""
    
    def __init__(self, use_modal: bool = True):
        # Use the directory where this script is located
        self.base_dir = Path(__file__).parent
        self.use_modal = use_modal
        
        self.scripts = {
            'cluster': self.base_dir / 'cluster_papers.py',
            'name': self.base_dir / 'name_clusters_modal.py'
        }
        
        self.files = {
            'input': '4_cluster/data/neurips_2025_papers_with_umap.parquet',
            'clustered': '4_cluster/data/neurips_2025_papers_with_clusters.parquet',
            'named': '4_cluster/data/neurips_2025_papers_with_cluster_names.parquet'
        }
    
    def check_file_exists(self, file_key: str) -> bool:
        """Check if a file exists"""
        file_path = Path(self.files[file_key])
        return file_path.exists()
    
    def run_step(self, script_name: str, args: list = None):
        """Run a pipeline step"""
        script_path = self.scripts[script_name]
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        # Build command
        if self.use_modal and script_name == 'name':
            cmd = ['modal', 'run', str(script_path)]
        else:
            cmd = [sys.executable, str(script_path)]
        
        if args:
            cmd.extend(args)
        
        print(f"\n▶️  Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Step failed with error code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            return False
    
    def step_cluster(self, n_clusters: int = None, auto_k: bool = True, force: bool = False):
        """Step 1: K-means clustering"""
        print("\n" + "="*70)
        print("STEP 1: K-means Clustering")
        print("="*70)
        
        if not force and self.check_file_exists('clustered'):
            print(f"✅ Clustered file already exists: {self.files['clustered']}")
            print("   Use --force to regenerate")
            return True
        
        if not self.check_file_exists('input'):
            print(f"❌ Input file not found: {self.files['input']}")
            print("   Run UMAP first to generate this file")
            return False
        
        # Build arguments
        args = [
            '--input', self.files['input'],
            '--output', self.files['clustered']
        ]
        
        if n_clusters:
            args.extend(['--n-clusters', str(n_clusters)])
        
        if auto_k:
            args.append('--auto-k')
        else:
            args.append('--no-auto-k')
        
        return self.run_step('cluster', args)
    
    def step_name(self, n_samples: int = 8, batch_size: int = 5, force: bool = False):
        """Step 2: Name clusters"""
        print("\n" + "="*70)
        mode = "Modal GPU" if self.use_modal else "Local"
        print(f"STEP 2: Name Clusters ({mode})")
        print("="*70)
        
        if not force and self.check_file_exists('named'):
            print(f"✅ Named file already exists: {self.files['named']}")
            print("   Use --force to regenerate")
            return True
        
        if not self.check_file_exists('clustered'):
            print(f"❌ Clustered file not found: {self.files['clustered']}")
            print("   Run step 1 first (--step cluster)")
            return False
        
        # Build arguments (Modal script uses different arg names)
        args = [
            '--input-file', self.files['clustered'],
            '--output-file', self.files['named'],
            '--n-samples', str(n_samples),
            '--batch-size', str(batch_size)
        ]
        
        return self.run_step('name', args)
    
    def run_all(
        self,
        n_clusters: int = None,
        auto_k: bool = True,
        n_samples: int = 8,
        batch_size: int = 5,
        force: bool = False
    ):
        """Run the complete pipeline"""
        print("\n" + "🎯 "*35)
        mode = "Modal GPU" if self.use_modal else "Local"
        print(f"CLUSTERING PIPELINE ({mode})")
        print("🎯 "*35)
        
        steps = [
            ('cluster', lambda: self.step_cluster(n_clusters, auto_k, force)),
            ('name', lambda: self.step_name(n_samples, batch_size, force))
        ]
        
        for step_name, step_func in steps:
            print(f"\n▶️  Running: {step_name}")
            success = step_func()
            
            if not success:
                print(f"\n❌ Pipeline failed at step: {step_name}")
                return False
        
        print("\n" + "🎉 "*35)
        print("CLUSTERING PIPELINE COMPLETE!")
        print("🎉 "*35)
        
        # Show final output location
        final_file = self.files['named']
        print(f"\n✅ Final output: {final_file}")
        
        # Show summary
        try:
            import pandas as pd
            df = pd.read_parquet(final_file)
            n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)
            
            print(f"\n📊 Summary:")
            print(f"   - Total papers: {len(df)}")
            print(f"   - Number of clusters: {n_clusters}")
            print(f"   - Papers with clusters: {(df['cluster'] >= 0).sum()}")
            
            # Show sample cluster names
            if 'cluster_name' in df.columns:
                cluster_names = df[df['cluster'] >= 0].groupby('cluster')['cluster_name'].first()
                print(f"\n📋 Sample cluster names:")
                for i, (cluster_id, name) in enumerate(cluster_names.head(5).items()):
                    print(f"   Cluster {cluster_id}: {name}")
                if len(cluster_names) > 5:
                    print(f"   ... and {len(cluster_names) - 5} more")
        except Exception as e:
            print(f"⚠️  Could not load summary: {e}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Clustering Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with auto k-selection
  python orchestrate_clustering.py --all --auto-k
  
  # Run with specific number of clusters
  python orchestrate_clustering.py --all --n-clusters 25
  
  # Run specific step
  python orchestrate_clustering.py --step cluster --n-clusters 20
  python orchestrate_clustering.py --step name
  
  # Force regeneration
  python orchestrate_clustering.py --all --force

Pipeline Steps:
  1. cluster - K-means clustering on UMAP coordinates
  2. name    - LLM-based cluster naming (Modal GPU)

Recommended Settings:
  - Auto k-selection: Good for exploration
  - k=20-30: Good balance for 5,772 papers
  - n_samples=8: Enough for LLM to understand cluster
  - batch_size=5: Balance speed and GPU memory
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--step', choices=['cluster', 'name'],
                       help='Run a specific step')
    
    # Clustering options
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters (if not auto-detecting)')
    parser.add_argument('--auto-k', action='store_true',
                       help='Automatically find optimal k')
    parser.add_argument('--no-auto-k', dest='auto_k', action='store_false',
                       help='Disable automatic k detection')
    parser.set_defaults(auto_k=True)
    
    # Naming options
    parser.add_argument('--n-samples', type=int, default=8,
                       help='Number of papers to sample per cluster for naming')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of clusters to name at once')
    
    # General options
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of existing files')
    
    # Input/output
    parser.add_argument('--input-file', default=None,
                       help='Input file with UMAP coordinates (default: 4_cluster/data/neurips_2025_papers_with_umap.parquet)')
    parser.add_argument('--output-file', default=None,
                       help='Final output file (default: 4_cluster/data/neurips_2025_papers_with_cluster_names.parquet)')
    
    args = parser.parse_args()
    
    if not args.all and not args.step:
        parser.print_help()
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = ClusteringPipeline(use_modal=True)  # Always use Modal for best performance
    
    # Update file paths only if custom values provided
    if args.input_file is not None:
        orchestrator.files['input'] = args.input_file
    if args.output_file is not None:
        orchestrator.files['named'] = args.output_file
    
    # Run pipeline
    if args.all:
        success = orchestrator.run_all(
            n_clusters=args.n_clusters,
            auto_k=args.auto_k,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            force=args.force
        )
        sys.exit(0 if success else 1)
    
    elif args.step:
        if args.step == 'cluster':
            success = orchestrator.step_cluster(args.n_clusters, args.auto_k, args.force)
        elif args.step == 'name':
            success = orchestrator.step_name(args.n_samples, args.batch_size, args.force)
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
