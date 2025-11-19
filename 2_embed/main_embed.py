"""
Master Pipeline for NeurIPS 2025 Paper Processing
Orchestrates the complete workflow from JSON to final visualization-ready data

Workflow:
1. Convert JSON to Parquet
2. Generate embeddings
3. Apply UMAP dimensionality reduction
4. (Optional) Convert back to JSON for web visualization

This script can be run in stages or all at once.
"""

import subprocess
import sys
from pathlib import Path
import argparse

class PipelineRunner:
    """Orchestrates the complete pipeline"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.scripts = {
            'convert': self.base_dir / 'json_to_parquet.py',
            'embed': self.base_dir / 'embed_papers_parquet.py',
            'umap': self.base_dir / 'umap_reduce_parquet.py',
            'to_json': self.base_dir / 'parquet_to_json.py'
        }
        
        self.files = {
            'input_json': self.base_dir / 'neurips_2025_papers_full.json',
            'parquet': self.base_dir / 'neurips_2025_papers_full.parquet',
            'embedded': self.base_dir / 'neurips_2025_papers_with_embeddings.parquet',
            'umap': self.base_dir / 'neurips_2025_papers_with_umap.parquet',
            'final_json': self.base_dir / 'neurips_2025_papers_final.json'
        }
    
    def check_file_exists(self, file_key):
        """Check if a file exists"""
        file_path = self.files[file_key]
        return file_path.exists()
    
    def run_step(self, script_name, args=None):
        """Run a pipeline step"""
        script_path = self.scripts[script_name]
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        try:
            # Run from the script's directory to ensure relative paths work
            result = subprocess.run(cmd, check=True, cwd=str(self.base_dir))
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Step failed with error code {e.returncode}")
            return False
    
    def step_convert(self, force=False):
        """Step 1: Convert JSON to Parquet"""
        print("\n" + "="*70)
        print("STEP 1: Convert JSON to Parquet")
        print("="*70)
        
        if not force and self.check_file_exists('parquet'):
            print(f"✅ Parquet file already exists: {self.files['parquet']}")
            print("   Use --force to regenerate")
            return True
        
        if not self.check_file_exists('input_json'):
            print(f"❌ Input file not found: {self.files['input_json']}")
            return False
        
        return self.run_step('convert')
    
    def step_embed(self, force=False):
        """Step 2: Generate embeddings"""
        print("\n" + "="*70)
        print("STEP 2: Generate Embeddings")
        print("="*70)
        
        if not self.check_file_exists('parquet'):
            print(f"❌ Input not found: {self.files['parquet']}")
            print("   Run step 1 first (--step convert)")
            return False
        
        if not force and self.check_file_exists('embedded'):
            print(f"✅ Embeddings file already exists: {self.files['embedded']}")
            print("   Use --force to regenerate")
            return True
        
        return self.run_step('embed')
    
    def step_umap(self, force=False):
        """Step 3: Apply UMAP"""
        print("\n" + "="*70)
        print("STEP 3: Apply UMAP Dimensionality Reduction")
        print("="*70)
        
        if not self.check_file_exists('embedded'):
            print(f"❌ Input not found: {self.files['embedded']}")
            print("   Run step 2 first (--step embed)")
            return False
        
        if not force and self.check_file_exists('umap'):
            print(f"✅ UMAP file already exists: {self.files['umap']}")
            print("   Use --force to regenerate")
            return True
        
        return self.run_step('umap')
    
    def step_to_json(self, no_embeddings=False):
        """Step 4: Convert to JSON"""
        print("\n" + "="*70)
        print("STEP 4: Convert to JSON for Visualization")
        print("="*70)
        
        if not self.check_file_exists('umap'):
            print(f"❌ Input not found: {self.files['umap']}")
            print("   Run step 3 first (--step umap)")
            return False
        
        args = [self.files['umap'], '-o', self.files['final_json']]
        if no_embeddings:
            args.append('--no-embeddings')
        
        return self.run_step('to_json', args)
    
    def run_all(self, force=False, to_json=False, no_embeddings=False):
        """Run the complete pipeline"""
        print("\n" + "🚀 "*35)
        print("RUNNING COMPLETE PIPELINE")
        print("🚀 "*35)
        
        steps = [
            ('convert', lambda: self.step_convert(force)),
            ('embed', lambda: self.step_embed(force)),
            ('umap', lambda: self.step_umap(force))
        ]
        
        if to_json:
            steps.append(('to_json', lambda: self.step_to_json(no_embeddings)))
        
        for step_name, step_func in steps:
            print(f"\n▶️  Running: {step_name}")
            success = step_func()
            
            if not success:
                print(f"\n❌ Pipeline failed at step: {step_name}")
                return False
        
        print("\n" + "🎉 "*35)
        print("PIPELINE COMPLETE!")
        print("🎉 "*35)
        
        # Show final output location
        final_file = self.files['final_json'] if to_json else self.files['umap']
        print(f"\n✅ Final output: {final_file}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='NeurIPS 2025 Paper Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py --all
  
  # Run specific step
  python pipeline.py --step convert
  python pipeline.py --step embed
  python pipeline.py --step umap
  
  # Run all and convert to JSON at the end
  python pipeline.py --all --to-json
  
  # Force regeneration of existing files
  python pipeline.py --all --force
  
  # Convert to JSON without embeddings (smaller file)
  python pipeline.py --step to_json --no-embeddings

Pipeline Steps:
  1. convert - Convert JSON to Parquet
  2. embed   - Generate embeddings using OpenAI API
  3. umap    - Apply UMAP dimensionality reduction
  4. to_json - Convert final Parquet to JSON (optional)
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--step', choices=['convert', 'embed', 'umap', 'to_json'],
                       help='Run a specific step')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of existing files')
    parser.add_argument('--to-json', action='store_true',
                       help='Convert final output to JSON (use with --all)')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Exclude embeddings when converting to JSON')
    
    args = parser.parse_args()
    
    if not args.all and not args.step:
        parser.print_help()
        sys.exit(1)
    
    runner = PipelineRunner()
    
    if args.all:
        success = runner.run_all(args.force, args.to_json, args.no_embeddings)
        sys.exit(0 if success else 1)
    
    elif args.step:
        step_methods = {
            'convert': runner.step_convert,
            'embed': runner.step_embed,
            'umap': runner.step_umap,
            'to_json': lambda: runner.step_to_json(args.no_embeddings)
        }
        
        success = step_methods[args.step](args.force if args.step != 'to_json' else False)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()