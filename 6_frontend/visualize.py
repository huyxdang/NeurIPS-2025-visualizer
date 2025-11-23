"""
NeurIPS 2025 Paper Visualization using DataMapPlot
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datamapplot


def create_visualization(
    input_file: str = "7_frontend/neurips_2025_papers_final.parquet",
    output_file: str = "7_frontend/neurips_visualization.html",
    darkmode: bool = True
):
    """Create interactive DataMapPlot visualization"""
    
    print("="*70)
    print("📊 Creating Interactive NeurIPS 2025 Visualization")
    print("="*70)
    
    # Load data
    print(f"\n📖 Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"✅ Loaded {len(df):,} papers")
    
    # Coordinates
    coordinates = df[['umap_x', 'umap_y']].values
    
    # Label layers
    meta_cluster_labels = df['meta_cluster_name'].fillna('Unclustered').values
    cluster_labels = df['cluster_name'].fillna('Unclustered').values
    
    print(f"   Meta-clusters: {df['meta_cluster_name'].nunique()}")
    print(f"   Clusters: {df['cluster_name'].nunique()}")
    
    # Hover text
    print(f"\n📝 Preparing hover text...")
    hover_texts = []
    for idx, row in df.iterrows():
        hover = f"<b>{row['paper']}</b>"
        
        # Authors
        try:
            if isinstance(row['authors'], (list, np.ndarray)):
                authors = ', '.join([str(a) for a in row['authors'][:3]])
                if len(row['authors']) > 3:
                    authors += ' et al.'
                hover += f"<br><b>Authors:</b> {authors}"
        except:
            pass
        
        # Track, Award
        if pd.notna(row.get('track')):
            hover += f"<br><b>Track:</b> {row['track']}"
        if pd.notna(row.get('award')):
            hover += f"<br><b>Award:</b> {row['award']}"
        
        # Clusters
        if pd.notna(row.get('meta_cluster_name')):
            hover += f"<br><b>Meta-cluster:</b> {row['meta_cluster_name']}"
        if pd.notna(row.get('cluster_name')):
            hover += f"<br><b>Cluster:</b> {row['cluster_name']}"
        
        # Problem, Solution, ELI5
        if pd.notna(row.get('problem')):
            hover += f"<br><br><b>Problem:</b> {str(row['problem'])[:200]}"
        if pd.notna(row.get('solution')):
            hover += f"<br><br><b>Solution:</b> {str(row['solution'])[:200]}"
        if pd.notna(row.get('eli5')):
            hover += f"<br><br><b>ELI5:</b> {str(row['eli5'])[:200]}"
        
        # Link
        if pd.notna(row.get('link')):
            hover += f"<br><br><a href='{row['link']}' target='_blank'>View Paper</a>"
        
        hover_texts.append(hover)
    
    hover_text = np.array(hover_texts)
    
    # Create plot
    print(f"\n🎨 Creating visualization...")
    plot = datamapplot.create_interactive_plot(
        coordinates,
        meta_cluster_labels,
        cluster_labels,
        hover_text=hover_text,
        title="NeurIPS 2025 Papers",
        sub_title="View ~6,000 Accepted Papers by Topic Clusters",
        font_family="Inter, sans-serif",
        darkmode=darkmode,
        enable_search=True,
    )
    
    # Save
    print(f"\n💾 Saving to {output_file}...")
    plot.save(output_file)
    
    import os
    file_size = os.path.getsize(output_file) / 1024**2
    
    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)
    print(f"   File: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"\n🌐 Open: file://{Path(output_file).absolute()}")


if __name__ == "__main__":
    create_visualization()