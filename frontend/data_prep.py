import pandas as pd
from pathlib import Path

# Load your final dataframe with UMAP coords and Clusters
input_file = "5_naming/data/neurips_papers_final.parquet"
if not Path(input_file).exists():
    # Try alternative path
    input_file = "fronend/data/neurips_papers_final.parquet"

df = pd.read_parquet(input_file)

# Select only what the frontend needs
frontend_df = df[[
    "paper",      
    "authors",
    "link",
    "track",
    "award",
    "umap_x", "umap_y",
    "problem",
    "solution",
    "eli5",
    "cluster_name",       
]].copy()

# Assign integer IDs to clusters for coloring
frontend_df['cluster_id'] = frontend_df['cluster_name'].astype('category').cat.codes

# Calculate cluster centers (mean x, y coordinates for each cluster)
cluster_centers = frontend_df.groupby('cluster_name')[['umap_x', 'umap_y']].mean().reset_index()
cluster_centers.columns = ['cluster_name', 'x', 'y']  # Rename for frontend

# Export cluster centers to JSON
cluster_centers.to_json("frontend/public/cluster_labels.json", orient="records")
print(f"✅ Created cluster_labels.json with {len(cluster_centers)} cluster centers")

# Export main data to JSON (must be in public folder for React to serve it)
frontend_df.to_json("frontend/public/neurips_data.json", orient="records")
print(f"✅ Created neurips_data.json with {len(frontend_df)} papers")
