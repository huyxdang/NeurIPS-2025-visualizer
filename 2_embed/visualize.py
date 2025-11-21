"""
Visualize UMAP embeddings using Plotly
"""

import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import numpy as np

# Load papers with UMAP coordinates
input_file = Path(__file__).parent / "data" / "neurips_2025_papers_with_umap.json"

print(f"Loading data from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Loaded {len(papers)} papers")

# Filter papers that have UMAP coordinates
papers_with_umap = [p for p in papers if "umap_x" in p and "umap_y" in p]
print(f"Found {len(papers_with_umap)} papers with UMAP coordinates")

if not papers_with_umap:
    print("❌ No papers with UMAP coordinates found. Please run umap.py first.")
    exit(1)

# Prepare data for visualization
data = []
for paper in papers_with_umap:
    data.append({
        "x": paper["umap_x"],
        "y": paper["umap_y"],
        "title": paper.get("paper", "N/A"),
        "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), (list, np.ndarray)) else str(paper.get("authors", "N/A")),
        "track": paper.get("track", "Unknown"),
        "award": paper.get("award", "Unknown"),
        "abstract": paper.get("abstract", "N/A")[:200] + "..." if len(paper.get("abstract", "")) > 200 else paper.get("abstract", "N/A"),
        "link": paper.get("link", ""),
        "paper_id": paper.get("paper_id", "")
    })

df = pd.DataFrame(data)

# Create interactive scatter plot
print("\nCreating visualization...")

# Color by award type
fig_award = px.scatter(
    df,
    x="x",
    y="y",
    color="award",
    hover_data=["title", "authors", "track"],
    title="NeurIPS 2025 Papers - UMAP Visualization (Colored by Award Type)",
    labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2"},
    color_discrete_map={
        "Oral": "#FF6B6B",
        "Spotlight": "#4ECDC4",
        "Poster": "#95E1D3"
    }
)

# Update hover template
fig_award.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Award: %{marker.color}<br>" +
                  "Track: %{customdata[2]}<br>" +
                  "Authors: %{customdata[1]}<br>" +
                  "<extra></extra>",
    hovertext=df["title"]
)

fig_award.update_layout(
    width=1200,
    height=800,
    hovermode="closest"
)

# Color by track
fig_track = px.scatter(
    df,
    x="x",
    y="y",
    color="track",
    hover_data=["title", "authors", "award"],
    title="NeurIPS 2025 Papers - UMAP Visualization (Colored by Track)",
    labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2"},
    color_discrete_map={
        "Main": "#3498DB",
        "Datasets and Benchmarks": "#E74C3C"
    }
)

fig_track.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Track: %{marker.color}<br>" +
                  "Award: %{customdata[2]}<br>" +
                  "Authors: %{customdata[1]}<br>" +
                  "<extra></extra>",
    hovertext=df["title"]
)

fig_track.update_layout(
    width=1200,
    height=800,
    hovermode="closest"
)

# Create interactive plot with custom hover (showing abstract)
fig_detailed = go.Figure()

# Add traces for each award type
for award_type in df["award"].unique():
    df_award = df[df["award"] == award_type]
    fig_detailed.add_trace(go.Scatter(
        x=df_award["x"],
        y=df_award["y"],
        mode="markers",
        name=award_type,
        text=df_award["title"],
        customdata=df_award[["authors", "track", "abstract", "link"]].values,
        hovertemplate="<b>%{text}</b><br>" +
                      "Award: " + award_type + "<br>" +
                      "Track: %{customdata[1]}<br>" +
                      "Authors: %{customdata[0]}<br>" +
                      "Abstract: %{customdata[2]}<br>" +
                      "<a href='%{customdata[3]}'>OpenReview Link</a>" +
                      "<extra></extra>",
        marker=dict(
            size=5,
            opacity=0.7
        )
    ))

fig_detailed.update_layout(
    title="NeurIPS 2025 Papers - UMAP Visualization (Interactive)",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2",
    width=1400,
    height=900,
    hovermode="closest",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Save visualizations
output_dir = Path(__file__).parent
fig_award.write_html(str(output_dir / "umap_visualization_award.html"))
fig_track.write_html(str(output_dir / "umap_visualization_track.html"))
fig_detailed.write_html(str(output_dir / "umap_visualization_detailed.html"))

print(f"\n✅ Saved visualizations:")
print(f"  - {output_dir / 'umap_visualization_award.html'}")
print(f"  - {output_dir / 'umap_visualization_track.html'}")
print(f"  - {output_dir / 'umap_visualization_detailed.html'}")

# Show the detailed plot
print("\nOpening interactive visualization...")
fig_detailed.show()

