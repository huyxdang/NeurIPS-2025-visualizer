"""
KMeans selection using combined metric:
    score(k) = silhouette(k) - 0.2 * davies_bouldin(k)

Much more reliable than stability and less biased than silhouette alone.

Workflow:
1. Run KMeans for k_min..k_max (step=k_step)
2. Compute silhouette and Davies–Bouldin scores
3. Select best k by score(k)
4. Fit final KMeans(k*)
5. Save parquet, score CSV, plots
"""

from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ======================================================================
# 1. LOAD DATA
# ======================================================================
def load_data(input_file: str):
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    print(f"\n📖 Loading {input_file} ...")
    df = pd.read_parquet(p)
    print(f"✅ Loaded {len(df)} rows")

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        raise ValueError("Missing UMAP coordinates ('umap_x', 'umap_y')")

    mask = df["umap_x"].notna() & df["umap_y"].notna()
    print(f"📊 Valid points: {mask.sum()}/{len(df)}")

    return df, mask


# ======================================================================
# 2. EVALUATE A SINGLE K
# ======================================================================
def evaluate_k(X: np.ndarray, k: int):
    print(f"\n⏳ Evaluating k={k} ...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)

    # compute metrics
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    score = sil - 0.2 * db

    print(f"   silhouette={sil:.4f}, davies={db:.4f}, score={score:.4f}")

    return sil, db, score, labels, kmeans


# ======================================================================
# 3. SEARCH OVER K
# ======================================================================
def search_k_range(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    step: int,
):
    results = []

    best_k = None
    best_score = -np.inf
    best_labels = None
    best_model = None

    for k in range(k_min, k_max + 1, step):
        sil, db, score, labels, model = evaluate_k(X, k)

        results.append({
            "k": k,
            "silhouette": sil,
            "davies": db,
            "score": score,
        })

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = model

    return best_k, best_score, best_labels, best_model, pd.DataFrame(results)


# ======================================================================
# 4. SAVE RESULTS
# ======================================================================
def save_results(df, valid_mask, labels, score_df, output_path):
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df["cluster"] = -1
    df.loc[valid_mask, "cluster"] = labels

    df.to_parquet(output_path, index=False)
    print(f"\n💾 Saved clustered data → {output_path}")

    csv_path = out_dir / "k_scores.csv"
    score_df.to_csv(csv_path, index=False)
    print(f"💾 Saved score table → {csv_path}")


# ======================================================================
# 5. PLOTS
# ======================================================================
def plot_score_curve(score_df, out_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(score_df["k"], score_df["score"], "-o", label="score(k)")
    plt.plot(score_df["k"], score_df["silhouette"], "--", label="silhouette")
    plt.plot(score_df["k"], score_df["davies"], "--", label="davies-bouldin")

    plt.xlabel("k")
    plt.ylabel("Metric value")
    plt.title("KMeans Metric Scores vs k")
    plt.grid(alpha=0.3)
    plt.legend()

    out = out_dir / "k_metric_scores.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved → {out}")


def plot_umap_clusters(df, out_dir):
    valid = df[df["cluster"] >= 0].copy()
    clusters = sorted(valid["cluster"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(clusters))))

    plt.figure(figsize=(12, 9))
    for idx, c in enumerate(clusters):
        sub = valid[valid["cluster"] == c]
        plt.scatter(
            sub["umap_x"], sub["umap_y"],
            s=10,
            alpha=0.7,
            c=[colors[idx % len(colors)]],
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Best-k UMAP Clusters")
    plt.grid(alpha=0.3)

    out = out_dir / "umap_bestk_clusters.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved → {out}")


# ======================================================================
# 6. MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="4_cluster/data/neurips_2025_papers_with_summaries.parquet")
    parser.add_argument("--output", default="4_cluster/data/neurips_2025_with_clusters.parquet")
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    print("\n=== KMeans Selection: silhouette - 0.2*db ===")

    df, mask = load_data(args.input)
    X = df.loc[mask, ["umap_x", "umap_y"]].values

    best_k, best_score, best_labels, best_model, score_df = search_k_range(
        X,
        k_min=args.k_min,
        k_max=args.k_max,
        step=args.step,
    )

    print(f"\n🏆 Best k = {best_k} (score={best_score:.4f})")

    out_path = Path(args.output)
    save_results(df, mask, best_labels, score_df, out_path)

    out_dir = out_path.parent
    plot_score_curve(score_df, out_dir)
    plot_umap_clusters(df, out_dir)

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
