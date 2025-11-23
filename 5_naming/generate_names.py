"""
Cluster Naming Script (KMeans-only version)
===========================================

This script generates names for clusters produced by your
KMeans pipeline (column: "cluster").

Method:
1. Extract abstracts in each cluster.
2. Generate TF-IDF keywords.
3. Generate semantic KeyBERT keywords.
4. Fuse both into GPT-4o-mini to produce a human-readable name.

Outputs:
- cluster_names.csv
- cluster_names.json
"""

import json
from pathlib import Path
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from openai import OpenAI


# ============================================================
# 0. Lazy global KeyBERT model
# ============================================================
_kw_model = None

def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model


# ============================================================
# 1. Extract cluster texts (abstracts + titles)
# ============================================================
def extract_cluster_texts(df, cluster_id):
    sub = df[df["cluster"] == cluster_id]

    # Fix: correct column name is "abstract"
    abstracts = sub["abstract"].dropna().astype(str).tolist()

    # Fix: use "title" column, not "paper"
    titles = sub["paper"].dropna().astype(str).tolist()

    return abstracts, titles


# ============================================================
# 2. TF-IDF keyword extraction
# ============================================================
def get_tfidf_keywords(texts, top_k=15):
    if not texts:
        return []

    doc = " ".join(texts)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform([doc])
    vocab = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    top_idx = scores.argsort()[-top_k:][::-1]
    return [vocab[i] for i in top_idx]


# ============================================================
# 3. KeyBERT keyword extraction
# ============================================================
def get_keybert_keywords(texts, top_k=12):
    if not texts:
        return []

    doc = " ".join(texts)
    kw_model = get_kw_model()

    try:
        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(1, 3),
            top_n=top_k,
        )
    except Exception:
        return []

    return [k for k, _ in keywords]


# ============================================================
# 4. GPT naming
# ============================================================
def generate_name_gpt(keywords, titles):
    client = OpenAI()

    # Trim keywords to something reasonable
    keywords = keywords[:25]

    prompt = f"""
You are generating a short, human-readable research topic name.

Representative keywords:
{keywords}

Example paper titles:
{titles}

Requirements:
- 5–8 words
- Must be specific and descriptive
- Avoid generic words like "machine learning" or "data"
- Capture the core scientific theme
- Output ONLY the name, with no commentary
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=25,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# ============================================================
# 5. Name a cluster
# ============================================================
def name_cluster(df, cluster_id):
    abstracts, titles = extract_cluster_texts(df, cluster_id)

    if not abstracts:
        return "Unknown Topic", []

    tfidf_kw = get_tfidf_keywords(abstracts, top_k=15)
    keybert_kw = get_keybert_keywords(abstracts, top_k=12)

    # merge & dedupe (preserve ordering)
    all_kw = list(dict.fromkeys(tfidf_kw + keybert_kw))

    # first 3 titles for context
    sample_titles = titles[:3] if titles else ["No titles available"]

    final_name = generate_name_gpt(all_kw, sample_titles)
    return final_name, all_kw


# ============================================================
# 6. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="4_cluster/data/neurips_2025_with_clusters.parquet"
    )
    parser.add_argument(
        "--output-dir",
        default="5_naming/data/"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = sorted(df["cluster"].dropna().unique().astype(int))
    rows = []

    print("\n=== Generating Cluster Names ===")
    for cid in cluster_ids:
        print(f"→ Naming cluster {cid}")

        name, keywords = name_cluster(df, cid)

        rows.append({
            "cluster": cid,
            "name": name,
            "keywords": ", ".join(keywords),
        })

    # CSV
    csv_path = out_dir / "cluster_names.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"💾 Saved {csv_path}")

    # JSON
    json_path = out_dir / "cluster_names.json"
    with open(json_path, "w") as f:
        json.dump({
            str(r["cluster"]): {
                "name": r["name"],
                "keywords": r["keywords"],
            } for r in rows
        }, f, indent=2)
    print(f"💾 Saved {json_path}")

    print("\n🎉 Done! All cluster names generated.\n")


if __name__ == "__main__":
    main()