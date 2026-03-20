#!/usr/bin/env python3
"""
Phase 1: Clustering pipeline for Empirical Cycling Podcast transcripts.
TF-IDF -> UMAP -> HDBSCAN -> clusters.json + visualizations.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sklearn.cluster import HDBSCAN

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
BOILERPLATE_LINES = 15  # lines to strip from start of each transcript


def load_transcripts():
    """Load all transcripts, strip boilerplate, return sorted by filename."""
    transcripts = []
    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        # Strip boilerplate intro lines
        cleaned = "\n".join(lines[BOILERPLATE_LINES:])
        # Extract numeric prefix for chronological ordering
        numeric_id = re.match(r"(\d+)", path.name)
        transcripts.append({
            "filename": path.name,
            "numeric_id": int(numeric_id.group(1)) if numeric_id else 0,
            "text": cleaned,
            "word_count": len(cleaned.split()),
        })
    print(f"Loaded {len(transcripts)} transcripts")
    return transcripts


def extract_series(filename):
    """Extract series name from filename."""
    if "ten-minute-tips" in filename:
        return "Ten Minute Tips"
    elif "watts-doc" in filename:
        return "Watts Doc"
    elif "perspectives" in filename:
        return "Perspectives"
    else:
        return "Other"


def cluster_transcripts(transcripts):
    """Run TF-IDF -> UMAP -> HDBSCAN pipeline."""
    texts = [t["text"] for t in transcripts]

    # TF-IDF
    print("Computing TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_df=0.85,
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # UMAP reduction to 15 dims for clustering
    print("Running UMAP (15D for clustering)...")
    reducer_15d = umap.UMAP(
        n_components=15,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.0,
        random_state=42,
    )
    embedding_15d = reducer_15d.fit_transform(tfidf_matrix.toarray())

    # HDBSCAN clustering (using sklearn's built-in implementation)
    print("Running HDBSCAN...")
    clusterer = HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(embedding_15d)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # UMAP reduction to 2D for visualization
    print("Running UMAP (2D for visualization)...")
    reducer_2d = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    embedding_2d = reducer_2d.fit_transform(tfidf_matrix.toarray())

    return tfidf_matrix, feature_names, labels, embedding_2d


def get_top_terms(tfidf_matrix, feature_names, labels, n_terms=15):
    """Get top TF-IDF terms per cluster."""
    cluster_terms = {}
    for cluster_id in sorted(set(labels)):
        label = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        mask = labels == cluster_id
        cluster_tfidf = tfidf_matrix[mask].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-n_terms:][::-1]
        cluster_terms[label] = [
            {"term": feature_names[i], "score": round(float(cluster_tfidf[i]), 4)}
            for i in top_indices
        ]
    return cluster_terms


def build_cluster_output(transcripts, labels, cluster_terms):
    """Build the clusters.json structure."""
    clusters = {}
    for cluster_id in sorted(set(labels)):
        label = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        mask = labels == cluster_id
        members = []
        for i in np.where(mask)[0]:
            t = transcripts[i]
            members.append({
                "filename": t["filename"],
                "numeric_id": t["numeric_id"],
                "word_count": t["word_count"],
                "series": extract_series(t["filename"]),
            })
        # Sort by numeric ID (chronological)
        members.sort(key=lambda x: x["numeric_id"])
        clusters[label] = {
            "count": int(mask.sum()),
            "top_terms": cluster_terms[label],
            "transcripts": members,
        }
    return clusters


def plot_clusters(embedding_2d, labels, transcripts, output_path):
    """2D scatter plot colored by cluster."""
    fig, ax = plt.subplots(figsize=(14, 10))
    unique_labels = sorted(set(labels))
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))

    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        if cluster_id == -1:
            color = "lightgray"
            label = f"Noise ({mask.sum()})"
            alpha = 0.4
            marker = "x"
        else:
            color = colors[idx % len(colors)]
            label = f"Cluster {cluster_id} ({mask.sum()})"
            alpha = 0.7
            marker = "o"
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color],
            label=label,
            alpha=alpha,
            marker=marker,
            s=40,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("Empirical Cycling Podcast — Transcript Clusters (UMAP 2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved cluster visualization to {output_path}")
    plt.close(fig)


def plot_similarity_heatmap(tfidf_matrix, labels, output_path):
    """Cosine similarity heatmap sorted by cluster."""
    # Sort by cluster label
    order = np.argsort(labels)
    sorted_matrix = tfidf_matrix[order]
    sim_matrix = cosine_similarity(sorted_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Transcript Cosine Similarity (sorted by cluster)")
    ax.set_xlabel("Transcript index (sorted)")
    ax.set_ylabel("Transcript index (sorted)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved similarity heatmap to {output_path}")
    plt.close(fig)


def main():
    transcripts = load_transcripts()
    tfidf_matrix, feature_names, labels, embedding_2d = cluster_transcripts(transcripts)
    cluster_terms = get_top_terms(tfidf_matrix, feature_names, labels)
    clusters = build_cluster_output(transcripts, labels, cluster_terms)

    # Save clusters.json
    clusters_path = OUTPUT_DIR / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"\nSaved {clusters_path}")

    # Print summary
    print("\n=== Cluster Summary ===")
    for name, data in clusters.items():
        terms = ", ".join(t["term"] for t in data["top_terms"][:5])
        print(f"  {name}: {data['count']} transcripts — {terms}")

    # Visualizations
    plot_clusters(embedding_2d, labels, transcripts, OUTPUT_DIR / "cluster_viz.png")
    plot_similarity_heatmap(tfidf_matrix, labels, OUTPUT_DIR / "cluster_similarity.png")

    print("\nDone! Review clusters.json and cluster_viz.png before proceeding.")


if __name__ == "__main__":
    main()
