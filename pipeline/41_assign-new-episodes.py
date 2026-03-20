#!/usr/bin/env python3
"""
Detect new transcripts not in clusters.json and assign them to the nearest
existing cluster using TF-IDF cosine similarity.

Run: uv run --with scikit-learn --with numpy python3 pipeline/41_assign-new-episodes.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
CLUSTERS_FILE = ROOT / "output" / "clusters.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "new_episode_assignments.json"
BOILERPLATE_LINES = 15


def load_clusters():
    with open(CLUSTERS_FILE) as f:
        return json.load(f)


def known_filenames(clusters):
    names = set()
    for cluster_data in clusters.values():
        for t in cluster_data["transcripts"]:
            names.add(t["filename"])
    return names


def load_transcript(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if len(lines) > BOILERPLATE_LINES:
        cleaned = "\n".join(lines[BOILERPLATE_LINES:])
    else:
        # Single-line or short transcripts: skip boilerplate by word count instead
        words = text.split()
        cleaned = " ".join(words[min(100, len(words)):]) if len(words) > 100 else text
    numeric_id = re.match(r"(\d+)", path.name)
    return {
        "filename": path.name,
        "numeric_id": int(numeric_id.group(1)) if numeric_id else 0,
        "text": cleaned,
        "word_count": len(cleaned.split()),
    }


def main():
    parser = argparse.ArgumentParser(description="Assign new episodes to clusters")
    parser.add_argument(
        "--update-clusters",
        action="store_true",
        help="Append new episodes to clusters.json",
    )
    args = parser.parse_args()

    clusters = load_clusters()
    known = known_filenames(clusters)

    # Find new transcripts
    all_paths = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    new_paths = [p for p in all_paths if p.name not in known]

    if not new_paths:
        print("No new episodes found.")
        sys.exit(0)

    print(f"Found {len(new_paths)} new episode(s):")
    for p in new_paths:
        print(f"  {p.name}")

    # Load all transcripts (existing + new)
    all_transcripts = [load_transcript(p) for p in all_paths]
    existing_transcripts = [t for t in all_transcripts if t["filename"] in known]
    new_transcripts = [t for t in all_transcripts if t["filename"] not in known]

    # Fit TF-IDF on all transcripts (same params as phase1_cluster.py)
    print("\nFitting TF-IDF...")
    all_texts = [t["text"] for t in all_transcripts]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_df=0.85,
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Build index mapping: filename -> row index in tfidf_matrix
    filename_to_idx = {t["filename"]: i for i, t in enumerate(all_transcripts)}

    # Compute cluster centroids (mean TF-IDF of member episodes)
    # Skip noise cluster for assignment purposes
    cluster_ids = sorted(
        [k for k in clusters if k != "noise"],
        key=lambda k: int(k.split("_")[1]),
    )

    centroids = {}
    for cid in cluster_ids:
        member_indices = [
            filename_to_idx[t["filename"]]
            for t in clusters[cid]["transcripts"]
            if t["filename"] in filename_to_idx
        ]
        if member_indices:
            centroid = tfidf_matrix[member_indices].mean(axis=0).A1
            centroids[cid] = centroid

    centroid_keys = list(centroids.keys())
    centroid_matrix = np.array([centroids[k] for k in centroid_keys])

    # Assign each new episode to nearest centroid
    assignments = []
    for t in new_transcripts:
        idx = filename_to_idx[t["filename"]]
        vec = tfidf_matrix[idx].toarray()
        sims = cosine_similarity(vec, centroid_matrix)[0]
        best_idx = sims.argmax()
        best_cluster = centroid_keys[best_idx]
        best_score = float(sims[best_idx])

        assignment = {
            "filename": t["filename"],
            "numeric_id": t["numeric_id"],
            "word_count": t["word_count"],
            "assigned_cluster": best_cluster,
            "cluster_id": int(best_cluster.split("_")[1]),
            "similarity_score": round(best_score, 4),
        }

        if best_score < 0.1:
            assignment["warning"] = "weak match — consider manual review or noise"
            print(f"  WARNING: {t['filename']} → {best_cluster} (score={best_score:.4f}) — WEAK MATCH")
        else:
            print(f"  {t['filename']} → {best_cluster} (score={best_score:.4f})")

        # Show top 3 clusters for context
        top3_indices = sims.argsort()[-3:][::-1]
        assignment["top_3"] = [
            {"cluster": centroid_keys[i], "score": round(float(sims[i]), 4)}
            for i in top3_indices
        ]

        assignments.append(assignment)

    # Write assignments
    with open(OUTPUT_FILE, "w") as f:
        json.dump(assignments, f, indent=2)
    print(f"\nAssignments written to {OUTPUT_FILE}")

    # Optionally update clusters.json
    if args.update_clusters:
        for a in assignments:
            cid = a["assigned_cluster"]
            clusters[cid]["transcripts"].append({
                "filename": a["filename"],
                "numeric_id": a["numeric_id"],
                "word_count": a["word_count"],
                "series": _extract_series(a["filename"]),
            })
            clusters[cid]["count"] += 1
        clusters[cid]["transcripts"].sort(key=lambda x: x["numeric_id"])

        with open(CLUSTERS_FILE, "w") as f:
            json.dump(clusters, f, indent=2)
        print(f"Updated {CLUSTERS_FILE}")


def _extract_series(filename):
    if "ten-minute-tips" in filename:
        return "Ten Minute Tips"
    elif "watts-doc" in filename:
        return "Watts Doc"
    elif "perspectives" in filename:
        return "Perspectives"
    return "Other"


if __name__ == "__main__":
    main()
