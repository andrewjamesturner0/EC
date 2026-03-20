# Updating the EC Knowledge Base

How to incorporate new Empirical Cycling Podcast episodes without re-running the full pipeline.

## Prerequisites

- **whisper.cpp** — built at `$HOME/whisper.cpp/` with large-v3 model. See `pipeline/00_setup-whisper.sh`.
- **uv** — Python package runner (`~/.local/bin/uv`)
- **Claude Code** — for extraction and synthesis LLM passes

## Quick Reference

```bash
# 1. Download & transcribe new episode(s) — see "Transcription" below
# 2. Run the update orchestrator:
bash pipeline/42_update-episodes.sh
# 3. Follow printed extraction/synthesis instructions with Claude Code
# 4. Reassemble:
bash pipeline/40_assemble-final.sh
```

---

## Full Workflow

### 1. Download the new episode

Download the audio file into `data/episodes/`. You can use the RSS feed or download manually.

### 2. Transcribe

```bash
bash pipeline/20_transcribe.sh data/episodes/NEW_EPISODE.mp3 data/transcripts/ 1
```

This produces `data/transcripts/{numeric_id}-empiricalcyclingpodcast-{slug}.txt`. Check the transcript for corruption (repeated lines) — if present, re-run with `--retranscribe`.

### 3. Assign to cluster

```bash
uv run --with scikit-learn --with numpy python3 pipeline/41_assign-new-episodes.py
```

This compares the new transcript's TF-IDF vector against existing cluster centroids using cosine similarity. Output goes to `pipeline/new_episode_assignments.json`.

Review the assignments. If the similarity score is below 0.1, the match is weak — consider:
- Manually assigning to a different cluster
- Adding to noise/uncategorized

### 4. Override cluster assignment (optional)

Edit `pipeline/new_episode_assignments.json` directly — change `assigned_cluster` and `cluster_id` to the desired cluster. The extraction/synthesis instructions in `pipeline/42_update-episodes.sh` read from this file.

### 5. Extract

For each new episode, run extraction using the standard prompt:

```
Read prompts/extraction.md and data/transcripts/{FILENAME}.
Append the extraction output to {EXTRACTION_FILE} (from the assignment).
```

Use Sonnet for extraction. You can run multiple episodes in parallel as background agents.

### 6. Re-synthesise affected themes

Each cluster maps to an extraction file, and extraction files map to editorial themes (see `output/cluster_theme_map.json`). Re-synthesise only the affected themes:

```
Read prompts/synthesis.md and the full extraction file(s) for the theme.
Write synthesis output to {SYNTHESIS_FILE}.
```

Use Opus for synthesis. If a theme draws from multiple extraction files (e.g., "Exercise Physiology" draws from clusters 9, 11, 12), include ALL extraction files for that theme, not just the one that changed.

### 7. Update clusters.json

After extraction is done:

```bash
uv run --with scikit-learn --with numpy python3 pipeline/41_assign-new-episodes.py --update-clusters
```

### 8. Reassemble final output

```bash
bash pipeline/40_assemble-final.sh
```

This rebuilds `output/final_output.md` from the synthesis files and framing content.

### 9. Review

Check the reassembled output. Update framing content if needed:
- `output/framing/executive_summary.md` — if findings change materially
- `output/framing/cross_references.md` — if new cross-cutting connections emerge
- `output/framing/methodology_note.md` — update episode count

---

## Or use the orchestrator

`bash pipeline/42_update-episodes.sh` runs step 3 automatically and prints instructions for steps 5-8.

---

## Handling Uncategorized Episodes

Episodes assigned to noise in the original clustering don't have their own theme. Their extractions go to `output/extractions/uncategorized_mixed_topics.md`. Relevant content from uncategorized episodes was manually distributed across themes during the editorial pass.

For new episodes that match poorly to any cluster:
1. Extract to `uncategorized_mixed_topics.md`
2. Manually review the extraction and decide which theme(s) to incorporate it into
3. Re-synthesise those themes

## Updating Framing Content

The framing files in `output/framing/` are separate from the synthesis pipeline:
- **Executive summary** — rewrite if major new findings shift the top-level recommendations
- **Cross-references** — add new entries if a new episode creates connections between themes
- **Methodology note** — update the episode count

After editing framing files, re-run `bash pipeline/40_assemble-final.sh`.

## When to Consider a Full Re-cluster

The lightweight approach works well for small batches. Consider a full re-cluster when:
- **20+ new episodes** have been added incrementally
- **A new topic area** emerges that doesn't fit existing clusters
- **Cluster assignments** are consistently weak (scores < 0.1)

Full re-cluster: `uv run --with scikit-learn --with umap-learn --with matplotlib --with numpy python3 pipeline/30_cluster.py`

This overwrites `output/clusters.json` and requires remapping extractions and re-running the full pipeline.

## Troubleshooting

**Weak similarity scores (<0.1):** The episode may cover a genuinely new topic. Assign to the closest cluster or treat as uncategorized. If multiple new episodes cluster around a new topic, consider a full re-cluster.

**TF-IDF parameters:** The assignment script uses the same parameters as `pipeline/30_cluster.py` (5000 features, 1-3 ngrams, sublinear_tf, max_df=0.85, min_df=2). These are hardcoded in both files.

**Corrupted transcripts:** Run `python3 pipeline/21_check-corruption.py` on new transcripts before assignment. Corrupted transcripts will have distorted TF-IDF vectors and unreliable cluster assignments.
