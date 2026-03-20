# Lessons Learned: Empirical Cycling Podcast Thematic Analysis

## Project Overview

This project transformed ~180 podcast episodes (~3.1 million words) into a 44K-word structured training knowledge base using a pipeline of: RSS download → Whisper transcription → TF-IDF/UMAP/HDBSCAN clustering → LLM extraction → LLM synthesis → editorial restructuring → grounding verification. The full pipeline ran across three machines (local dev, GPU transcription server, Claude Code) over approximately one week.

---

## 1. Whisper Hallucination Loops

**The single biggest technical challenge of the project.**

### Problem

Whisper large-v3 produced severe hallucination loops in 41% of transcripts (78/191). The symptom: a single phrase or sentence repeated hundreds or thousands of times, filling the transcript with garbage. The worst cases had 97–99% corrupted lines — a file with 7,922 lines where 7,704 were the same sentence repeated.

### Root Cause

Whisper conditions each audio segment on the text of the previous segment (the "previous-text context" or `max-context` window). When it hallucinates a repeated line, that line becomes the context for the next segment, making repetition more likely. This creates a self-reinforcing feedback loop that, once entered, is nearly impossible to escape.

### What Didn't Work

- **Lower temperature (0.2):** Improved 16 episodes but left 42 severely corrupted. Low temperature makes the model *more* deterministic, which means once it enters a loop, it stays there even more reliably.
- **Re-running with identical settings:** Loops are deterministic at low temperature. Same audio + same settings = same loop.
- **Hoping smaller segments would help:** The loop propagates across segments via the context window.

### What Worked

The `-mc 0` flag (max-context 0) in whisper.cpp, which disables previous-text conditioning entirely. Each segment is decoded independently, breaking even 99%-corrupted loops.

**Effective re-transcription flags:**
```bash
-mc 0 --temperature 0.4 --entropy-thold 1.8 -fa
```

- `-mc 0`: Breaks the feedback loop (the critical fix)
- `--temperature 0.4`: Higher than default to escape attractor states
- `--entropy-thold 1.8`: More aggressive per-segment fallback (default 2.4)
- `-fa`: Flash attention for Hopper GPUs

**Result:** 0/191 corrupted transcripts after re-transcription.

### Lesson

**Always run corruption detection before using Whisper output.** A simple consecutive-repeated-line counter (3+ = flagged) catches virtually all hallucination loops. Build this into the transcription pipeline as a post-processing step, not an afterthought. The `-mc 0` flag should be the default for long-form audio (podcasts, lectures, interviews) unless cross-segment coherence is critical.

---

## 2. GPU Transcription Parallelism

### Problem

Transcribing 191 episodes serially on an H200 would take ~12 hours. The GPU was underutilised at 1 worker.

### Solution

Run 8 parallel whisper.cpp workers. Each large-v3 instance uses ~4.5GB VRAM; with 143GB available on the H200, 8 workers fit comfortably. CPU-side, each worker needs ~4 threads for audio preprocessing.

**Effective throughput:** ~28x real-time with 8 parallel workers.

### Gotchas

- **Flash attention (`-fa`) must be on ALL invocations.** The original `transcribe.sh` had `-fa` on the initial transcription call but not on the retry path. This caused retries to run 3-4x slower and sometimes fail on long episodes.
- **Bash arithmetic under `set -euo pipefail`:** `((var++))` returns exit code 1 when `var` is 0 (because 0 is "false" in arithmetic context). This silently killed the download script after parsing, downloading nothing. Fix: use `var=$(( var + 1 ))` instead.

### Lesson

**Test your retry/fallback paths as thoroughly as your happy path.** The retry invocation missing `-fa` was invisible during normal operation and only surfaced when corruption forced retries on long episodes.

---

## 3. Clustering: Corrupted Transcripts Poisoned TF-IDF

### Problem

The initial clustering run (before re-transcription) produced nonsensical clusters because ~37% of transcripts had corrupted TF-IDF vectors. A transcript where "and the next thing you know" is repeated 3,620 times has a very different term frequency profile than the actual episode content.

### Solution

Re-transcribe all corrupted episodes, then re-run clustering from scratch. The clean transcripts produced dramatically better clusters — 18 coherent themes plus 33 noise points, versus the original run which had many episodes misassigned.

### Parameters That Worked

```python
TF-IDF: max_features=5000, ngram_range=(1,3), max_df=0.85, min_df=2
UMAP: n_neighbors=15, n_components=15, metric='cosine'
HDBSCAN: min_cluster_size=5, min_samples=2, cluster_selection_method='leaf'
```

- `n_neighbors=15` was the key tuning parameter — lower values (5, 10) produced too many micro-clusters; higher values (25+) merged distinct topics.
- `cluster_selection_method='leaf'` gave finer-grained clusters than `'eom'` (the default), which was important for a corpus with many overlapping cycling-specific topics.
- Stripping the first 15 lines of each transcript removed boilerplate intros that otherwise dominated similarity scores.

### Lesson

**Garbage in, garbage out is especially brutal for unsupervised methods.** Supervised approaches might tolerate some corrupted inputs; clustering has no ground truth to correct against, so even a moderate corruption rate can silently ruin the entire structure.

---

## 4. Python Environment: No pip, No Python.h

### Problem

The transcription/clustering machine had no `pip` and no `Python.h` headers, so packages requiring C compilation (like the standalone `hdbscan` package) couldn't be installed.

### Solution

Use `uv` (installed at `~/.local/bin/uv`) with inline dependency resolution:
```bash
uv run --with scikit-learn --with umap-learn --with matplotlib --with numpy python3 phase1_cluster.py
```

scikit-learn's built-in HDBSCAN (`sklearn.cluster.HDBSCAN`, available since v1.3) replaced the standalone package, avoiding the C compilation requirement entirely.

### Lesson

**Prefer tools that work without environment setup.** `uv run --with` is excellent for one-off scripts on machines you don't control. And check whether your dependency has been absorbed into a larger package before reaching for the standalone version.

---

## 5. LLM Extraction: Agent Parallelism and Context Limits

### Problem

191 episodes across 18 clusters + noise. Sequential extraction would take many hours. But individual transcripts can be 15–80K words, and some batches exceeded Claude's context window.

### Solution

- **Batch by word count:** Keep each batch under ~40K words (leaving room for prompt + output).
- **Run batches as parallel background agents:** Up to 18 agents simultaneously, one per cluster.
- **Split large clusters:** The noise cluster (33 episodes) was split into 2 batches of 16.
- **Use Sonnet for extraction** (cost-efficient at $3/$15 per MTok) and **Opus for synthesis** (higher editorial judgment at $15/$75 per MTok).

### Gotchas

- **Episode count verification is non-trivial.** Different extraction agents used slightly different heading formats (`**Filename:**` vs `**File:**` vs `## Episode 1`). A simple `grep -c "^**Filename"` missed episodes with variant formats. The reliable count method: `grep -oP '\d{9,}' "$f" | sort -u | wc -l` (extract all 9+ digit numeric IDs, deduplicate).
- **Previously extracted episodes from old cluster assignments** caused mismatches. After re-clustering, 27 episodes had been extracted under the old cluster structure. These had to be systematically moved to their correct new cluster files using line-range extraction.

### Lesson

**Always verify counts with format-agnostic methods.** When multiple agents produce output, they will use slightly different formatting. Build verification around content identifiers (like episode IDs), not formatting conventions.

---

## 6. Synthesis Agent Output Variability

### Problem

19 synthesis agents (one per theme) produced output with inconsistent formatting: different heading styles, different section depths, some with Evolution of Views sections and some without.

### Solution

The prompt template (`prompts/synthesis.md`) was prescriptive about format, but agents still varied. This was acceptable because the editorial pass (Pass 3) normalised everything. Attempting to enforce perfect consistency at the synthesis stage would have required re-running agents, which wasn't worth the cost.

### Lesson

**Accept variability in intermediate outputs; enforce consistency at the final stage.** LLM agents given the same prompt will produce structurally different outputs. Design your pipeline so the final pass handles normalisation rather than trying to prevent variation upstream.

---

## 7. Final Output Assembly: Agent Write Limits

### Problem

The editorial pass agent was asked to read all 13 synthesis files (~43K words) and write a single `final_output.md`. It stalled for 39 minutes after reading all files and never produced output — the Write tool hit output token limits trying to emit a 43K-word file in one call.

### Solution

Split the task:
1. **Framing agent:** Write only the executive summary, table of contents, cross-references, and methodology note (~1K words) to `final_framing.md`.
2. **Bash concatenation:** Assemble the final document by inserting the 13 synthesis files between the header and footer sections, with heading levels adjusted programmatically.

```bash
# Insert numbered themes between header and footer
sed -n '1,/\[THEMES GO HERE\]/p' final_framing.md | head -n -1 > final_output.md
# ... concatenate each synthesis file with section numbering ...
sed -n '/\[THEMES GO HERE\]/,$ p' final_framing.md | tail -n +2 >> final_output.md
```

Heading levels were fixed with a Python one-liner that demoted `##` to `###` for sub-sections while preserving `##` for top-level theme headings.

### Lesson

**Don't ask an LLM to write a 40K-word document in one shot.** Split generation (framing/editorial content) from assembly (concatenation). The LLM adds value in writing summaries, cross-references, and editorial judgments — not in copying 43K words through its output buffer. Use the right tool for each sub-task: LLM for synthesis, shell for assembly.

---

## 8. Grounding Verification: LLM Error Patterns in Synthesis

### Problem

Even with source citations, the synthesis pipeline introduced factual errors. An extended verification of 60 claims (~10% of specific factual assertions) found 5 errors (8% error rate), falling into five distinct categories:

1. **Factual conflation** (2 cases): The model merges details from different studies or contexts into a single claim. The Burke race walker study results (stagnant 10K times) were conflated with the Havemann cycling study results (40–70W deficit) because both were discussed in the same episode. A "30-40 watts of FTP gain from reducing endurance intensity" was synthesised from a figure describing gains in a completely different context.

2. **Fabricated structure** (2 cases): The model creates authoritative-sounding terminology not in the source. A "TEAM framework (Treat Emotions As Messengers)" was attributed to a guest who never used that acronym. A "5-4-3-2-1 grounding exercise" was imported from general therapeutic knowledge and attributed to a specific podcast guest who described a different exercise entirely.

3. **Numerical inversion** (1 case): PGC-1alpha/beta knockout mice were described as having a "~30% performance deficit" when they actually performed at 30% of control levels — a ~70% deficit. The model confused "X% of control" with "X% deficit."

4. **Fabricated detail** (1 case): "10x more NRF1 and 2x mitochondrial density" was stated as a compensatory response in knockout mice. The transcript actually says "3-fold citrate synthase and 10-fold cytochrome oxidase." NRF1 is never mentioned — the model substituted a plausible molecular biology term.

5. **Attribution error** (1 case): A sub-100W recovery ride prescription was attributed to a specific coach (Rory Porteous) but could not be confirmed in his episode transcript.

### Solution

An initial 10-claim sample caught the first two error types (Burke/Havemann conflation, TEAM acronym). An extended 50-claim verification across all 13 themes caught the remaining three. All 5 errors were corrected in both the synthesis files and the final output.

### Aggregate Results

| Rating | Count | Percentage |
|--------|-------|------------|
| Fully supported | 42 | 70% |
| Partially supported | 13 | 22% |
| Errors found and corrected | 5 | 8% |

### Lesson

**LLMs hallucinate structure, not just facts.** The errors share a common pattern: the model makes the output *more coherent* than the source material — smoothing over distinctions between two studies discussed in sequence, packaging diffuse concepts into catchy acronyms, importing well-known domain knowledge and attributing it to a specific source. These errors read convincingly and survive casual review.

The 8% error rate on specific factual claims means a 44K-word document likely contains 20-30 undetected errors beyond those found. **Budget for grounding verification as a non-optional pipeline stage, not an afterthought.** Even a 10% sample dramatically improves output quality, and the error taxonomy it produces (which types of errors the model makes) is as valuable as the specific corrections.

---

## 9. Theme Restructuring: 19 → 13

### Problem

The initial 19 clusters (from HDBSCAN) produced synthesis files with extensive thematic overlap. Three molecular biology themes (biochemistry, mitochondrial, cell biology) covered the same AMPK/PGC-1alpha pathways. Two coaching themes duplicated content on auto-regulation and easy rides. The uncategorized cluster contained 10 distinct topics that belonged in other themes.

### Solution

A review agent read all 19 synthesis files and proposed: 4 merges, 1 split, full redistribution of uncategorized content. This reduced 19 themes to 13 with minimal content loss and significant deduplication.

**Merges:**
- 3 molecular biology themes → Exercise Physiology & Molecular Adaptation
- 2 coaching themes + uncategorized coaching → Coaching, Goal-Setting & Season Planning
- Endurance + Aerobic/Anaerobic + uncategorized sections → Training Philosophy & Intensity Distribution
- FTP Testing + Critical Power + uncategorized FTP critique → Threshold, FTP & Performance Testing

**Key decision:** Run merge agents in parallel (7 simultaneous agents), each responsible for one merge operation. This completed in ~5 minutes rather than the ~35 minutes sequential execution would have taken.

### Lesson

**Unsupervised clustering is a starting point, not an endpoint.** HDBSCAN produced reasonable initial groupings, but domain-aware editorial judgment was needed to merge related clusters and redistribute orphaned content. Plan for a human-in-the-loop (or LLM-in-the-loop) restructuring pass after initial clustering.

---

## 10. Context Window Management Across Sessions

### Problem

The full pipeline required multiple Claude Code sessions because individual sessions hit context limits. State had to be preserved across sessions, including: which extractions/syntheses were complete, which episodes belonged to which clusters, and what the current pipeline status was.

### Solution

- **`run_pipeline.md`:** Master operator guide with checkbox-style progress tracking, updated after each major step.
- **`batch_assignments.md`:** Complete episode-to-cluster mapping with full filenames (not abbreviated).
- **`MEMORY.md`:** Claude Code's persistent memory system, used to store cluster themes, pipeline status, and key decisions.
- **Session summaries:** When context compaction occurred, the system generated a detailed summary that the next session could pick up from.

### Lesson

**Design for session boundaries from the start.** Use filesystem-based state (checklists, assignment files, status logs) rather than relying on conversation context. Every piece of state that matters should be written to a file before the session might end.

---

## Cost Summary

| Phase | Model | Estimated Cost |
|-------|-------|---------------|
| Extraction (191 episodes) | Sonnet | ~$10 |
| Synthesis (19 themes) | Opus | ~$20 |
| Restructuring + Editorial | Opus + Sonnet | ~$8 |
| Grounding Verification (60 claims) | Opus | ~$15 |
| **Total API cost** | | **~$53** |

Transcription cost was zero (self-hosted whisper.cpp on GPU).

---

## Pipeline Architecture Summary

```
RSS Feed
  → download_podcast.sh
  → MP3 files (180 episodes)

MP3 files
  → transcribe.sh (whisper.cpp, 8 parallel workers, -fa -mc 0)
  → Plain text transcripts (191 files, ~3.1M words)
  → check_corruption.py (quality gate)

Transcripts
  → phase1_cluster.py (TF-IDF + UMAP + HDBSCAN)
  → clusters.json (18 clusters + 33 noise)

Clusters + Transcripts
  → Pass 1: Extraction (Sonnet, parallel agents, prompts/extraction.md)
  → output/extractions/*.md (191 episodes extracted)

Extractions
  → Pass 2: Synthesis (Opus, parallel agents, prompts/synthesis.md)
  → output/synthesis/*.md (19 → 13 themes after restructuring)

Synthesis files
  → Pass 3: Editorial (Opus framing + bash assembly)
  → output/final_output.md (44K words, 142 sub-sections)
  → Grounding verification (60 claims, ~10% coverage, 5 errors corrected)
```

---

## If I Were Doing This Again

1. **Run whisper with `-mc 0` from the start.** The re-transcription campaign added a full day to the timeline. The trade-off (slightly less coherent segment boundaries) is negligible for podcast transcription.

2. **Build corruption detection into the transcription script from day one.** The check was added retroactively; it should be a post-processing step that automatically triggers re-transcription.

3. **Use a two-stage clustering approach.** First pass: broad clusters (8-10). Second pass: sub-cluster within each broad group. This would have avoided the over-segmentation that required manual merging in Pass 3.

4. **Set explicit output format contracts for extraction agents.** The heading-format variability (`**Filename:**` vs `**File:**` vs `## Episode`) caused downstream verification headaches. A stricter prompt or post-processing normalisation step would have saved time.

5. **Never ask an LLM to write a document longer than ~10K words in one shot.** Plan for assembly-based approaches from the start for any output over that threshold.

6. **Budget for grounding verification at ≥10% coverage.** It's tempting to skip, but the initial 10-claim sample caught 1 error, and extending to 60 claims caught 4 more — including a numerical inversion and a fabricated molecular biology term that would have been significant factual mistakes. The error taxonomy (conflation, fabrication, inversion, attribution) is itself a useful output that informs future pipeline design.
