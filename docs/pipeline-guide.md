# Pipeline Operator Guide

## Status
- **Current phase:** Pass 3 COMPLETE — Extended grounding verified, all corrections applied
- **Last action:** Extended grounding verification (60 claims, ~10% coverage): 42 supported (70%), 13 partial (22%), 5 errors (8%). All errors corrected in synthesis files and final_output.md.
- **Next action:** Final human review of `output/final_output.md`.
- **Models used:** Sonnet for extraction (background agents in parallel), Opus for synthesis and finalize.

---

## Phase 0: Re-transcription (corruption fix)

### Context
An automated scan found 78/191 Whisper transcripts have repeated-line corruption (hallucination loops). 40 are severe (>50% repeated lines). This distorts TF-IDF clustering — 37% of transcripts have corrupted vectors. Full details in `logs/corrupted_transcripts.md`.

### Prerequisites
- **whisper.cpp** built at `$HOME/whisper.cpp/build/bin/whisper-cli` with model `$HOME/whisper.cpp/models/ggml-large-v3.bin` (large-v3). Use `pipeline/00_setup-whisper.sh` to build with GPU support: `bash pipeline/00_setup-whisper.sh`. This builds with `-DGGML_CUDA=ON` and downloads the model. Requires CUDA toolkit (nvcc) — checks `/usr/local/cuda` automatically.
- **ffmpeg** (installed by `pipeline/00_setup-whisper.sh` if missing)
- **curl** (for RSS feed download)
- **uv** package manager (for clustering step): install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Steps

1. **Download audio** for corrupted episodes only:
   ```bash
   chmod +x pipeline/11_download-corrupted.sh
   ./pipeline/11_download-corrupted.sh
   ```
   This parses `logs/corrupted_transcripts.md`, fetches the RSS feed from `data/rss-url.txt`, matches numeric IDs to audio URLs, and downloads to `data/episodes/`. Check the summary output — some episodes may not match if they've been removed from the feed.

2. **Re-transcribe** the corrupted episodes:
   ```bash
   chmod +x pipeline/20_transcribe.sh
   ./pipeline/20_transcribe.sh data/episodes/ data/transcripts/ 8 --retranscribe
   ```
   - `--retranscribe` skips the temp=0 pass and goes straight to the effective settings (known-corrupted files don't need temp=0 to discover they're corrupted)
   - Default 8 parallel jobs — tune to GPU VRAM (large-v3 ≈ 4.5GB per process; H200 143GB fits 16+)
   - **Effective whisper flags for corruption:** `-mc 0 --temperature 0.4 --entropy-thold 1.8 -fa`
     - `-mc 0`: disables previous-text conditioning — the root cause of hallucination loops is whisper feeding its own corrupted output back as context; this breaks that feedback
     - `--temperature 0.4`: enough randomness to escape deep attractor states
     - `--entropy-thold 1.8`: more aggressive internal per-segment fallback (default 2.4)
     - `-fa`: flash attention — critical for Hopper GPUs (compute 9.0+), must be on all invocations
   - Temperature alone (0.2) is insufficient for severe corruption — `-mc 0` is the key flag
   - A timestamped log file (`transcribe_*.log`) records all results
   - **Note:** `pipeline/20_transcribe.sh` expects whisper.cpp at `$HOME/whisper.cpp/` — edit `WHISPER_DIR`, `WHISPER_BIN`, `MODEL` at the top of the script if your paths differ

3. **Verify improvement** — re-run the corruption scan:
   ```bash
   python3 pipeline/21_check-corruption.py
   ```
   This scans all transcripts in `data/transcripts/` for repeated-line patterns and prints SEVERE/MODERATE/MINOR results. Update `logs/corrupted_transcripts.md` with new results. Goal: severe corruption count drops from 40 to near zero.

4. **Re-cluster** (back on analysis machine, or here if `uv` is available):
   ```bash
   uv run --with scikit-learn --with umap-learn --with matplotlib --with numpy python3 pipeline/30_cluster.py
   ```
   This overwrites `output/clusters.json`, `output/cluster_viz.png`, `output/cluster_similarity.png`. Parameters are hardcoded in the script (min_cluster_size=5, min_samples=2, cluster_selection_method="leaf", n_neighbors=15).

5. **Remap existing extractions** — completed extractions exist in `output/extractions/` for clusters 0 (Strength), 1 (Nutrition), 2 (Technique). Each file contains per-episode blocks headed `## Episode: {filename}`. A script should split these by episode, look up each episode's new cluster, and reassemble into new extraction files. Episodes that were severely corrupted and now have clean transcripts need fresh extraction.

6. **Regenerate batch assignments** — `docs/batch_assignments.md` and the batch lists in this file need updating to reflect new cluster memberships and word counts.

### Files involved
| File | Role |
|------|------|
| `data/rss-url.txt` | SoundCloud RSS feed URL |
| `logs/corrupted_transcripts.md` | Full corruption log with severity, usable words, repeated line text |
| `pipeline/11_download-corrupted.sh` | Downloads audio for corrupted episodes only |
| `pipeline/20_transcribe.sh` | Whisper transcription with corruption detection + temp fallback |
| `data/transcripts/` | Output directory for .txt transcripts (191 files) |
| `data/episodes/` | Audio files (created by download script) |
| `pipeline/21_check-corruption.py` | Scans all transcripts for repeated-line corruption, prints summary |
| `pipeline/30_cluster.py` | Clustering pipeline |
| `output/clusters.json` | Current cluster assignments (will be overwritten) |
| `output/extractions/` | All 19 extraction files (191 episodes total) |

---

---

## Phase 1: Clustering

- [x] Install dependencies
- [x] Run clustering: `python pipeline/30_cluster.py`
- [x] Review `output/clusters.json` — 18 clusters + 33 noise points
- [x] Review `output/cluster_viz.png` and `output/cluster_similarity.png`
- [x] Phase 1 complete
- Params: min_cluster_size=5, min_samples=2, cluster_selection_method="leaf", n_neighbors=15

---

## Pass 1: Extraction

> **How to run each batch:** Ask Claude to read the extraction prompt (`prompts/extraction.md`) and the listed transcript(s), then save output to the specified extraction file (create on first batch, append on subsequent batches).
>
> **Batching rules applied:** <40K words per batch. Full filenames in `docs/batch_assignments.md`.
>
> **27 episodes already extracted** from previous run (marked ✅). These have been remapped to new clusters.

### Cluster 0: "Strength Training" (14 transcripts) ✅ COMPLETE → `output/extractions/strength_training.md`
### Cluster 1: "Research & Methodology" (11 transcripts) ✅ COMPLETE → `output/extractions/research_methodology.md`
### Cluster 2: "VO2max & Oxygen Uptake" (9 transcripts) ✅ COMPLETE → `output/extractions/vo2max_oxygen_uptake.md`
### Cluster 3: "Critical Power & Lactate" (5 transcripts) ✅ COMPLETE → `output/extractions/critical_power_lactate.md`
### Cluster 4: "FTP & Testing" (10 transcripts) ✅ COMPLETE → `output/extractions/ftp_testing.md`
### Cluster 5: "Interval Workouts" (5 transcripts) ✅ COMPLETE → `output/extractions/interval_workouts.md`
### Cluster 6: "Nutrition & Fueling" (9 transcripts) ✅ COMPLETE → `output/extractions/nutrition_fueling.md`
### Cluster 7: "Aerobic/Anaerobic Systems" (6 transcripts) ✅ COMPLETE → `output/extractions/aerobic_anaerobic_systems.md`
### Cluster 8: "Psychology & Coaching" (7 transcripts) ✅ COMPLETE → `output/extractions/psychology_coaching.md`
### Cluster 9: "Biochemistry Foundations" (7 transcripts) ✅ COMPLETE → `output/extractions/biochemistry_foundations.md`
### Cluster 10: "Endurance & Zone Training" (7 transcripts) ✅ COMPLETE → `output/extractions/endurance_zone_training.md`
### Cluster 11: "Mitochondrial Physiology" (7 transcripts) ✅ COMPLETE → `output/extractions/mitochondrial_physiology.md`
### Cluster 12: "Cell Biology & Signaling" (11 transcripts) ✅ COMPLETE → `output/extractions/cell_biology_signaling.md`
### Cluster 13: "Recovery & Workout Design" (17 transcripts) ✅ COMPLETE → `output/extractions/recovery_workout_design.md`
### Cluster 14: "Equipment & Technique" (10 transcripts) ✅ COMPLETE → `output/extractions/equipment_technique.md`
### Cluster 15: "Racing & Team Tactics" (12 transcripts) ✅ COMPLETE → `output/extractions/racing_team_tactics.md`
### Cluster 16: "Coaching & Goals A" (5 transcripts) ✅ COMPLETE → `output/extractions/coaching_goals_a.md`
### Cluster 17: "Coaching & Goals B" (6 transcripts) ✅ COMPLETE → `output/extractions/coaching_goals_b.md`
### Noise: "Uncategorized / Mixed Topics" (33 transcripts) ✅ COMPLETE → `output/extractions/uncategorized_mixed_topics.md`

---

## Pass 2: Synthesis

> **Per theme:** Read the synthesis prompt (`prompts/synthesis.md`), insert the full extraction file, and save output to `output/synthesis/{theme_name}.md`.

- [x] Strength Training → `output/synthesis/strength_training.md` (3200w)
- [x] Research & Methodology → `output/synthesis/research_methodology.md` (2688w)
- [x] VO2max & Oxygen Uptake → `output/synthesis/vo2max_oxygen_uptake.md` (2702w)
- [x] Critical Power & Lactate → `output/synthesis/critical_power_lactate.md` (2174w)
- [x] FTP & Testing → `output/synthesis/ftp_testing.md` (3076w)
- [x] Interval Workouts → `output/synthesis/interval_workouts.md` (2246w)
- [x] Nutrition & Fueling → `output/synthesis/nutrition_fueling.md` (3044w)
- [x] Aerobic/Anaerobic Systems → `output/synthesis/aerobic_anaerobic_systems.md` (2824w)
- [x] Psychology & Coaching → `output/synthesis/psychology_coaching.md` (2520w)
- [x] Biochemistry Foundations → `output/synthesis/biochemistry_foundations.md` (2369w)
- [x] Endurance & Zone Training → `output/synthesis/endurance_zone_training.md` (2534w)
- [x] Mitochondrial Physiology → `output/synthesis/mitochondrial_physiology.md` (2643w)
- [x] Cell Biology & Signaling → `output/synthesis/cell_biology_signaling.md` (2372w)
- [x] Recovery & Workout Design → `output/synthesis/recovery_workout_design.md` (2971w)
- [x] Equipment & Technique → `output/synthesis/equipment_technique.md` (2844w)
- [x] Racing & Team Tactics → `output/synthesis/racing_team_tactics.md` (2936w)
- [x] Coaching & Goals A → `output/synthesis/coaching_goals_a.md` (2801w)
- [x] Coaching & Goals B → `output/synthesis/coaching_goals_b.md` (2809w)
- [x] Uncategorized / Mixed Topics → `output/synthesis/uncategorized_mixed_topics.md` (3422w)

---

## Pass 3: Final Editing

- [x] Read all synthesis files from `output/synthesis/`
- [x] Review the thematic categorisation.
   - [x] Merged 19 themes → 13 themes:
      - coaching_goals_a + coaching_goals_b + uncategorized coaching → `coaching_practice_season_planning.md`
      - biochemistry + mitochondrial + cell_biology + uncategorized fiber type → `exercise_physiology_molecular_adaptation.md`
      - endurance_zone + aerobic_anaerobic + uncategorized easy rides/stimulus → `training_philosophy_intensity_distribution.md`
      - ftp_testing + critical_power + uncategorized FTP critique → `threshold_ftp_testing.md`
      - recovery_workout_design → `recovery_periodization.md` (all content preserved)
      - psychology_coaching → `psychology_performance.md` (renamed, coaching content moved)
      - nutrition_fueling, strength_training, racing_team_tactics updated with relevant uncategorized sections
      - vo2max_oxygen_uptake trimmed (endurance pacing + lactate testing moved to merged files)
      - equipment_technique, interval_workouts, research_methodology unchanged
- [x] Run editorial pass → `output/final_output.md` (44,393 words, 2,203 lines, 142 sub-sections)
- [x] Cross-reference verification: 14 cross-references in Cross-References section
- [x] Grounding verification: 60 claims sampled (~10% coverage) → 42 supported (70%), 13 partial (22%), 5 errors (8%). Report: `output/grounding_verification.md`
   - Fixed: Burke study conflation with Havemann study (40-70W attribution)
   - Fixed: Removed fabricated "TEAM" acronym from psychology section
   - Fixed: PGC-1alpha knockout deficit inverted (~30% → ~70%)
   - Fixed: NRF1 fabrication → replaced with actual markers (citrate synthase, cytochrome oxidase)
   - Fixed: 5-4-3-2-1 grounding exercise → replaced with Ryan's actual sensory contact exercise
   - Fixed: 30-40W unsupported case study → rewritten as general coaching observation
   - Fixed: Rory Porteous sub-100W attribution → changed to "Coaches prescribe"
- [ ] Final human review of `output/final_output.md`

---

## Scripts Reference
- `pipeline/10_download-episodes.sh` — download all episodes from RSS feed
- `pipeline/11_download-corrupted.sh` — download audio only for corrupted transcripts
- `pipeline/20_transcribe.sh` — whisper.cpp transcription with corruption detection, temperature fallback, per-run log file, resume robustness. Use `--retranscribe` to force re-transcription of existing files.
- `pipeline/21_check-corruption.py` — scan all transcripts for repeated-line corruption, prints SEVERE/MODERATE/MINOR summary
- `pipeline/30_cluster.py` — TF-IDF + UMAP + HDBSCAN clustering

## Prompts Reference
- Extraction prompt: `prompts/extraction.md`
- Synthesis prompt: `prompts/synthesis.md`
- Finalize prompt: `prompts/finalize.md`
- Full filenames: `docs/batch_assignments.md`

---

## Execution Summary

| Step | Action | Output | User Review |
|------|--------|--------|-------------|
| 1 | ~~Clustering~~ | `output/clusters.json`, `output/cluster_viz.png` | Done |
| 2–N | ~~Extract batches~~ (191/191 done) | `output/extractions/*.md` | Done |
| N+1 to N+19 | ~~Synthesize each theme (19x)~~ | `output/synthesis/*.md` | Review each |
| Final | Editorial pass + grounding check | `output/final_output.md` | Final review |

**Note:** Full unabbreviated filenames are in `docs/batch_assignments.md` for reference when running batches.
