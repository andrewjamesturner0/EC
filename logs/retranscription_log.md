# Re-transcription Log

## Overview

79 episodes downloaded and re-transcribed to fix Whisper hallucination loops identified in `corrupted_transcripts.md` (40 severe, 27 moderate, 11 minor).

---

## Setup

- **Machine:** Linux, NVIDIA H200 (143GB VRAM), CUDA 12.9, 24 CPU cores, 235GB RAM
- **Model:** whisper.cpp large-v3 (`~/whisper.cpp/build/bin/whisper-cli`)
- **Build:** `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`
- **Bugs fixed in `download_corrupted.sh`:** Five `((var++))` calls under `set -euo pipefail` caused script to exit on first match. Changed to `var=$(( var + 1 ))`.
- **Changes to `transcribe.sh`:**
  - Default parallel jobs: 4 → 8 (H200 fits ~47 large-v3 instances)
  - Minimum CPU threads per worker: 1 → 4 (GPU does inference; CPU handles preprocessing)
  - Bug fix: `-fa` (flash attention) was missing from the temp=0.2 retry invocation
  - `--retranscribe` mode now skips temp=0 pass entirely and goes straight to temp=0.2, since all target files are already known-corrupted (saves ~half the GPU time)

---

## Pass 1: temp=0.2

- **Command:** `./transcribe.sh episodes/ transcripts/ 8 --retranscribe`
- **Started:** 2026-03-11 13:17 UTC
- **ETA:** 2026-03-11 ~19:40 UTC (~6.5 hours, 122.3 hours of audio at ~26x effective throughput)
- **Completed:** 2026-03-11 19:00 UTC (5h 42m)

### Final results (79/79 complete)

Full `check_corruption.py` scan of all 191 transcripts:

| Group | Before Pass 1 | After Pass 1 |
|-------|--------------|--------------|
| SEVERE (>50% repeated) | 40 | 42 |
| MODERATE | 27 | 15 |
| MINOR | 11 | 5 |
| **Total corrupted** | **78** | **62** |
| Clean | 113 | 129 |

Net improvement: 16 episodes fixed outright. However the severe count slightly increased — some moderate cases that were partially recoverable at temp=0 became worse at temp=0.2 (randomness produced new loops rather than breaking old ones).

**Key finding:** Temperature 0.2 alone is insufficient for severely corrupted episodes. The hallucination loop is self-reinforcing — whisper feeds its own repeated output back as context, perpetuating the loop regardless of temperature. The `-mc 0` flag (max-context 0, disabling previous-text conditioning) is needed to break this.

### Episodes fixed by Pass 1 (sample)
- `1419554557-watts-doc-41-does-overtraining`: 89% → 1.4% ✓
- `1843418724-perspectives-34-quantifying-training-volume`: 75% → 0.3% ✓
- `1960584159-ten-minute-tips-45-simplest-training-plan`: 66% → 0.7% ✓
- `1419554557` (overtraining, 67K word corruption): now 21K clean words ✓

### Still severely corrupted after Pass 1 (42 episodes, selected worst)
| Episode | Repeated% | Max run |
|---------|-----------|---------|
| `1370242147-watts-doc-40-endurance-adaptation` | 99% | 3620 |
| `2168476551-ten-minute-tips-63-best-worst-ftp-test` | 99% | 2826 |
| `1949321231-watts-doc-51-performance-phenotype` | 98% | 1664 |
| `1216529890-perspectives-8-why-hard-to-rest` | 98% | 3409 |
| `1233728686-watts-doc-37-fast-twitch-fibers` | 97% | 4862 |
| `1874275173-ten-minute-tips-39-individualizing-training` | 97% | 5613 |
| `1731147645-ten-minute-tips-34-strength-training-mistakes` | 96% | 6138 |
| `1608430374-watts-doc-45-ampk` | 94% | 6178 |
| `1681202379-watts-doc-47-redox-role` | 91% | 5471 |

---

## Pass 2: temp=0.4, -mc 0, --entropy-thold 1.8

- **Target:** 62 still-corrupted episodes (42 severe + 15 moderate + 5 minor from Pass 1, plus 4 re-run from benchmark test)
- **Command:** `./transcribe.sh episodes/ transcripts/ 8 --retranscribe`
- **Started:** 2026-03-11 19:18 UTC
- **Completed:** 2026-03-11 21:54 UTC (2h 36m)

### Benchmark test (4 most severe episodes)

Before running the full pass, tested on the 4 worst episodes (98-99% corrupted in Pass 1):

| Episode | Pass 1 | Pass 2 |
|---------|--------|--------|
| `watts-doc-40-endurance-adaptation` | 99%, run 3620 | **0.2%, run 2** ✓ |
| `ten-minute-tips-63-best-worst-ftp-test` | 99%, run 2826 | **0.0%, run 1** ✓ |
| `watts-doc-51-performance-phenotype` | 98%, run 1664 | **0.3%, run 2** ✓ |
| `perspectives-8-why-hard-to-rest` | 98%, run 3409 | **0.3%, run 2** ✓ |

### Final results (62/62 complete)

`check_corruption.py` scan of all 191 transcripts: **0 corrupted.** Every episode clean.

| Pass | Corrupted | Severe | Duration |
|------|-----------|--------|----------|
| Original | 78/191 | 40 | — |
| After Pass 1 (temp=0.2) | 62/191 | 42 | 5h 42m |
| After Pass 2 (temp=0.4, -mc 0) | **0/191** | **0** | 2h 36m |

---

## Lessons learned

### Root cause of Whisper hallucination loops
Whisper conditions each segment on the previous segment's output (previous-text context). When it produces a repeated line, that line is fed back as context, making the next segment very likely to repeat too — a self-reinforcing feedback loop. This is why temperature alone doesn't fix severe corruption: the model is being given strong evidence that repetition is correct.

**The fix: `-mc 0`** (max-context 0) disables previous-text conditioning entirely. Each segment is decoded independently. Combined with higher temperature (0.4) and lower entropy threshold (1.8), this breaks even the most severe loops (99% → <1%).

### What doesn't work
- **Temperature 0.2 alone:** Helps moderate corruption, ineffective for severe (97-99%). Improved 16 episodes, left 42 severely corrupted.
- **Smaller models:** Not the answer — corruption is a decoding artifact, not a capacity problem. Smaller models produce worse transcripts without reliably reducing loops.
- **Re-running at same settings:** Loops are deterministic at low temperature; same audio + same settings = same loop.

### Effective flags for known-corrupted re-transcription
```
-mc 0 --temperature 0.4 --entropy-thold 1.8 -fa
```

### GPU setup (H200 / CUDA)
- Build with `-DGGML_CUDA=ON` — flash attention (`-fa`) is critical for Hopper (compute 9.0+)
- `-fa` must be on **all** whisper invocations, including retries (bug: was missing from retry)
- Minimum 4 CPU threads per worker when GPU-accelerated (CPU handles preprocessing only)
- 8 parallel jobs optimal for H200 (143GB VRAM, large-v3 ≈ 4.5GB each)
- Effective throughput: ~28x real-time (8 workers × 3.5x per worker)

### Bash gotcha
`((var++))` under `set -euo pipefail` exits with code 1 when `var` is 0 (arithmetic false). Use `var=$(( var + 1 ))` instead. This bug caused `download_corrupted.sh` to silently exit after parsing, downloading nothing.

---

## To do after Pass 2
- [x] Run `python3 check_corruption.py` on full transcript set — 0 corrupted ✓
- [ ] Update `corrupted_transcripts.md` with new counts
- [ ] Re-run `phase1_cluster.py` to regenerate clusters with clean transcripts
- [ ] Remap existing extractions (`output/extractions/`) to new cluster assignments
- [ ] Regenerate `batch_assignments.md`
