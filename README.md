# EC — Empirical Cycling Podcast Thematic Analysis

Thematic knowledge base distilled from ~190 episodes of the Empirical Cycling Podcast, using automated transcription, NLP clustering, and LLM-driven extraction/synthesis.

## Directory Layout

```
data/           Transcripts (.txt) and audio episodes (.mp3)
pipeline/       All scripts, numbered by stage (00–50)
prompts/        LLM prompt templates (extraction, synthesis, finalize)
output/         Extractions, synthesis, framing, final_output.md, cluster data
site/           Browsable HTML version of the knowledge base
docs/           Pipeline guide, update guide, methods, spec, batch assignments
logs/           Transcription logs, corruption reports
```

## Quick Start

**Run the full pipeline** (one-time):

```bash
bash pipeline/00_setup-whisper.sh          # build whisper.cpp + download model
bash pipeline/10_download-episodes.sh      # fetch audio from RSS
bash pipeline/20_transcribe.sh data/episodes/ data/transcripts/
uv run --with scikit-learn --with umap-learn --with matplotlib --with numpy \
    python3 pipeline/30_cluster.py         # cluster transcripts
```

Then run extraction and synthesis passes with Claude Code — see [docs/pipeline-guide.md](docs/pipeline-guide.md).

**Add new episodes** (incremental update):

```bash
bash pipeline/42_update-episodes.sh
```

Follow the printed instructions. See [docs/update-guide.md](docs/update-guide.md).

**Browse the knowledge base:**

```bash
python3 -m http.server -d site
# open http://localhost:8000
```

Or regenerate after updates:

```bash
python3 pipeline/50_build-site.py
```

## Prerequisites

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with large-v3 model (built by `00_setup-whisper.sh`)
- [uv](https://github.com/astral-sh/uv) — Python package runner
- [Claude Code](https://claude.ai/claude-code) — for LLM extraction and synthesis passes
- ffmpeg, curl
