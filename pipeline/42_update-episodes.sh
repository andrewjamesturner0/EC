#!/usr/bin/env bash
#
# Lightweight update orchestrator for new podcast episodes.
# Detects new transcripts, assigns to clusters, prints instructions
# for extraction, re-synthesis, and final assembly.
#
# Usage: bash pipeline/42_update-episodes.sh
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAP="$ROOT/output/cluster_theme_map.json"
ASSIGN_SCRIPT="$ROOT/pipeline/41_assign-new-episodes.py"
ASSIGNMENTS="$ROOT/pipeline/new_episode_assignments.json"

echo "=== EC Incremental Update ==="
echo ""

# Step 1: Detect and assign new episodes
echo "Step 1: Detecting new episodes..."
echo ""
uv run --with scikit-learn --with numpy python3 "$ASSIGN_SCRIPT"

if [[ $? -ne 0 ]] || [[ ! -f "$ASSIGNMENTS" ]]; then
    echo "No new episodes to process. Done."
    exit 0
fi

echo ""
echo "=========================================="
echo ""

# Step 2: Print assignments with detail
echo "Step 2: Cluster Assignments"
echo ""

read -r -d '' PYSCRIPT << 'PYEOF' || true
import json, sys

with open(sys.argv[1]) as f:
    assignments = json.load(f)
with open(sys.argv[2]) as f:
    theme_map = json.load(f)

cluster_to_extraction = theme_map["cluster_to_extraction"]
extraction_to_theme = theme_map["extraction_to_theme"]

# Group by cluster
by_cluster = {}
for a in assignments:
    cid = str(a["cluster_id"])
    by_cluster.setdefault(cid, []).append(a)

affected_themes = set()

for cid, episodes in sorted(by_cluster.items(), key=lambda x: int(x[0])):
    extraction_file = cluster_to_extraction.get(cid, "UNKNOWN")
    theme_key = extraction_to_theme.get(extraction_file)
    theme_name = theme_map["themes"][theme_key]["display_name"] if theme_key else "Uncategorized"
    affected_themes.add(theme_key)

    print(f"Cluster {cid} → {extraction_file}")
    print(f"  Theme: {theme_name}")
    for ep in episodes:
        warn = " ⚠ WEAK MATCH" if ep.get("warning") else ""
        print(f"  - {ep['filename']} (score={ep['similarity_score']:.4f}, {ep['word_count']} words){warn}")
        top3 = ep.get("top_3", [])
        if len(top3) > 1:
            alts = ", ".join(f"{t['cluster']}={t['score']:.4f}" for t in top3[1:])
            print(f"    Alternatives: {alts}")
    print()

# Step 3: Extraction instructions
print("==========================================")
print()
print("Step 3: Extraction Instructions")
print()
print("For each new episode, run extraction using prompts/extraction.md.")
print("Append output to the extraction file listed above.")
print()
print("Example Claude Code command (per episode):")
print('  Read prompts/extraction.md and data/transcripts/{FILENAME},')
print('  then append extraction output to {EXTRACTION_FILE}.')
print()

for cid, episodes in sorted(by_cluster.items(), key=lambda x: int(x[0])):
    extraction_file = cluster_to_extraction.get(cid, "UNKNOWN")
    for ep in episodes:
        print(f"  - Extract: {ep['filename']} → append to {extraction_file}")
print()

# Step 4: Re-synthesis instructions
print("==========================================")
print()
print("Step 4: Re-synthesis Instructions")
print()
print("After extraction, re-synthesise each affected theme.")
print("Read prompts/synthesis.md and the FULL extraction file(s),")
print("then write output to the synthesis file.")
print()

affected_themes.discard(None)
for theme_key in sorted(affected_themes):
    theme = theme_map["themes"][theme_key]
    source_clusters = theme["source_clusters"]
    extraction_files = [cluster_to_extraction[str(c)] for c in source_clusters]
    print(f"  Theme: {theme['display_name']}")
    print(f"    Synthesis file: {theme['synthesis_file']}")
    print(f"    Input extractions: {', '.join(extraction_files)}")
    print()

print("==========================================")
print()
print("Step 5: After LLM work is done, reassemble final output:")
print("  bash pipeline/40_assemble-final.sh")
print()
print("Step 6: Update clusters.json with new assignments:")
print("  uv run --with scikit-learn --with numpy python3 pipeline/41_assign-new-episodes.py --update-clusters")
PYEOF

python3 -c "$PYSCRIPT" "$ASSIGNMENTS" "$MAP"

echo ""
echo "Done. Follow the instructions above."
