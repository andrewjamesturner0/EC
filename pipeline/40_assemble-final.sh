#!/usr/bin/env bash
#
# Rebuild output/final_output.md from synthesis files and framing content.
# Reads theme ordering from output/cluster_theme_map.json.
#
# Usage: bash pipeline/40_assemble-final.sh
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAP="$ROOT/output/cluster_theme_map.json"
OUT="$ROOT/output/final_output.md"
FRAMING="$ROOT/output/framing"
SYNTH="$ROOT/output/synthesis"

# Check prerequisites
for f in "$FRAMING/executive_summary.md" "$FRAMING/cross_references.md" "$FRAMING/methodology_note.md"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing framing file: $f"
        echo "Extract framing content from final_output.md first."
        exit 1
    fi
done

if [[ ! -f "$MAP" ]]; then
    echo "ERROR: Missing $MAP"
    exit 1
fi

# Parse theme order and display names from JSON
# Using python since jq may not be available
read -r -d '' PYSCRIPT << 'PYEOF' || true
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
for key in m["theme_order"]:
    t = m["themes"][key]
    print(f"{key}\t{t['display_name']}\t{t['synthesis_file']}")
PYEOF

THEMES=$(python3 -c "$PYSCRIPT" "$MAP")

# Start building output
{
    echo "# Empirical Cycling Podcast: Training Knowledge Base"
    echo ""
    echo "## Executive Summary"
    echo ""
    cat "$FRAMING/executive_summary.md"
    echo ""

    # Table of Contents
    echo "## Table of Contents"
    echo ""
    section_num=1
    while IFS=$'\t' read -r key display_name synth_file; do
        # Build anchor: lowercase, spaces to hyphens, strip non-alphanumeric except hyphens
        anchor=$(echo "$section_num. $display_name" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g; s/[^a-z0-9-]//g; s/--*/-/g')
        echo "$section_num. [$display_name](#$anchor)"
        section_num=$((section_num + 1))
    done <<< "$THEMES"
    echo ""
    echo "---"
    echo ""

    # Theme sections
    echo ""
    section_num=1
    while IFS=$'\t' read -r key display_name synth_file; do
        synth_path="$ROOT/$synth_file"
        if [[ ! -f "$synth_path" ]]; then
            echo "WARNING: Missing synthesis file: $synth_path" >&2
            section_num=$((section_num + 1))
            continue
        fi

        echo "## $section_num. $display_name"
        echo ""

        # Read synthesis file, strip the top-level heading, downshift ## to ###
        tail -n +1 "$synth_path" | sed '1{/^# /d;}' | sed '/./,$!d' | sed 's/^## /### /'
        echo ""
        echo ""
        echo "---"
        echo ""

        section_num=$((section_num + 1))
    done <<< "$THEMES"

    echo "---"
    echo ""

    # Cross-references
    echo "## Cross-References"
    echo ""
    cat "$FRAMING/cross_references.md"
    echo ""

    # Methodology note
    echo "## Methodology Note"
    echo ""
    cat "$FRAMING/methodology_note.md"

} > "$OUT"

# Report
lines=$(wc -l < "$OUT")
words=$(wc -w < "$OUT")
echo "Assembled $OUT: $lines lines, $words words"
