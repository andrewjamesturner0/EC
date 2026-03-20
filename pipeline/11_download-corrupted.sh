#!/usr/bin/env bash
set -euo pipefail

# Download audio ONLY for corrupted transcripts listed in corrupted_transcripts.md

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RSS_URL=$(tr -d '[:space:]' < "$ROOT/data/rss-url.txt")
DOWNLOAD_DIR="$ROOT/data/episodes"
CORRUPTED_MD="$ROOT/logs/corrupted_transcripts.md"

if [ ! -f "$CORRUPTED_MD" ]; then
    echo "ERROR: $CORRUPTED_MD not found"
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

# 1. Extract numeric prefixes from corrupted_transcripts.md
#    Lines look like: | `636044361-...-slug.txt` | ...
#    We extract the numeric prefix before the first dash
mapfile -t corrupted_ids < <(
    grep -oP '`\K[0-9]+(?=-\.\.\.)' "$CORRUPTED_MD" | sort -u
)

echo "Found ${#corrupted_ids[@]} corrupted transcript IDs"

if [ ${#corrupted_ids} -eq 0 ]; then
    echo "ERROR: No corrupted IDs extracted from $CORRUPTED_MD"
    exit 1
fi

# Build an associative array for fast lookup
declare -A id_lookup
for id in "${corrupted_ids[@]}"; do
    id_lookup["$id"]=1
done

# 2. Fetch RSS feed and extract enclosure URLs
echo "Fetching RSS feed: $RSS_URL"
feed_content=$(curl -sL "$RSS_URL")

mapfile -t urls < <(
    echo "$feed_content" | grep -oP '<enclosure[^>]+url="[^"]+' | grep -oP 'url="\K[^"]+'
)

echo "Found ${#urls[@]} episodes in RSS feed"

# 3. Match corrupted transcripts to their audio URLs by numeric prefix
matched=0
unmatched_ids=()
download_urls=()
download_names=()

# For each URL, extract the track ID from the filename portion
declare -A url_by_id
for url in "${urls[@]}"; do
    # URL looks like: .../stream/2265497594-empiricalcyclingpodcast-slug.mp3
    filename=$(basename "$url" | sed 's/?.*//')  # strip query params
    track_id=$(echo "$filename" | grep -oP '^[0-9]+' || true)
    if [ -n "$track_id" ]; then
        url_by_id["$track_id"]="$url"
    fi
done

echo "Parsed ${#url_by_id[@]} track IDs from RSS feed"
echo "==========================================="

# Match each corrupted ID to its URL
for id in "${corrupted_ids[@]}"; do
    if [ -n "${url_by_id[$id]+x}" ]; then
        url="${url_by_id[$id]}"
        filename=$(basename "$url" | sed 's/?.*//')
        download_urls+=("$url")
        download_names+=("$filename")
        matched=$(( matched + 1 ))
    else
        unmatched_ids+=("$id")
    fi
done

echo "Matched: $matched / ${#corrupted_ids[@]}"
if [ ${#unmatched_ids[@]} -gt 0 ]; then
    echo ""
    echo "UNMATCHED IDs (not found in RSS feed):"
    for id in "${unmatched_ids[@]}"; do
        echo "  - $id"
    done
fi
echo "==========================================="

if [ $matched -eq 0 ]; then
    echo "No matches found. Nothing to download."
    exit 1
fi

# 4. Download matched audio files
count=0
downloaded=0
skipped=0
failed=0

for i in "${!download_urls[@]}"; do
    count=$(( count + 1 ))
    url="${download_urls[$i]}"
    filename="${download_names[$i]}"

    if [ -f "$DOWNLOAD_DIR/$filename" ]; then
        echo "[$count/$matched] SKIP (exists): $filename"
        skipped=$(( skipped + 1 ))
    else
        echo "[$count/$matched] Downloading: $filename"
        if curl -L -o "$DOWNLOAD_DIR/$filename" "$url" 2>/dev/null; then
            downloaded=$(( downloaded + 1 ))
        else
            echo "  FAILED to download: $filename"
            rm -f "$DOWNLOAD_DIR/$filename"
            failed=$(( failed + 1 ))
        fi
    fi
done

# 5. Summary
echo "==========================================="
echo "Summary:"
echo "  Corrupted transcripts: ${#corrupted_ids[@]}"
echo "  Matched in RSS feed:   $matched"
echo "  Downloaded:            $downloaded"
echo "  Already existed:       $skipped"
echo "  Failed:                $failed"
echo "  Unmatched:             ${#unmatched_ids[@]}"
echo "  Audio saved to:        $DOWNLOAD_DIR/"
