#!/bin/bash
# Download all episodes from a podcast RSS feed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RSS_URL=$(cat "$ROOT/data/rss-url.txt" | tr -d '[:space:]')
DOWNLOAD_DIR="$ROOT/data/episodes"

mkdir -p "$DOWNLOAD_DIR"

echo "Fetching RSS feed: $RSS_URL"

# Extract enclosure URLs from the feed
urls=$(curl -sL "$RSS_URL" | grep -oP '<enclosure[^>]+url="[^"]+' | grep -oP 'url="\K[^"]+')

if [ -z "$urls" ]; then
    echo "No episodes found in feed."
    exit 1
fi

total=$(echo "$urls" | wc -l)
echo "Found $total episodes."

count=0
while IFS= read -r url; do
    count=$((count + 1))
    filename=$(basename "$url" | sed 's/?.*//')  # strip query params
    if [ -f "$DOWNLOAD_DIR/$filename" ]; then
        echo "[$count/$total] Already exists: $filename"
    else
        echo "[$count/$total] Downloading: $filename"
        curl -L -o "$DOWNLOAD_DIR/$filename" "$url"
    fi
done <<< "$urls"

echo "Done. Episodes saved to $DOWNLOAD_DIR"
