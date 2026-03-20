#!/usr/bin/env bash
set -euo pipefail

WHISPER_DIR="$HOME/whisper.cpp"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
MODEL="$WHISPER_DIR/models/ggml-large-v3.bin"

# Parse arguments: <input-dir> [output-dir] [jobs] [--retranscribe]
RETRANSCRIBE=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --retranscribe) RETRANSCRIBE=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

INPUT_DIR="${POSITIONAL[0]:?Usage: $0 <input-dir> [output-dir] [jobs] [--retranscribe]}"
OUTPUT_DIR="${POSITIONAL[1]:-data/transcripts}"
JOBS="${POSITIONAL[2]:-8}"

# CPU threads per worker — share cores across parallel jobs.
# With GPU acceleration, CPU threads handle pre/post-processing only;
# floor at 4 to prevent preprocessing starvation regardless of JOBS count.
# H200 (143GB VRAM): large-v3 is ~3GB, so 16+ parallel jobs fit easily.
THREADS=$(( $(nproc) / JOBS ))
THREADS=$(( THREADS < 4 ? 4 : THREADS ))

# Minimum file size (bytes) to consider a transcript valid
MIN_SIZE=100

# Kill all child processes on exit/interrupt
trap 'kill 0 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

mkdir -p "$OUTPUT_DIR"

# Set up log file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/transcribe_$(date +%Y%m%d_%H%M%S).log"
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >> "$LOG_FILE"
}

log "=== Transcription run started ==="
log "Input dir: $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "Parallel workers: $JOBS (${THREADS} CPU threads each)"
log "Retranscribe mode: $( [ $RETRANSCRIBE -eq 1 ] && echo 'ON' || echo 'OFF' )"

# Collect audio files
mapfile -t files < <(find "$INPUT_DIR" -type f \
    \( -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.wav" \
       -o -iname "*.ogg" -o -iname "*.flac" -o -iname "*.aac" \) | sort)

total=${#files[@]}
if [ "$total" -eq 0 ]; then
    echo "No audio files found in $INPUT_DIR"
    log "No audio files found. Exiting."
    exit 1
fi

echo "Found $total audio files in $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Parallel workers: $JOBS  (${THREADS} CPU threads each)"
echo "Log file: $LOG_FILE"
[ $RETRANSCRIBE -eq 1 ] && echo "RETRANSCRIBE mode: ignoring existing transcripts"
echo "==========================================="

log "Found $total audio files"

# Shared counters via temp files (atomic-ish for bash)
COUNT_DIR=$(mktemp -d)
trap 'rm -rf "$COUNT_DIR"; kill 0 2>/dev/null; wait 2>/dev/null' EXIT INT TERM
echo 0 > "$COUNT_DIR/done"
echo 0 > "$COUNT_DIR/skipped"
echo 0 > "$COUNT_DIR/failed"
echo 0 > "$COUNT_DIR/retried"

start_time=$SECONDS

# Detect corruption: count max consecutive repeated lines
# Returns the count via stdout. A file with 3+ consecutive identical lines is "corrupted".
count_max_consecutive_repeats() {
    local file="$1"
    awk '
    {
        if ($0 == prev) {
            run++
        } else {
            if (run > max) max = run
            run = 1
        }
        prev = $0
    }
    END { if (run > max) max = run; print (max ? max : 0) }
    ' "$file"
}

# Count total repeated lines (lines that are the same as the previous line)
count_total_repeated_lines() {
    local file="$1"
    awk '
    NR > 1 && $0 == prev { count++ }
    { prev = $0 }
    END { print (count ? count : 0) }
    ' "$file"
}

transcribe_one() {
    local f="$1"
    local name out_file tmp_wav

    name=$(basename "$f" | sed 's/\.[^.]*$//')
    out_file="$OUTPUT_DIR/${name}.txt"
    tmp_wav=$(mktemp /tmp/whisper_XXXX.wav)
    trap 'rm -f "$tmp_wav"' RETURN

    # Resume check: skip if file exists, is large enough, and not in retranscribe mode
    if [ $RETRANSCRIBE -eq 0 ] && [ -f "$out_file" ] && [ "$(stat -c%s "$out_file" 2>/dev/null || echo 0)" -ge "$MIN_SIZE" ]; then
        echo "SKIP (exists): $name"
        log "SKIP: $name (existing file $(stat -c%s "$out_file") bytes)"
        flock "$COUNT_DIR/skipped" -c "echo \$(( \$(cat \"$COUNT_DIR/skipped\") + 1 )) > \"$COUNT_DIR/skipped\""
        return
    fi

    # Convert to 16kHz mono WAV
    if ! ffmpeg -y -i "$f" -ar 16000 -ac 1 "$tmp_wav" 2>/dev/null; then
        echo "FAILED (ffmpeg): $name"
        log "FAIL: $name (ffmpeg conversion failed)"
        flock "$COUNT_DIR/failed" -c "echo \$(( \$(cat \"$COUNT_DIR/failed\") + 1 )) > \"$COUNT_DIR/failed\""
        return
    fi

    # In retranscribe mode, skip temp=0 and go straight to temp=0.4 with -mc 0 —
    # these files are already known-corrupted so the temp=0 pass is wasted work.
    # -mc 0: disable previous-text conditioning to break self-reinforcing loops.
    # --temperature 0.4: higher than pass 1 (0.2) to escape deeper attractor states.
    # --entropy-thold 1.8: more aggressive internal per-segment fallback (default 2.4).
    local temp_used="0"
    if [ $RETRANSCRIBE -eq 0 ]; then
        if ! "$WHISPER_BIN" -m "$MODEL" -t "$THREADS" -f "$tmp_wav" \
                -of "$OUTPUT_DIR/$name" -otxt --no-prints -fa 2>/dev/null; then
            echo "FAILED (whisper): $name"
            log "FAIL: $name (whisper failed on first attempt)"
            flock "$COUNT_DIR/failed" -c "echo \$(( \$(cat \"$COUNT_DIR/failed\") + 1 )) > \"$COUNT_DIR/failed\""
            return
        fi
    fi

    # Check for corruption (3+ consecutive identical lines).
    # In retranscribe mode the file doesn't exist yet, so treat as corrupted.
    local max_repeats=3
    if [ $RETRANSCRIBE -eq 0 ]; then
        max_repeats=$(count_max_consecutive_repeats "$out_file")
    fi

    if [ "$max_repeats" -ge 3 ]; then
        if [ $RETRANSCRIBE -eq 0 ]; then
            log "CORRUPTION DETECTED: $name (max $max_repeats consecutive repeated lines at temp=0)"
            echo "CORRUPTION DETECTED ($max_repeats repeated lines), retrying with temp=0.2: $name"
        else
            log "RETRANSCRIBE: $name (skipping temp=0, starting at temp=0.4 -mc 0)"
            echo "RETRANSCRIBE (starting at temp=0.4 -mc 0): $name"
        fi

        # Save first attempt stats (only if we ran temp=0)
        local first_repeated=999999
        if [ $RETRANSCRIBE -eq 0 ]; then
            first_repeated=$(count_total_repeated_lines "$out_file")
            cp "$out_file" "${out_file}.temp0"
        fi

        # Retry with temperature 0.4, no previous-text context, aggressive entropy threshold
        if "$WHISPER_BIN" -m "$MODEL" -t "$THREADS" -f "$tmp_wav" \
                -of "$OUTPUT_DIR/$name" -otxt --no-prints -fa \
                -mc 0 --temperature 0.4 --entropy-thold 1.8 2>/dev/null; then

            local retry_max_repeats retry_repeated
            retry_max_repeats=$(count_max_consecutive_repeats "$out_file")
            retry_repeated=$(count_total_repeated_lines "$out_file")

            if [ "$retry_max_repeats" -ge 3 ]; then
                log "RETRY ALSO CORRUPTED: $name (max $retry_max_repeats consecutive repeated lines at temp=0.2)"
                # In retranscribe mode there's no temp=0 fallback — keep temp=0.2 regardless
                if [ $RETRANSCRIBE -eq 0 ] && [ "$first_repeated" -le "$retry_repeated" ]; then
                    cp "${out_file}.temp0" "$out_file"
                    temp_used="0 (kept over 0.2: $first_repeated vs $retry_repeated repeated lines)"
                    log "KEPT temp=0 version: $first_repeated repeated lines vs $retry_repeated at temp=0.2"
                else
                    temp_used="0.2 (kept over 0: $retry_repeated vs $first_repeated repeated lines)"
                    log "KEPT temp=0.2 version: $retry_repeated repeated lines vs $first_repeated at temp=0"
                fi
            else
                temp_used="0.2"
                log "RETRY OK: $name (temp=0.2 clean, max consecutive repeats: $retry_max_repeats)"
            fi
        else
            # temp=0.2 failed entirely
            if [ $RETRANSCRIBE -eq 0 ]; then
                cp "${out_file}.temp0" "$out_file"
            fi
            temp_used="0.2 (failed)"
            log "RETRY FAILED: $name (whisper failed at temp=0.2)"
        fi

        rm -f "${out_file}.temp0"
        flock "$COUNT_DIR/retried" -c "echo \$(( \$(cat \"$COUNT_DIR/retried\") + 1 )) > \"$COUNT_DIR/retried\""
    fi

    # Success
    local done_count elapsed rate
    done_count=$(flock "$COUNT_DIR/done" -c "echo \$(( \$(cat \"$COUNT_DIR/done\") + 1 )) | tee \"$COUNT_DIR/done\"")
    elapsed=$(( SECONDS - start_time ))
    rate=$(awk "BEGIN {printf \"%.1f\", $done_count / ($elapsed/60.0)}" 2>/dev/null || echo "?")
    echo "OK [${done_count}/${total}] (${rate} files/min avg, temp=${temp_used}): $name"
    log "OK: $name (temp=$temp_used)"
}

export -f transcribe_one count_max_consecutive_repeats count_total_repeated_lines log
export WHISPER_BIN MODEL THREADS OUTPUT_DIR total start_time COUNT_DIR RETRANSCRIBE MIN_SIZE LOG_FILE

# Job pool using a named-pipe semaphore
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm -f "$FIFO"

# Pre-fill pipe with JOBS tokens
for (( i=0; i<JOBS; i++ )); do printf 'x' >&3; done

pids=()
for f in "${files[@]}"; do
    read -n1 -u3          # acquire a token (blocks if all workers busy)
    ( trap 'printf "x" >&3' EXIT; transcribe_one "$f" ) &
    pids+=($!)
done

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

elapsed=$(( SECONDS - start_time ))
hours=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))
done_count=$(cat "$COUNT_DIR/done")
skipped=$(cat "$COUNT_DIR/skipped")
failed=$(cat "$COUNT_DIR/failed")
retried=$(cat "$COUNT_DIR/retried")

echo "==========================================="
echo "Done! $total files: $done_count transcribed, $skipped skipped, $failed failed, $retried retried"
echo "Time: ${hours}h ${mins}m"
echo "Output: $OUTPUT_DIR/"
echo "Log: $LOG_FILE"

log "=== Transcription run finished ==="
log "Total: $total files"
log "Transcribed: $done_count"
log "Skipped: $skipped"
log "Failed: $failed"
log "Retried (corruption detected): $retried"
log "Time: ${hours}h ${mins}m"
