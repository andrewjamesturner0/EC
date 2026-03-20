import os
import re
from collections import Counter
from pathlib import Path

transcript_dir = str(Path(__file__).resolve().parent.parent / "data" / "transcripts")
results = []

for fname in sorted(os.listdir(transcript_dir)):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(transcript_dir, fname)
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    if total_lines == 0:
        results.append((fname, "EMPTY", 0, 0, 0, ""))
        continue
    
    # Count line frequencies (strip whitespace for comparison)
    stripped = [l.strip() for l in lines if l.strip()]
    total_nonblank = len(stripped)
    
    if total_nonblank == 0:
        results.append((fname, "EMPTY", 0, 0, 0, ""))
        continue
    
    counter = Counter(stripped)
    most_common_line, most_common_count = counter.most_common(1)[0]
    
    # Calculate total words
    text = "".join(lines)
    word_count = len(text.split())
    
    # Repeated line ratio
    repeat_ratio = most_common_count / total_nonblank
    
    # Find runs of repeated lines (consecutive identical lines)
    max_run = 1
    current_run = 1
    run_line = ""
    run_start = 0
    best_run_start = 0
    for i in range(1, len(stripped)):
        if stripped[i] == stripped[i-1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                run_line = stripped[i]
                best_run_start = run_start
        else:
            current_run = 1
            run_start = i
    
    # Flag if: most common line appears >20% of all lines, or max consecutive run > 15
    severity = "ok"
    if most_common_count > total_nonblank * 0.5:
        severity = "SEVERE"
    elif most_common_count > total_nonblank * 0.2 or max_run > 30:
        severity = "MODERATE"
    elif max_run > 15:
        severity = "MINOR"
    
    if severity != "ok":
        # Estimate usable words: count words in non-repeated-line content
        repeated_line = most_common_line
        usable_lines = [l for l in stripped if l != repeated_line]
        usable_words = sum(len(l.split()) for l in usable_lines)
        
        truncated_line = most_common_line[:80] + "..." if len(most_common_line) > 80 else most_common_line
        results.append((fname, severity, word_count, most_common_count, total_nonblank, max_run, truncated_line, usable_words))

# Print results
print(f"Scanned {len(os.listdir(transcript_dir))} files\n")
print(f"Found {len(results)} corrupted transcripts:\n")

for r in sorted(results, key=lambda x: x[1]):
    fname, severity, wc, rep_count, total, max_run, line, usable = r
    pct = rep_count/total*100 if total > 0 else 0
    print(f"[{severity}] {fname}")
    print(f"  Words: {wc:,} | Repeated: {rep_count}/{total} lines ({pct:.0f}%) | Max consecutive run: {max_run}")
    print(f"  Repeated line: \"{line}\"")
    print(f"  Estimated usable words: ~{usable:,}")
    print()
