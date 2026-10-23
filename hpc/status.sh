#!/bin/bash
# Shows your queued/running jobs and how far each training run has got.
# Usage: bash hpc/status.sh
cd "$(dirname "$0")/.."

squeue -u "$USER" -o "%.14i %.10j %.10P %.8T %.10M %.10l %.20R"
echo
for f in hpc/out/delf_*.out; do
    [ -e "$f" ] || continue
    echo "$(basename "$f" .out): $(tr '\r' '\n' < "$f" | grep -E '[0-9]+/[0-9]+ \[' | tail -1 | sed -E 's/\|[^|]*\| /| /')"
done
