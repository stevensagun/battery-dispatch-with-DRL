#!/bin/bash
# Show queued/running jobs and the latest output from stack experiments.
set -eo pipefail
cd "$(dirname "$0")/.."

squeue -u "$USER" -o "%.14i %.36j %.12P %.8T %.10M %.10l %.20R"
for file in hpc_stacks/out/stack*.out; do
    [ -e "$file" ] || continue
    printf '\n%s\n' "$file"
    tr '\r' '\n' < "$file" | tail -n 6
done
