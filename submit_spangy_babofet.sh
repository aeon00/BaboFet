#!/bin/bash
# Submit the SPANGY array job, then chain the merge job after it.
#   ./submit_spangy_babofet.sh                       # all subjects
#   ./submit_spangy_babofet.sh sub-Aziza sub-Borgne  # subset

set -euo pipefail

BASE=/envau/work/meca/users/dienye.h/python_files/Babofet
mkdir -p "$BASE/logs"

ARRAY_ID=$(sbatch --parsable "$BASE/run_spangy_babofet.sh" "$@")
echo "Array job submitted: $ARRAY_ID"

MERGE_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" \
                  "$BASE/merge_spangy_babofet.sh" "$@")
echo "Merge job submitted:  $MERGE_ID (runs after $ARRAY_ID)"
