#!/usr/bin/env bash
# Repeatedly run the FCTP data generator and solver.
# Default: infinite loop, sequential execution to avoid race conditions.
# Usage:
#   ./run_fctp_loop.sh                # infinite loop
#   ./run_fctp_loop.sh -n 100         # run 100 iterations
#   ./run_fctp_loop.sh -n 50 -d 2     # 50 iterations with 2s delay between runs
#   ./run_fctp_loop.sh --pipe         # use pipeline instead of sequential (not recommended)
#   ./run_fctp_loop.sh -n 10 --pipe   # 10 iterations using pipe

set -euo pipefail

ITERATIONS=-1   # -1 means infinite
DELAY=0         # seconds
USE_PIPE=false

usage() {
  cat <<'EOF'
run_fctp_loop.sh - loop FCTP instance generation and solve

Options:
  -n NUM        Number of iterations to run (default: infinite)
  -d SECONDS    Delay between iterations in seconds (default: 0)
  --pipe        Use pipeline: "python3 fctp_input_data_5_5.py | python3 fctp_gap_5_5.py"
                Note: safer is sequential with && to ensure data file is fully written before solving
  -h, --help    Show this help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for -n"; exit 2; }
      ITERATIONS="$1";
      [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || { echo "-n must be a non-negative integer"; exit 2; }
      ;;
    -d)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for -d"; exit 2; }
      DELAY="$1";
      [[ "$DELAY" =~ ^[0-9]+$ ]] || { echo "-d must be a non-negative integer"; exit 2; }
      ;;
    --pipe)
      USE_PIPE=true
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown argument: $1"; usage; exit 2
      ;;
  esac
  shift
done

# Always run from the script directory so relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

trap 'echo; echo "Stopping loop."; exit 0' INT

run_once() {
  local ts="$(date +'%F %T')"
  echo "[$ts] Starting iteration $1..."
  if $USE_PIPE; then
    # Pipeline requested by user; solver may ignore stdin, but this still runs both
    python3 fctp_input_data_5_5.py | python3 fctp_gap_5_5.py
  else
    # Safer sequential execution: ensure data is generated before solving
    python3 fctp_input_data_5_5.py && python3 fctp_gap_5_5.py
  fi
  echo "[$ts] Iteration $1 completed."
}

i=1
while :; do
  run_once "$i"
  if [[ "$ITERATIONS" -gt 0 && "$i" -ge "$ITERATIONS" ]]; then
    echo "Reached $ITERATIONS iterations. Exiting."; break
  fi
  i=$(( i + 1 ))
  if [[ "$DELAY" -gt 0 ]]; then
    sleep "$DELAY"
  fi
done
