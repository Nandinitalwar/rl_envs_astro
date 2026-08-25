#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
if ! python3 /tests/grader.py; then
  printf '%s\n' '{"reward": 0.0, "success": 0.0, "distance_progress": 0.0, "efficiency": 0.0}' > /logs/verifier/reward.json
fi
