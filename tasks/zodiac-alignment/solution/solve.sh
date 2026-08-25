#!/bin/bash
set -euo pipefail

cat > /app/answer.json <<'JSON'
{"actions":[1,1,1,3,3,3,3,3,4,4,4,4,4,7,7,7,7,7,9,9,9,9,11,13,13]}
JSON
