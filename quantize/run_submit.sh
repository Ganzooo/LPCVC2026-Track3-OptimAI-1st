#!/bin/bash
# Package submission
# Usage: bash run_submit.sh [team_name]

TEAM=${1:-OptimAI}

python quantize/prepare_submission.py --team_name $TEAM --from_sample
