#!/usr/bin/env bash
# ------------------------------------------------------------
#  Launch Crystal Studio locally (the interactive demo).
#
#  Before running, put the trained checkpoint in the repo root:
#      fno_grains_tuned.pt        (the physics_weight=0.02 model)
#  Download it from Google Drive (MyDrive/fno_grains_tuned.pt).
#
#  Then just:   bash crystal_tool/run_local.sh
#  It opens in your browser at http://localhost:8501
# ------------------------------------------------------------
cd "$(dirname "$0")/.." || exit 1                  # repo root (so the ckpt path resolves)
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if [ ! -f fno_grains_tuned.pt ]; then
  echo "WARNING: fno_grains_tuned.pt not found in $(pwd)."
  echo "Download it from Google Drive (MyDrive/fno_grains_tuned.pt) and place it here,"
  echo "or set a different path in the app's sidebar once it opens."
  echo ""
fi

exec python3 -m streamlit run crystal_tool/app.py "$@"
