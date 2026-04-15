# backend/beatrice2 — Beatrice 2 voice conversion pipeline
#
# Package layout:
#   beatrice_trainer/   — original trainer (refactored in M2)
#   training.py         — BeatriceTrainingJob wrapper for FastAPI
#   inference.py        — BeatriceInferenceEngine for offline + realtime
#   preprocess.py       — audio preprocessing (split into 4 s chunks)
#   download_assets.py  — asset downloader (updated paths for rvc-web)
