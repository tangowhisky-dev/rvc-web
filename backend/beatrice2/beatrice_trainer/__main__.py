# Entry point for: python3 -m beatrice_trainer
#
# All model definitions, training loop, and utilities have been extracted into
# focused sub-modules (see package layout below). This file is kept as the
# runnable entry point only — it imports the training loop from train/loop.py
# and executes it.
#
# Package layout:
#   config.py              — hyperparams, repo_root, arg parsing
#   io.py                  — dump_params, dump_layer (binary serialisation)
#   layers/
#     conv.py              — CausalConv1d, WSConv1d, WSLinear
#     attention.py         — CrossAttention, ConvNeXtBlock, ConvNeXtStack
#     vq.py                — VectorQuantizer
#   models/
#     phone_extractor.py   — FeatureExtractor, FeatureProjection, PhoneExtractor
#     pitch_estimator.py   — extract_pitch_features, PitchEstimator
#     vocoder.py           — D4C synthesis helpers, Vocoder, compute_loudness
#     converter.py         — ConverterNetwork
#     discriminator.py     — SANConv2d, DiscriminatorP/R, MultiPeriodDiscriminator
#   train/
#     loss.py              — GradBalancer
#     evaluation.py        — QualityTester
#     checkpoint.py        — optimizer state compression helpers
#     loop.py              — prepare_training() + training loop
#   data/
#     augment.py           — augmentation helpers
#     dataset.py           — WavDataset
#     filelist.py          — AUDIO_FILE_SUFFIXES constant

# Re-import everything so that `from beatrice_trainer.__main__ import X` still
# works (backward compatibility with any tooling that imports from here directly).
from .config import *  # noqa: F401, F403
from .io import *  # noqa: F401, F403
from .layers import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403
from .train import *  # noqa: F401, F403
from .train.checkpoint import *  # noqa: F401, F403
from .data import *  # noqa: F401, F403

# The actual training loop lives in train/loop.py. When this module is the
# __main__ module (python3 -m beatrice_trainer), loop.py's
# `if __name__ == "__main__"` block does NOT fire (its __name__ is
# beatrice_trainer.train.loop, not __main__). We therefore replicate the
# entry-point logic here, delegating to prepare_training() which is imported
# above from train/loop.py.

if __name__ == "__main__":
    import sys, traceback
    from .train.loop import _run_main
    try:
        _run_main()
    except Exception:
        print("\n[FATAL] Unhandled exception in training loop:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
