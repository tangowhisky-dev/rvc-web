import os, pathlib

import torch
from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                str(pathlib.Path(root, name))
                for path in [os.getenv("outside_index_root"), os.getenv("index_root")]
                for root, _, files in os.walk(path, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(device, is_half):
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        hubert_path = os.path.join(project_root, "assets", "hubert", "hubert_base.pt")
    else:
        hubert_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "assets", "hubert", "hubert_base.pt")
    try:
        from fairseq.data.dictionary import Dictionary
        torch.serialization.add_safe_globals([Dictionary])
    except (ImportError, AttributeError):
        pass
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
