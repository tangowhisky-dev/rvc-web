import os
import sys
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
    embedder_path = None          # legacy: use default hubert
elif len(sys.argv) == 8:
    # New form: device n_part i_part exp_dir version is_half embedder_path
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
    embedder_path = sys.argv[7] if sys.argv[7] else None
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
    embedder_path = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] else None

import numpy as np
import torch
import torch.nn.functional as F

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml
    import fairseq

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(" ".join(sys.argv))

# ---------------------------------------------------------------------------
# Resolve embedder path
# ---------------------------------------------------------------------------
# If no embedder_path arg was supplied, fall back to the PROJECT_ROOT-relative
# default (spin-v2 is our new default; hubert_base.pt is the legacy fairseq
# fallback kept for backward compat).
# ---------------------------------------------------------------------------

def _default_embedder_path() -> str:
    project_root = os.environ.get("PROJECT_ROOT", "")
    if project_root:
        candidate = os.path.join(project_root, "assets", "embedders", "spin-v2")
        if os.path.isdir(candidate):
            return candidate
        # Legacy fairseq hubert fallback
        return os.path.join(project_root, "assets", "hubert", "hubert_base.pt")
    # Relative fallback when PROJECT_ROOT is not set (shouldn't happen in prod)
    return os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "assets", "embedders", "spin-v2",
    )

if embedder_path is None:
    embedder_path = _default_embedder_path()

printt("exp_dir: " + exp_dir)
printt("embedder: " + embedder_path)

wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = load_audio(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# ---------------------------------------------------------------------------
# Load embedder — HuggingFace HuBERT (pytorch_model.bin + config.json) or
# legacy fairseq checkpoint (.pt).
# ---------------------------------------------------------------------------

printt("load model(s) from {}".format(embedder_path))

# Check whether the embedder is a HuggingFace directory or a fairseq .pt file
_use_hf = os.path.isdir(embedder_path) and os.path.exists(
    os.path.join(embedder_path, "config.json")
)

if _use_hf:
    # HuggingFace transformers HuBERT — used by SPIN, SPIN-v2, ContentVec,
    # and language-specific HuBERT variants.  No fairseq dependency needed.
    try:
        from torch import nn
        from transformers import HubertModel

        class HubertModelWithFinalProj(HubertModel):
            """HuBERT with an additional linear projection head.

            Matches the architecture expected by RVC v2 training:
            output dimension = classifier_proj_size (256 in standard HuBERT).
            The projection is loaded from the pretrained weights; the base
            model weights come from HuBERT/SPIN pretrained checkpoints.
            """
            def __init__(self, config):
                super().__init__(config)
                self.final_proj = nn.Linear(
                    config.hidden_size, config.classifier_proj_size
                )

        # Suppress noisy transformers load warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        model = HubertModelWithFinalProj.from_pretrained(embedder_path)
        model = model.to(device).float()
        model.eval()
        _is_hf_model = True
        printt("HuggingFace %s model loaded, moved to %s" % (os.path.basename(embedder_path), device))
    except Exception as e:
        printt("Failed to load HuggingFace model: %s" % traceback.format_exc())
        exit(1)
else:
    # Legacy fairseq checkpoint (.pt file)
    if not os.path.isfile(embedder_path):
        printt(
            "Error: Extracting is shut down because %s does not exist" % embedder_path
        )
        exit(0)
    try:
        import fairseq
        from fairseq.data.dictionary import Dictionary
        try:
            torch.serialization.add_safe_globals([Dictionary])
        except (ImportError, AttributeError):
            pass
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [embedder_path],
            suffix="",
        )
        model = models[0]
        model = model.to(device)
        if is_half:
            if device not in ["mps", "cpu"]:
                model = model.half()
        model.eval()
        _is_hf_model = False
        printt("fairseq model loaded, moved to %s" % device)
    except Exception:
        printt(traceback.format_exc())
        exit(1)

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # print at most 10 progress lines
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                if _is_hf_model:
                    # HuggingFace path — load directly, no layer_norm needed
                    # (HuBERT/SPIN accept raw waveform; model handles normalisation)
                    wav, sr = load_audio(wav_path)
                    assert sr == 16000
                    feats = torch.from_numpy(wav).float().to(device).view(1, -1)
                    with torch.no_grad():
                        result = model(feats)["last_hidden_state"]
                    feats_out = result.squeeze(0).float().cpu().numpy()
                else:
                    # Fairseq path — uses saved_cfg.task.normalize flag
                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if is_half and device not in ["mps", "cpu"]
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats_out_t = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )
                    feats_out = feats_out_t.squeeze(0).float().cpu().numpy()

                if np.isnan(feats_out).sum() == 0:
                    np.save(out_path, feats_out, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats_out.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
