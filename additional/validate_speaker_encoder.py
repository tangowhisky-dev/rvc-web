"""Cross-validate pure PyTorch ECAPA-TDNN against NeMo's implementation.

Runs in the rvc environment (no NeMo needed) but compares against
pre-computed NeMo embeddings saved by a companion script.

Usage:
    # Step 1: Generate NeMo reference embeddings (run in nemo env)
    conda run -n nemo python validate_speaker_encoder.py --gen-ref

    # Step 2: Compare PyTorch vs NeMo (run in rvc env)
    python validate_speaker_encoder.py
"""

import os
import sys
import argparse
import torch
import numpy as np


PT_PATH = os.path.join(os.path.dirname(__file__), "assets", "ecapa", "ecapa_tdnn.pt")
NEMO_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "ecapa", "ecapa_tdnn.nemo"
)
REF_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "ecapa", "nemo_ref_embeddings.pt"
)


def generate_nemo_reference():
    """Generate reference embeddings using NeMo (run in nemo env)."""
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model...")
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(NEMO_PATH)
    model.eval()

    torch.manual_seed(42)
    inputs = [torch.randn(1, 16000) for _ in range(10)]
    inputs.append(torch.randn(2, 16000))
    inputs.append(torch.randn(1, 32000))
    inputs.append(torch.randn(1, 8000))

    embeddings = []
    for i, inp in enumerate(inputs):
        with torch.no_grad():
            output = model(
                input_signal=inp, input_signal_length=torch.tensor([inp.shape[-1]])
            )
            if isinstance(output, tuple):
                emb = output[1]
            else:
                emb = output
            embeddings.append(emb.clone())
            print(
                f"  Input {i}: shape={tuple(inp.shape)} → emb shape={tuple(emb.shape)}"
            )

    torch.save({"embeddings": embeddings, "num_inputs": len(inputs)}, REF_PATH)
    print(f"\nSaved {len(inputs)} reference embeddings to {REF_PATH}")


def validate_pytorch():
    """Load pure PyTorch model and compare against NeMo references."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "rvc"))
    from infer.lib.train.speaker_encoder import SpeakerEncoder

    print("Loading pure PyTorch ECAPA-TDNN...")
    encoder = SpeakerEncoder(pt_path=PT_PATH, device=torch.device("cpu"))

    ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    ref_embeddings = ref["embeddings"]
    num_inputs = ref["num_inputs"]

    torch.manual_seed(42)
    inputs = [torch.randn(1, 16000) for _ in range(10)]
    inputs.append(torch.randn(2, 16000))
    inputs.append(torch.randn(1, 32000))
    inputs.append(torch.randn(1, 8000))

    print(f"\nComparing {num_inputs} inputs:")
    print("-" * 70)

    all_pass = True
    for i, inp in enumerate(inputs):
        with torch.no_grad():
            pt_emb = encoder.get_embedding(inp)

        ref_emb = ref_embeddings[i]
        diff = (pt_emb - ref_emb).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            pt_emb.flatten(), ref_emb.flatten(), dim=0
        ).item()

        status = "PASS" if diff < 1e-3 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  Input {i:2d}: shape={str(tuple(inp.shape)):15s} | "
            f"max_diff={diff:.2e} | cos_sim={cos_sim:.6f} | {status}"
        )

    print("-" * 70)
    if all_pass:
        print("ALL TESTS PASSED — pure PyTorch matches NeMo exactly")
    else:
        print("SOME TESTS FAILED — check implementation")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-ref",
        action="store_true",
        help="Generate NeMo reference embeddings (run in nemo env)",
    )
    args = parser.parse_args()

    if args.gen_ref:
        generate_nemo_reference()
    else:
        validate_pytorch()
