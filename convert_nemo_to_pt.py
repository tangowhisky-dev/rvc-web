#!/usr/bin/env python3
"""
Convert ECAPA-TDNN .nemo model to PyTorch .pt format and validate.

Usage:
    python convert_nemo_to_pt.py
"""

import os
import sys
import torch
import yaml
import nemo.collections.asr as nemo_asr


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets", "ecapa")
NEMO_PATH = os.path.join(ASSETS_DIR, "ecapa_tdnn.nemo")
OUTPUT_PT_PATH = os.path.join(ASSETS_DIR, "ecapa_tdnn.pt")
OUTPUT_CONFIG_PATH = os.path.join(ASSETS_DIR, "ecapa_tdnn_config.yaml")


def main():
    print(f"Input: {NEMO_PATH}")
    print(f"Output: {OUTPUT_PT_PATH}")

    # Load the model using NeMo's restore_from
    print("\nLoading NeMo model...")
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(NEMO_PATH)
    model.eval()

    # Get the state dict
    state_dict = model.state_dict()
    print(f"Model state_dict has {len(state_dict)} keys")
    print(f"First 10 keys: {list(state_dict.keys())[:10]}")

    # Get model config
    config = model.cfg

    # Save the state dict
    checkpoint = {
        "state_dict": state_dict,
        "config": dict(config) if hasattr(config, "__dict__") else config,
        "model_class": "EncDecSpeakerLabelModel",
        "model_type": "ecapa_tdnn",
    }

    torch.save(checkpoint, OUTPUT_PT_PATH)
    print(f"\nSaved PyTorch model to {OUTPUT_PT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PT_PATH) / 1024 / 1024:.2f} MB")

    # Save config separately
    with open(OUTPUT_CONFIG_PATH, "w") as f:
        yaml.dump(
            dict(config) if hasattr(config, "__dict__") else config,
            f,
            default_flow_style=False,
        )
    print(f"Saved config to {OUTPUT_CONFIG_PATH}")

    # Validate by loading back
    print("\nValidating saved model...")
    loaded = torch.load(OUTPUT_PT_PATH, map_location="cpu", weights_only=False)

    orig_keys = set(state_dict.keys())
    loaded_keys = set(loaded["state_dict"].keys())

    if orig_keys == loaded_keys:
        print("✓ All keys match")
    else:
        print(
            f"✗ Key mismatch: missing={orig_keys - loaded_keys}, extra={loaded_keys - orig_keys}"
        )

    all_match = all(
        torch.equal(state_dict[k], loaded["state_dict"][k]) for k in orig_keys
    )
    print(f"{'✓' if all_match else '✗'} All weights match")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_audio = torch.randn(1, 16000)
    audio_length = torch.tensor([16000])

    with torch.no_grad():
        output = model.forward(
            input_signal=dummy_audio, input_signal_length=audio_length
        )

    if isinstance(output, tuple):
        print(
            f"Output shapes: {[o.shape if hasattr(o, 'shape') else o for o in output]}"
        )
    else:
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else output}")

    print("\n✓ Conversion and validation complete!")
    print(f"PyTorch model: {OUTPUT_PT_PATH}")
    print(f"Config: {OUTPUT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
