"""Stringent layer-by-layer validation of pure PyTorch ECAPA-TDNN vs NeMo.

Step 1: Generate NeMo reference outputs (run in nemo env):
    conda run -n nemo python validate_speaker_encoder_strict.py --gen-ref

Step 2: Compare PyTorch vs NeMo layer-by-layer (run in rvc env):
    python validate_speaker_encoder_strict.py
"""

import os
import sys
import argparse
import torch


PT_PATH = os.path.join(os.path.dirname(__file__), "assets", "ecapa", "ecapa_tdnn.pt")
NEMO_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "ecapa", "ecapa_tdnn.nemo"
)
REF_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "ecapa", "nemo_strict_ref.pt"
)


def generate_nemo_reference():
    """Generate layer-by-layer reference outputs using NeMo."""
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model...")
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(NEMO_PATH)
    model.eval()

    test_inputs = [
        ("short_1ch", torch.randn(1, 8000)),
        ("std_1ch", torch.randn(1, 16000)),
        ("std_2ch", torch.randn(2, 16000)),
        ("long_1ch", torch.randn(1, 32000)),
        ("long_4ch", torch.randn(4, 48000)),
    ]

    refs = {"test_cases": []}

    for name, audio in test_inputs:
        tc = {
            "name": name,
            "audio": audio.clone(),
            "audio_shape": tuple(audio.shape),
            "layers": {},
        }

        with torch.no_grad():
            # Preprocessor
            mel, mel_len = model.preprocessor(
                input_signal=audio,
                length=torch.tensor([audio.shape[-1]] * audio.shape[0]),
            )
            tc["layers"]["preprocessor"] = {
                "mel": mel.clone(),
                "mel_len": mel_len.clone(),
            }

            # Encoder
            encoded, enc_len = model.encoder(audio_signal=mel, length=mel_len)
            tc["layers"]["encoder"] = {
                "encoded": encoded.clone(),
                "enc_len": enc_len.clone() if enc_len is not None else None,
            }

            # Decoder - pooling
            pool = model.decoder._pooling(encoded, enc_len)
            tc["layers"]["decoder_pool"] = pool.clone()

            # Decoder - embedding layers
            last_pool = pool
            emb_layer_outs = []
            for i, layer in enumerate(model.decoder.emb_layers):
                last_pool = pool
                pool = layer(pool)
                emb_layer_outs.append(pool.clone())
            tc["layers"]["decoder_emb_layers"] = emb_layer_outs

            # Decoder - final embedding
            emb = model.decoder.emb_layers[-1](last_pool)
            tc["layers"]["decoder_emb"] = emb.clone()

            # Full model output
            logits, embs = model(
                input_signal=audio,
                input_signal_length=torch.tensor([audio.shape[-1]] * audio.shape[0]),
            )
            tc["layers"]["full_model"] = {
                "logits": logits.clone(),
                "embs": embs.clone(),
            }

        refs["test_cases"].append(tc)
        print(
            f"  {name}: audio={tuple(audio.shape)} "
            f"→ mel={tuple(mel.shape)} "
            f"→ encoded={tuple(encoded.shape)} "
            f"→ embs={tuple(embs.shape)}"
        )

    torch.save(refs, REF_PATH)
    print(f"\nSaved {len(test_inputs)} test cases to {REF_PATH}")


def validate_pytorch():
    """Load pure PyTorch model and compare layer-by-layer against NeMo references."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "rvc"))
    from infer.lib.train.speaker_encoder import ECAPATDNN, SpeakerEncoder

    print("Loading pure PyTorch ECAPA-TDNN...")
    encoder = SpeakerEncoder(pt_path=PT_PATH, device=torch.device("cpu"))
    model = encoder.model

    ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    test_cases = ref["test_cases"]

    print(f"\nValidating {len(test_cases)} test cases layer-by-layer:")
    print("=" * 80)

    all_pass = True
    total_checks = 0
    passed_checks = 0

    for tc in test_cases:
        name = tc["name"]
        audio = tc["audio"]
        print(f"\n  Test: {name} (audio shape: {tc['audio_shape']})")
        print("-" * 60)

        with torch.no_grad():
            # --- Preprocessor ---
            mel_pt, mel_len_pt = model.preprocessor(audio)
            mel_ref = tc["layers"]["preprocessor"]["mel"]
            mel_len_ref = tc["layers"]["preprocessor"]["mel_len"]

            diff_mel = (mel_pt - mel_ref).abs().max().item()
            diff_len = (mel_len_pt - mel_len_ref).abs().max().item()
            status = "PASS" if diff_mel < 1e-5 and diff_len == 0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            print(
                f"    Preprocessor:       mel_diff={diff_mel:.2e} | len_diff={diff_len} | {status}"
            )

            # --- Encoder ---
            encoded_pt, enc_len_pt = model.encoder(mel_pt, mel_len_pt)
            encoded_ref = tc["layers"]["encoder"]["encoded"]

            diff = (encoded_pt - encoded_ref).abs().max().item()
            # Encoder has many layers — allow slightly higher tolerance for accumulated fp32 error
            threshold = 5e-5
            status = "PASS" if diff < threshold else "FAIL"
            if status == "FAIL":
                all_pass = False
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            print(f"    Encoder output:     max_diff={diff:.2e} | {status}")

            # --- Decoder pooling ---
            pool_pt = model.decoder._pooling(encoded_pt, enc_len_pt)
            pool_ref = tc["layers"]["decoder_pool"]

            diff = (pool_pt - pool_ref).abs().max().item()
            status = "PASS" if diff < 1e-5 else "FAIL"
            if status == "FAIL":
                all_pass = False
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            print(f"    Decoder pool:       max_diff={diff:.2e} | {status}")

            # --- Decoder embedding layers ---
            last_pool = pool_pt
            for i, layer in enumerate(model.decoder.emb_layers):
                last_pool = pool_pt
                pool_pt = layer(pool_pt)
                emb_ref = tc["layers"]["decoder_emb_layers"][i]

                diff = (pool_pt - emb_ref).abs().max().item()
                status = "PASS" if diff < 1e-5 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                total_checks += 1
                if status == "PASS":
                    passed_checks += 1
                print(f"    Decoder emb_layer[{i}]: max_diff={diff:.2e} | {status}")

            # --- Final embedding ---
            emb_pt = model.decoder.emb_layers[-1](last_pool)
            emb_ref = tc["layers"]["decoder_emb"]

            diff = (emb_pt - emb_ref).abs().max().item()
            status = "PASS" if diff < 1e-5 else "FAIL"
            if status == "FAIL":
                all_pass = False
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            print(f"    Decoder emb:        max_diff={diff:.2e} | {status}")

            # --- Full model output ---
            logits_pt, embs_pt = model.decoder(encoded_pt, enc_len_pt)
            logits_ref = tc["layers"]["full_model"]["logits"]
            embs_ref = tc["layers"]["full_model"]["embs"]

            diff_logits = (logits_pt - logits_ref).abs().max().item()
            diff_embs = (embs_pt - embs_ref).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                embs_pt.flatten(), embs_ref.flatten(), dim=0
            ).item()

            status = "PASS" if diff_embs < 1e-5 else "FAIL"
            if status == "FAIL":
                all_pass = False
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            print(f"    Full model logits:  max_diff={diff_logits:.2e}")
            print(
                f"    Full model embs:    max_diff={diff_embs:.2e} | cos_sim={cos_sim:.8f} | {status}"
            )

    print("\n" + "=" * 80)
    print(f"Results: {passed_checks}/{total_checks} checks passed")
    if all_pass:
        print("ALL TESTS PASSED — pure PyTorch matches NeMo exactly at every layer")
    else:
        print("SOME TESTS FAILED — check implementation")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-ref",
        action="store_true",
        help="Generate NeMo reference outputs (run in nemo env)",
    )
    args = parser.parse_args()

    if args.gen_ref:
        generate_nemo_reference()
    else:
        validate_pytorch()
