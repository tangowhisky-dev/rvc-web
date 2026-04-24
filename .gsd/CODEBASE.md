# Codebase Map

Generated: 2026-04-24T12:44:39Z | Files: 230 | Described: 0/230
<!-- gsd:codebase-meta {"generatedAt":"2026-04-24T12:44:39Z","fingerprint":"1b1b0c7e46516c87011e75bfe7f71c2a345d7ef0","fileCount":230,"truncated":false} -->

### (root)/
- `.env.example`
- `.gitignore`
- `install.bat`
- `install.sh`
- `pytest.ini`
- `README.md`
- `requirements.txt`
- `start.bat`
- `start.sh`
- `stop.bat`
- `stop.sh`

### additional/
- `additional/convert_nemo_to_pt.py`
- `additional/validate_speaker_encoder_strict.py`
- `additional/validate_speaker_encoder.py`

### assets/hubert/
- `assets/hubert/.gitignore`

### assets/indices/
- `assets/indices/.gitignore`

### assets/pretrained/
- `assets/pretrained/.gitignore`

### assets/pretrained_v2/
- `assets/pretrained_v2/.gitignore`

### assets/rmvpe/
- `assets/rmvpe/.gitignore`

### assets/uvr5_weights/
- `assets/uvr5_weights/.gitignore`

### assets/weights/
- `assets/weights/.gitignore`

### backend/
- `backend/__init__.py`

### backend/app/
- `backend/app/__init__.py`
- `backend/app/_index_worker.py`
- `backend/app/_realtime_worker.py`
- `backend/app/audio_preprocessing.py`
- `backend/app/db.py`
- `backend/app/main.py`
- `backend/app/realtime.py`
- `backend/app/training.py`
- `backend/app/voice_analysis.py`

### backend/app/routers/
- `backend/app/routers/__init__.py`
- `backend/app/routers/analysis.py`
- `backend/app/routers/devices.py`
- `backend/app/routers/offline.py`
- `backend/app/routers/profiles.py`
- `backend/app/routers/realtime.py`
- `backend/app/routers/training.py`

### backend/beatrice2/
- `backend/beatrice2/__init__.py`
- `backend/beatrice2/db_writer.py`
- `backend/beatrice2/download_assets.py`
- `backend/beatrice2/inference.py`
- `backend/beatrice2/preprocess.py`
- `backend/beatrice2/training.py`

### backend/beatrice2/beatrice_trainer/
- `backend/beatrice2/beatrice_trainer/__init__.py`
- `backend/beatrice2/beatrice_trainer/__main__.py`
- `backend/beatrice2/beatrice_trainer/config.py`
- `backend/beatrice2/beatrice_trainer/io.py`

### backend/beatrice2/beatrice_trainer/assets/
- `backend/beatrice2/beatrice_trainer/assets/default_config.json`

### backend/beatrice2/beatrice_trainer/data/
- `backend/beatrice2/beatrice_trainer/data/__init__.py`
- `backend/beatrice2/beatrice_trainer/data/augment.py`
- `backend/beatrice2/beatrice_trainer/data/dataset.py`
- `backend/beatrice2/beatrice_trainer/data/filelist.py`

### backend/beatrice2/beatrice_trainer/layers/
- `backend/beatrice2/beatrice_trainer/layers/__init__.py`
- `backend/beatrice2/beatrice_trainer/layers/attention.py`
- `backend/beatrice2/beatrice_trainer/layers/conv.py`
- `backend/beatrice2/beatrice_trainer/layers/vq.py`

### backend/beatrice2/beatrice_trainer/models/
- `backend/beatrice2/beatrice_trainer/models/__init__.py`
- `backend/beatrice2/beatrice_trainer/models/converter.py`
- `backend/beatrice2/beatrice_trainer/models/discriminator.py`
- `backend/beatrice2/beatrice_trainer/models/phone_extractor.py`
- `backend/beatrice2/beatrice_trainer/models/pitch_estimator.py`
- `backend/beatrice2/beatrice_trainer/models/vocoder.py`

### backend/beatrice2/beatrice_trainer/models/utmos/
- `backend/beatrice2/beatrice_trainer/models/utmos/__init__.py`
- `backend/beatrice2/beatrice_trainer/models/utmos/fairseq_alt.py`
- `backend/beatrice2/beatrice_trainer/models/utmos/model.py`

### backend/beatrice2/beatrice_trainer/train/
- `backend/beatrice2/beatrice_trainer/train/__init__.py`
- `backend/beatrice2/beatrice_trainer/train/checkpoint.py`
- `backend/beatrice2/beatrice_trainer/train/evaluation.py`
- `backend/beatrice2/beatrice_trainer/train/loop.py`
- `backend/beatrice2/beatrice_trainer/train/loss.py`

### backend/rvc/
- `backend/rvc/__init__.py`

### backend/rvc/configs/
- `backend/rvc/configs/__init__.py`
- `backend/rvc/configs/config.json`
- `backend/rvc/configs/config.py`

### backend/rvc/configs/inuse/v1/
- `backend/rvc/configs/inuse/v1/32k.json`
- `backend/rvc/configs/inuse/v1/40k.json`
- `backend/rvc/configs/inuse/v1/48k.json`

### backend/rvc/configs/inuse/v2/
- `backend/rvc/configs/inuse/v2/32k.json`
- `backend/rvc/configs/inuse/v2/48k.json`

### backend/rvc/i18n/
- `backend/rvc/i18n/__init__.py`
- `backend/rvc/i18n/i18n.py`

### backend/rvc/i18n/locale/
- `backend/rvc/i18n/locale/en_US.json`
- `backend/rvc/i18n/locale/es_ES.json`
- `backend/rvc/i18n/locale/fr_FR.json`
- `backend/rvc/i18n/locale/it_IT.json`
- `backend/rvc/i18n/locale/ja_JP.json`
- `backend/rvc/i18n/locale/ko_KR.json`
- `backend/rvc/i18n/locale/pt_BR.json`
- `backend/rvc/i18n/locale/ru_RU.json`
- `backend/rvc/i18n/locale/tr_TR.json`
- `backend/rvc/i18n/locale/zh_CN.json`
- `backend/rvc/i18n/locale/zh_HK.json`
- `backend/rvc/i18n/locale/zh_SG.json`
- `backend/rvc/i18n/locale/zh_TW.json`

### backend/rvc/infer/
- `backend/rvc/infer/__init__.py`

### backend/rvc/infer/lib/
- `backend/rvc/infer/lib/__init__.py`
- `backend/rvc/infer/lib/audio.py`
- `backend/rvc/infer/lib/rtrvc.py`
- `backend/rvc/infer/lib/slicer2.py`

### backend/rvc/infer/lib/train/
- `backend/rvc/infer/lib/train/__init__.py`
- `backend/rvc/infer/lib/train/adamspd.py`
- `backend/rvc/infer/lib/train/data_utils.py`
- `backend/rvc/infer/lib/train/db_writer.py`
- `backend/rvc/infer/lib/train/losses.py`
- `backend/rvc/infer/lib/train/mel_processing.py`
- `backend/rvc/infer/lib/train/process_ckpt.py`
- `backend/rvc/infer/lib/train/speaker_encoder.py`
- `backend/rvc/infer/lib/train/utils.py`

### backend/rvc/infer/modules/
- `backend/rvc/infer/modules/__init__.py`

### backend/rvc/infer/modules/train/
- `backend/rvc/infer/modules/train/__init__.py`
- `backend/rvc/infer/modules/train/extract_f0_print.py`
- `backend/rvc/infer/modules/train/extract_feature_print.py`
- `backend/rvc/infer/modules/train/extract_profile_embedding.py`
- `backend/rvc/infer/modules/train/preprocess.py`
- `backend/rvc/infer/modules/train/train.py`

### backend/rvc/infer/modules/vc/
- `backend/rvc/infer/modules/vc/__init__.py`
- `backend/rvc/infer/modules/vc/hash.py`
- `backend/rvc/infer/modules/vc/info.py`
- `backend/rvc/infer/modules/vc/lgdsng.npz`
- `backend/rvc/infer/modules/vc/modules.py`
- `backend/rvc/infer/modules/vc/pipeline.py`
- `backend/rvc/infer/modules/vc/rmvpe.py`
- `backend/rvc/infer/modules/vc/utils.py`

### backend/rvc/rvc/
- `backend/rvc/rvc/__init__.py`
- `backend/rvc/rvc/hubert.py`
- `backend/rvc/rvc/synthesizer.py`

### backend/rvc/rvc/f0/
- `backend/rvc/rvc/f0/__init__.py`
- `backend/rvc/rvc/f0/crepe.py`
- `backend/rvc/rvc/f0/deepunet.py`
- `backend/rvc/rvc/f0/dio.py`
- `backend/rvc/rvc/f0/e2e.py`
- `backend/rvc/rvc/f0/f0.py`
- `backend/rvc/rvc/f0/fcpe.py`
- `backend/rvc/rvc/f0/gen.py`
- `backend/rvc/rvc/f0/harvest.py`
- `backend/rvc/rvc/f0/mel.py`
- `backend/rvc/rvc/f0/models.py`
- `backend/rvc/rvc/f0/pm.py`
- `backend/rvc/rvc/f0/rmvpe.py`
- `backend/rvc/rvc/f0/stft.py`

### backend/rvc/rvc/ipex/
- `backend/rvc/rvc/ipex/__init__.py`
- `backend/rvc/rvc/ipex/attention.py`
- `backend/rvc/rvc/ipex/gradscaler.py`
- `backend/rvc/rvc/ipex/hijacks.py`
- `backend/rvc/rvc/ipex/init.py`

### backend/rvc/rvc/jit/
- `backend/rvc/rvc/jit/__init__.py`
- `backend/rvc/rvc/jit/jit.py`

### backend/rvc/rvc/layers/
- `backend/rvc/rvc/layers/__init__.py`
- `backend/rvc/rvc/layers/attentions.py`
- `backend/rvc/rvc/layers/discriminators.py`
- `backend/rvc/rvc/layers/encoders.py`
- `backend/rvc/rvc/layers/generators.py`
- `backend/rvc/rvc/layers/norms.py`
- `backend/rvc/rvc/layers/nsf.py`
- `backend/rvc/rvc/layers/refinegan.py`
- `backend/rvc/rvc/layers/residuals.py`
- `backend/rvc/rvc/layers/synthesizers.py`
- `backend/rvc/rvc/layers/transforms.py`
- `backend/rvc/rvc/layers/utils.py`

### backend/rvc/rvc/onnx/
- `backend/rvc/rvc/onnx/__init__.py`
- `backend/rvc/rvc/onnx/exporter.py`
- `backend/rvc/rvc/onnx/infer.py`
- `backend/rvc/rvc/onnx/synthesizer.py`

### backend/tests/
- `backend/tests/__init__.py`
- `backend/tests/conftest.py`
- `backend/tests/test_devices.py`
- `backend/tests/test_profiles.py`
- `backend/tests/test_realtime_integration.py`
- `backend/tests/test_realtime.py`
- `backend/tests/test_training_integration.py`
- `backend/tests/test_training.py`

### data/samples/07d44be257194f168506323e6c4b4d16/
- `data/samples/07d44be257194f168506323e6c4b4d16/v.wav`

### data/samples/63f32d4d908f4207b98101feabd631dc/
- `data/samples/63f32d4d908f4207b98101feabd631dc/v.wav`

### data/samples/702f85efcb484fbd9e62dfd028c13c19/
- `data/samples/702f85efcb484fbd9e62dfd028c13c19/voice.wav`

### data/samples/752becfa3a0f40a1a6b0835f86ef2b03/
- `data/samples/752becfa3a0f40a1a6b0835f86ef2b03/voice.wav`

### data/samples/7ad0b332195a478aa7829d87d9d0b579/
- `data/samples/7ad0b332195a478aa7829d87d9d0b579/v.wav`

### data/samples/94747c442d524d7a944cac4f19c002e4/
- `data/samples/94747c442d524d7a944cac4f19c002e4/voice.wav`

### data/samples/a607e85468e7443ba613a72edb497837/
- `data/samples/a607e85468e7443ba613a72edb497837/v.wav`

### data/samples/c56298aa46464d0cbd4ebce0c65f97ec/
- `data/samples/c56298aa46464d0cbd4ebce0c65f97ec/v.wav`

### data/samples/fa1a3b24795c40f2a8fc5b3e141782f5/
- `data/samples/fa1a3b24795c40f2a8fc5b3e141782f5/v.wav`

### docs/
- `docs/AMPHION-VEVO2-ANALYSIS.md`
- `docs/BEATRICE-ANALYSIS.md`
- `docs/BEATRICE2_INTEGRATION.md`
- `docs/beatrice2-augmentation-precompute.md`
- `docs/beatrice2-realtime-pipeline.md`
- `docs/CROSS_PLATFORM_ANALYSIS.md`
- `docs/DEITERIS-VC-ANALYSIS.md`
- `docs/KNOWLEDGE.md`
- `docs/REALTIME_CLIENT_IO_MIGRATION.md`
- `docs/RT-VC-ANALYSIS.md`
- `docs/TRAINING_COMPARISON.md`
- `docs/TRAINING_PIPELINE_AUDIT.md`
- `docs/TRAINING_PIPELINE.md`
- `docs/TRAINING_STRATEGY.md`
- `docs/VOCODER-COMPARISON.md`
- `docs/WAY-FORWARD.md`

### frontend/
- `frontend/.gitignore`
- `frontend/next.config.ts`
- `frontend/package.json`
- `frontend/pnpm-lock.yaml`
- `frontend/pnpm-workspace.yaml`
- `frontend/postcss.config.mjs`
- `frontend/README.md`
- `frontend/tsconfig.json`

### frontend/app/
- `frontend/app/globals.css`
- `frontend/app/layout.tsx`
- `frontend/app/NavBar.tsx`
- `frontend/app/page.tsx`
- `frontend/app/ProfilePicker.tsx`
- `frontend/app/SettingsGuide.tsx`
- `frontend/app/TipsPanel.tsx`

### frontend/app/analysis/
- `frontend/app/analysis/page.tsx`

### frontend/app/offline/
- `frontend/app/offline/page.tsx`

### frontend/app/realtime/
- `frontend/app/realtime/page.tsx`

### frontend/app/setup/
- `frontend/app/setup/page.tsx`

### frontend/app/training/
- `frontend/app/training/page.tsx`

### frontend/public/
- `frontend/public/waveform-worker.js`

### logs/mute/0_gt_wavs/
- `logs/mute/0_gt_wavs/mute32k.wav`
- `logs/mute/0_gt_wavs/mute40k.wav`
- `logs/mute/0_gt_wavs/mute48k.spec.pt`
- `logs/mute/0_gt_wavs/mute48k.wav`

### logs/mute/1_16k_wavs/
- `logs/mute/1_16k_wavs/mute.wav`

### logs/mute/2a_f0/
- `logs/mute/2a_f0/mute.wav.npy`

### logs/mute/2b-f0nsf/
- `logs/mute/2b-f0nsf/mute.wav.npy`

### logs/mute/3_feature256/
- `logs/mute/3_feature256/mute.npy`

### logs/mute/3_feature768/
- `logs/mute/3_feature768/mute.npy`

### original_beatrice/
- `original_beatrice/__init__.py`
- `original_beatrice/__main__.py`
- `original_beatrice/convert.py`
