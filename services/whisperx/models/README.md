## WhisperX Local Models

This directory stores local model artifacts used by the WhisperX service.

- `speaker_role/distilbert/` is loaded by `speaker_role_classifier.py`.
- Populate it once from `speaker_classifier_export.zip` using:
  - `make prepare-speaker-model`
  - or `python infra/scripts/prepare_speaker_role_model.py --delete-zip`

Keep large model binaries out of git unless explicitly required.
