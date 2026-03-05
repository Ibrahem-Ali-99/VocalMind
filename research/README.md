# 📓 Research Experiments

This directory contains Jupyter notebooks and experimental scripts from VocalMind's research phase. These are **not production code** — they are reference implementations and exploratory work.

## Experiments

| Directory          | Description                               | Key Files                                             |
| :----------------- | :---------------------------------------- | :---------------------------------------------------- |
| `asr/`             | Automatic Speech Recognition (WhisperX)   | `automatic-speech-recognition.ipynb`                  |
| `diarization/`     | Speaker Diarization (pyannote, NeMo)      | `speaker diarization.ipynb`                           |
| `emotion/`         | Emotion Recognition from speech & text    | 4 notebooks comparing different approaches            |
| `voice-gen/`       | Voice synthesis / TTS experiments         | `voice_generation.py`, `voice_generation_overlap.py`  |
| `speech-pipeline/` | Full ASR → Diarization → Emotion pipeline | `pipeline_experiment.py` (consolidated final version) |

## Running Notebooks

```bash
cd research
pip install jupyterlab
jupyter lab
```

> **Note:** Some notebooks require GPU access and specific model downloads. Check each notebook's setup cells for requirements.
