# Noise Remover

Removes steady background noise (hiss, fan, air conditioning, room tone) from audio.
It runs entirely on your computer and needs Python 3 and ffmpeg.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Browser interface

```bash
.venv/bin/python app.py
```

Open http://127.0.0.1:8765, choose a file, then compare Original, Cleaned and Removed noise.

## Command line

```bash
.venv/bin/python denoise.py recording.mp3
```

This writes `recording_clean.wav`. Useful options:

| Option | Effect |
| --- | --- |
| `-o clean.flac` | Output file; the extension sets the format (wav, flac, mp3, m4a, ogg) |
| `-r 24` | dB to turn noise down (default 18; try 10–30) |
| `-s 1.8` | Sensitivity; higher removes more but can thin the voice (default 1.3) |
| `--highpass 70` | Also cut rumble below 70 Hz (for voice) |
| `--noise-sample 0:1.5` | Learn the noise from these seconds, which must contain only noise |

## Limits

- It handles steady noise best. Sudden sounds such as a dog bark, keyboard click or
  background conversation mostly stay, because they don't look like a steady noise floor.
- Long sustained notes in music can be mistaken for noise. For music, use `--noise-sample`.
- A video file gives you a cleaned audio file; the video itself isn't re-exported.

## Tests

```bash
.venv/bin/python tests/test_denoise.py
```
