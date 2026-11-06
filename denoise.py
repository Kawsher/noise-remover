#!/usr/bin/env python3
"""Remove background noise (hiss, fan, room tone, hum rumble) from an audio file.

Usage:
    python denoise.py input.mp3
    python denoise.py input.m4a -o clean.flac --reduction 24 --highpass 70
    python denoise.py input.wav --noise-sample 0:1.5     # seconds that hold only noise

How it works: the audio is converted to a time-frequency representation, the
noise level is estimated in each frequency band from the quietest moments, and
each band is turned down according to how far it sits above that noise level.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import istft, stft

CHUNK_SECONDS = 60.0  # long files are processed in pieces to bound memory use
OVERLAP_SECONDS = 2.0
NOISE_WINDOW_SECONDS = 6.0  # how far around each moment to look for quiet passages
NOISE_BLOCK_SECONDS = 0.15
NOISE_PERCENTILE = 15
# A low percentile of block power reads below true mean noise power. This factor
# undoes that; it was measured on Gaussian noise (see tests/test_denoise.py).
NOISE_BIAS_CORRECTION = 1.45
SMOOTHING = 0.98  # decision-directed smoothing; higher means less "musical" noise


@dataclass
class Settings:
    reduction_db: float = 18.0  # how far noise is turned down
    sensitivity: float = 1.3  # >1 treats more of the signal as noise
    highpass_hz: float = 0.0  # 0 disables the rumble filter
    noise_sample: tuple[float, float] | None = None  # (start, end) in seconds


# ---------------------------------------------------------------- audio file I/O

def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any format ffmpeg understands to float32 of shape (samples, channels)."""
    path = str(Path(path).resolve())
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", path],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    if probe.returncode != 0 or not streams:
        raise ValueError("This file has no readable audio. Try an MP3, WAV, M4A or FLAC file.")
    rate, channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])

    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-f", "f32le",
         "-acodec", "pcm_f32le", "-ar", str(rate), "-ac", str(channels), "-"],
        capture_output=True,
    )
    if decoded.returncode != 0:
        raise ValueError("This file looks damaged and could not be decoded.")
    audio = np.frombuffer(decoded.stdout, dtype=np.float32).reshape(-1, channels)
    if audio.size == 0:
        raise ValueError("This file contains no audio samples.")
    return audio, rate


def write_audio(path: str | Path, audio: np.ndarray, rate: int) -> None:
    """Encode to the format implied by the file extension."""
    path = Path(path).resolve()
    codec_args = {
        ".wav": ["-c:a", "pcm_s24le"],
        ".flac": ["-c:a", "flac"],
        ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        ".m4a": ["-c:a", "aac", "-b:a", "192k"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
    }.get(path.suffix.lower())
    if codec_args is None:
        raise ValueError(f"Unsupported output format: {path.suffix} (use wav, flac, mp3, m4a or ogg)")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate),
         "-ac", str(audio.shape[1]), "-i", "-", *codec_args, str(path)],
        input=audio.astype(np.float32).tobytes(), capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not write {path}: {result.stderr.decode(errors='replace')}")


# ---------------------------------------------------------------- noise estimation

def fft_params(rate: int) -> tuple[int, int]:
    """About a 43 ms window at 75% overlap, whatever the sample rate."""
    n_fft = int(2 ** np.round(np.log2(rate * 0.043)))
    return n_fft, n_fft // 4


def adaptive_noise(power: np.ndarray, frames_per_second: float) -> np.ndarray:
    """Track each band's noise floor over time; returns the same shape as power."""
    n_bins, n_frames = power.shape
    block = max(1, round(NOISE_BLOCK_SECONDS * frames_per_second))
    n_blocks = n_frames // block
    if n_blocks < 3:  # clip too short to tell quiet from loud
        return np.repeat(power.mean(axis=1, keepdims=True), n_frames, axis=1)

    blocks = power[:, : n_blocks * block].reshape(n_bins, n_blocks, block).mean(axis=2)
    window = min(n_blocks, max(3, round(NOISE_WINDOW_SECONDS * frames_per_second / block)))
    floor = percentile_filter(np.log(blocks + 1e-20), NOISE_PERCENTILE,
                              size=(1, window), mode="reflect")
    floor = uniform_filter1d(floor, size=max(1, window // 4), axis=1, mode="nearest")
    floor = np.exp(floor) * NOISE_BIAS_CORRECTION

    # Interpolate from block centres back to individual frames.
    position = np.clip((np.arange(n_frames) - block / 2) / block, 0, n_blocks - 1)
    left = np.floor(position).astype(int)
    right = np.minimum(left + 1, n_blocks - 1)
    weight = (position - left).astype(np.float32)
    return floor[:, left] * (1 - weight) + floor[:, right] * weight


def noise_from_sample(sample: np.ndarray, rate: int) -> np.ndarray:
    """Mean noise power per band from a passage containing only noise."""
    n_fft, hop = fft_params(rate)
    _, _, spec = stft(sample, nperseg=n_fft, noverlap=n_fft - hop, axis=0)
    return (np.abs(spec) ** 2).mean(axis=(1, 2))[:, None]


# ---------------------------------------------------------------- suppression

def suppression_gain(power: np.ndarray, noise: np.ndarray, floor: float) -> np.ndarray:
    """Per-band gain from a Wiener filter with decision-directed SNR smoothing."""
    gamma = power / (noise + 1e-20)
    gains = np.empty_like(power)
    previous = np.zeros(power.shape[0], dtype=power.dtype)  # last frame's clean/noise ratio
    for t in range(power.shape[1]):
        xi = SMOOTHING * previous + (1 - SMOOTHING) * np.maximum(gamma[:, t] - 1, 0)
        gain = xi / (1 + xi)
        gains[:, t] = gain
        previous = gain ** 2 * gamma[:, t]
    return np.maximum(gains, floor)


def process_chunk(audio: np.ndarray, rate: int, settings: Settings,
                  fixed_noise: np.ndarray | None) -> np.ndarray:
    n_fft, hop = fft_params(rate)
    length = len(audio)
    if length < n_fft:  # clips shorter than one analysis window
        audio = np.pad(audio, ((0, n_fft - length), (0, 0)))
    freqs, _, spec = stft(audio, fs=rate, nperseg=n_fft, noverlap=n_fft - hop, axis=0)
    spec = spec.astype(np.complex64)  # (bins, channels, frames)

    # One gain for all channels, so the stereo image doesn't wander.
    power = (np.abs(spec) ** 2).mean(axis=1)
    noise = fixed_noise if fixed_noise is not None else adaptive_noise(power, rate / hop)
    gain = suppression_gain(power, noise * settings.sensitivity, 10 ** (-settings.reduction_db / 20))

    if settings.highpass_hz > 0:  # zero-phase 4th-order Butterworth magnitude response
        ratio = (freqs / settings.highpass_hz) ** 8
        gain = gain * np.sqrt(ratio / (1 + ratio))[:, None].astype(gain.dtype)

    spec *= gain[:, None, :]
    _, out = istft(spec, nperseg=n_fft, noverlap=n_fft - hop, time_axis=2, freq_axis=0)
    return out.T[:length].astype(np.float32)  # istft gives (channels, samples)


def denoise(audio: np.ndarray, rate: int, settings: Settings | None = None,
            progress=None) -> np.ndarray:
    """Return a cleaned copy of audio, with the same shape and time alignment."""
    settings = settings or Settings()
    fixed_noise = None
    if settings.noise_sample:
        start, end = (round(t * rate) for t in settings.noise_sample)
        sample = audio[max(0, start):min(end, len(audio))]
        if len(sample) < fft_params(rate)[0] * 2:
            raise ValueError("The noise sample is too short; select at least 0.2 seconds.")
        fixed_noise = noise_from_sample(sample, rate).astype(np.float32)

    chunk = round(CHUNK_SECONDS * rate)
    overlap = round(OVERLAP_SECONDS * rate)
    if len(audio) <= chunk + overlap:
        result = process_chunk(audio, rate, settings, fixed_noise)
    else:
        result = np.zeros_like(audio)
        fade = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
        starts = range(0, len(audio) - overlap, chunk)
        for i, start in enumerate(starts):
            end = min(start + chunk + overlap, len(audio))
            piece = process_chunk(audio[start:end], rate, settings, fixed_noise)
            if start > 0:
                piece[:overlap] *= fade
            if end < len(audio):
                piece[-overlap:] *= fade[::-1]
            result[start:end] += piece
            if progress:
                progress((i + 1) / len(starts))

    # Filtering can push peaks slightly up; only ever scale down, to avoid clipping.
    peak = float(np.abs(result).max())
    return result * (0.999 / peak) if peak > 0.999 else result


# ---------------------------------------------------------------- command line

def parse_noise_sample(text: str) -> tuple[float, float]:
    try:
        start, end = (float(part) for part in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("use START:END in seconds, for example 0:1.5")
    if end <= start or start < 0:
        raise argparse.ArgumentTypeError("END must be after START, and START at least 0")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove background noise from an audio file.")
    parser.add_argument("input", help="audio or video file (anything ffmpeg can read)")
    parser.add_argument("-o", "--output", help="output file (default: <input>_clean.wav)")
    parser.add_argument("-r", "--reduction", type=float, default=18.0,
                        help="how many dB to turn noise down (default 18; try 10-30)")
    parser.add_argument("-s", "--sensitivity", type=float, default=1.3,
                        help="higher removes more but risks thinning the voice (default 1.3)")
    parser.add_argument("--highpass", type=float, default=0.0, metavar="HZ",
                        help="also cut rumble below this frequency (try 70 for voice)")
    parser.add_argument("--noise-sample", type=parse_noise_sample, metavar="START:END",
                        help="seconds containing only noise, to use as the noise profile")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite the output file")
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output) if args.output else source.with_name(f"{source.stem}_clean.wav")
    if target.exists() and target.resolve() == source.resolve():
        parser.error("the output would overwrite the input file")
    if target.exists() and not args.force:
        parser.error(f"{target} already exists (use --force to overwrite)")

    settings = Settings(args.reduction, args.sensitivity, args.highpass, args.noise_sample)
    try:
        audio, rate = read_audio(source)
        print(f"Read {len(audio) / rate:.1f}s at {rate} Hz, {audio.shape[1]} channel(s)")
        cleaned = denoise(audio, rate, settings,
                          progress=lambda f: print(f"\r{f:4.0%}", end="", flush=True))
        write_audio(target, cleaned, rate)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"\nSaved {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
