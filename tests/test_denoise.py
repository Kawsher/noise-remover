"""Run with: .venv/bin/python tests/test_denoise.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import denoise as d

RATE = 44100
rng = np.random.default_rng(7)


def db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def voice_like(seconds, rate=RATE):
    """Harmonic bursts with a moving pitch, separated by pauses, roughly like speech."""
    t = np.arange(int(seconds * rate)) / rate
    pitch = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    tone = sum(np.sin(k * phase) / k for k in range(1, 16))
    syllables = np.clip(np.sin(2 * np.pi * 1.3 * t) * 3, 0, 1)
    return (0.15 * tone * syllables).astype(np.float32)


def noise_like(n, level_db=-32):
    """Pink noise over the audible band at a given level."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    # Without this, about half the energy would be sub-20 Hz drift nobody can hear.
    spectrum[: int(50 * n / RATE)] = 0
    noise = np.fft.irfft(spectrum, n)
    noise *= 10 ** (level_db / 20) / np.sqrt(np.mean(noise ** 2))
    return noise.astype(np.float32)


def test_noise_estimate_is_calibrated():
    n_fft, hop = d.fft_params(RATE)
    for rate in (RATE,):
        white = rng.standard_normal((rate * 30, 1)).astype(np.float32)
        _, _, spec = stft(white, nperseg=n_fft, noverlap=n_fft - hop, axis=0)
        power = (np.abs(spec) ** 2).mean(axis=1)
        ratio = float(np.median(d.adaptive_noise(power, rate / hop) / power.mean(axis=1, keepdims=True)))
        print(f"noise estimate / true noise power = {ratio:.3f}")
        assert 0.9 < ratio < 1.1, ratio


def test_improves_snr_and_keeps_voice():
    voice = voice_like(20)
    noise = noise_like(len(voice))
    noisy = (voice + noise)[:, None]
    cleaned = d.denoise(noisy, RATE)[:, 0]
    assert cleaned.shape == voice.shape
    before = db(voice) - db(noisy[:, 0] - voice)
    after = db(voice) - db(cleaned - voice)
    # Real pauses only: drop zero crossings, and stay 50 ms clear of the voice.
    pauses = binary_erosion(np.abs(voice) < 1e-6, np.ones(int(0.12 * RATE)))
    assert pauses.mean() > 0.2
    drop = db(noisy[:, 0][pauses]) - db(cleaned[pauses])
    # Project the output onto the voice: how much of the voice survived.
    kept = 20 * np.log10(np.dot(cleaned, voice) / np.dot(voice, voice))
    print(f"SNR {before:.1f} -> {after:.1f} dB, noise in pauses down {drop:.1f} dB, voice level {kept:+.2f} dB")
    assert after > before + 5
    assert drop > 15
    assert kept > -1.0


def test_highpass_cuts_rumble_and_keeps_voice():
    # Regression: frequencies were once read as fractions of the sample rate,
    # which made the filter mute everything.
    t = np.arange(RATE * 8) / RATE
    rumble = (0.2 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    voice = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    # Noise reduction is effectively off here, so only the filter acts on these steady tones.
    settings = d.Settings(highpass_hz=70, reduction_db=0.001, sensitivity=0.5)
    level = lambda x: db(d.denoise(x[:, None], RATE, settings)[RATE:-RATE, 0]) - db(x)
    rumble_change, voice_change = level(rumble), level(voice)
    print(f"highpass 70 Hz: 30 Hz rumble {rumble_change:.1f} dB, 300 Hz voice {voice_change:.2f} dB")
    assert rumble_change < -20
    assert voice_change > -0.5


def test_noise_sample_mode():
    voice = voice_like(10)
    voice[: RATE * 2] = 0
    noisy = (voice + noise_like(len(voice)))[:, None]
    cleaned = d.denoise(noisy, RATE, d.Settings(noise_sample=(0, 2)))[:, 0]
    drop = db(noisy[: RATE * 2, 0]) - db(cleaned[: RATE * 2])
    print(f"noise-sample mode: noise down {drop:.1f} dB")
    assert drop > 14


def test_long_file_stereo_chunk_seams():
    # Longer than one chunk, so the crossfaded join is exercised.
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(RATE * 130) / RATE).astype(np.float32)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    stereo += 1e-4 * rng.standard_normal(stereo.shape).astype(np.float32)
    # A steady tone needs a fixed noise profile, or it would be taken for noise itself.
    noise_only = 1e-4 * rng.standard_normal((RATE, 2)).astype(np.float32)
    stereo[:RATE] = noise_only
    cleaned = d.denoise(stereo, RATE, d.Settings(noise_sample=(0, 1)))
    assert cleaned.shape == stereo.shape
    assert np.isfinite(cleaned).all()
    # Level must hold steady across the joins at 60 and 120 seconds.
    block = RATE // 10
    usable = cleaned[RATE * 2 : len(cleaned) - RATE, 0]
    levels = usable[: len(usable) // block * block].reshape(-1, block).std(axis=1)
    swing = 20 * np.log10(levels.max() / levels.min())
    balance = 20 * np.log10(cleaned[RATE * 2 :, 0].std() / cleaned[RATE * 2 :, 1].std())
    print(f"level swing across joins {swing:.3f} dB, L/R balance {balance:.2f} dB (expected 6.02)")
    assert swing < 0.1
    assert abs(balance - 6.02) < 0.1


def test_edge_cases():
    silence = np.zeros((RATE, 1), dtype=np.float32)
    assert np.allclose(d.denoise(silence, RATE), 0)
    tiny = rng.standard_normal((500, 2)).astype(np.float32) * 0.01  # shorter than a window
    assert d.denoise(tiny, RATE).shape == tiny.shape
    loud = np.clip(rng.standard_normal((RATE, 1)), -1, 1).astype(np.float32)
    assert np.abs(d.denoise(loud, RATE)).max() <= 1.0


def test_file_round_trip():
    voice = voice_like(3)[:, None]
    with tempfile.TemporaryDirectory() as folder:
        for ext in ("wav", "flac", "mp3", "m4a", "ogg"):
            path = Path(folder) / f"-odd name.{ext}"
            d.write_audio(path, voice, RATE)
            back, rate = d.read_audio(path)
            assert rate == RATE and back.shape[1] == 1 and abs(len(back) - len(voice)) < 4096, ext
        bad = Path(folder) / "bad.wav"
        bad.write_text("not audio")
        try:
            d.read_audio(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on a non-audio file")
        # CLI refuses to overwrite the source or an existing output without --force.
        src = Path(folder) / "in.wav"
        d.write_audio(src, voice, RATE)
        script = str(Path(d.__file__).resolve())
        assert subprocess.run([sys.executable, script, str(src), "-o", str(src), "-f"],
                              capture_output=True).returncode != 0
        assert subprocess.run([sys.executable, script, str(src)], capture_output=True).returncode == 0
        assert (Path(folder) / "in_clean.wav").exists()
        assert subprocess.run([sys.executable, script, str(src)], capture_output=True).returncode != 0


if __name__ == "__main__":
    failed = 0
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            try:
                test()
                print(f"PASS {name}")
            except Exception as error:
                failed += 1
                print(f"FAIL {name}: {error!r}")
    sys.exit(1 if failed else 0)
