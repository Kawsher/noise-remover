#!/usr/bin/env python3
"""Local web interface for the noise remover. Run it, then open http://127.0.0.1:8765"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import denoise as engine

PORT = 8765
INDEX = Path(__file__).with_name("index.html")
WORK_DIR = Path(tempfile.mkdtemp(prefix="noise-remover-"))
OUTPUT_FORMATS = {"wav", "flac", "mp3", "m4a"}
MIME = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg", "m4a": "audio/mp4"}
jobs: dict[str, dict[str, Path]] = {}  # job id -> track name -> file


def clean_upload(data: bytes, filename: str, query: dict) -> dict:
    job = uuid.uuid4().hex
    folder = WORK_DIR / job
    folder.mkdir()
    suffix = re.sub(r"[^a-zA-Z0-9.]", "", Path(filename).suffix)[:10]
    source = folder / f"upload{suffix}"
    source.write_bytes(data)

    def number(name: str, default: float, low: float, high: float) -> float:
        return min(high, max(low, float(query.get(name, [default])[0])))

    noise_sample = None
    if query.get("noise_start") and query.get("noise_end"):
        noise_sample = (number("noise_start", 0, 0, 1e7), number("noise_end", 0, 0, 1e7))
        if noise_sample[1] <= noise_sample[0]:
            raise ValueError("The noise sample's end must be after its start.")
    settings = engine.Settings(
        reduction_db=number("reduction", 18, 3, 60),
        sensitivity=number("sensitivity", 1.3, 0.5, 4),
        highpass_hz=number("highpass", 0, 0, 300),
        noise_sample=noise_sample,
    )
    output_format = query.get("format", ["wav"])[0]
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")

    audio, rate = engine.read_audio(source)
    cleaned = engine.denoise(audio, rate, settings)
    tracks = {
        "original": folder / "original.wav",
        "cleaned": folder / "cleaned.wav",
        "removed": folder / "removed.wav",
        "download": folder / f"download.{output_format}",
    }
    engine.write_audio(tracks["original"], audio, rate)
    engine.write_audio(tracks["cleaned"], cleaned, rate)
    engine.write_audio(tracks["removed"], audio - cleaned, rate)
    engine.write_audio(tracks["download"], cleaned, rate)
    source.unlink()
    jobs[job] = tracks
    return {"job": job, "format": output_format, "duration": len(audio) / rate,
            "rate": rate, "channels": audio.shape[1]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def allowed(self) -> bool:
        # Only answer requests addressed to this machine, so other websites
        # open in the browser can't reach the tool through DNS tricks.
        if self.headers.get("Host") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return True
        self.send_error(403)
        return False

    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return
        url = urlparse(self.path)
        if url.path == "/":
            return self.reply(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        match = re.fullmatch(r"/audio/([0-9a-f]{32})/(\w+)", url.path)
        track = jobs.get(match[1], {}).get(match[2]) if match else None
        if track is None:
            return self.send_error(404)
        self.reply(200, track.read_bytes(), MIME[track.suffix[1:]])

    def do_POST(self):
        if not self.allowed():
            return
        url = urlparse(self.path)
        if url.path != "/clean":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            filename = unquote(self.headers.get("X-Filename", "audio"))
            result = clean_upload(self.rfile.read(length), filename, parse_qs(url.query))
            self.reply(200, json.dumps(result).encode(), "application/json")
        except ValueError as error:
            self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Noise remover running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
