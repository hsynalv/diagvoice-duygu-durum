#!/usr/bin/env python3
"""
DiagVoice benchmark inference — Flask HTTP katmanı.
Motor: inference_api.engine

Çalıştırma (Docker içi örnek):
  DIAGVOICE_BENCHMARK_CONFIG=/app/inference_api/config.docker.yaml \\
  python -m flask --app inference_api.app run --host 127.0.0.1 --port 5008
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from inference_api import engine

app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    try:
        info = engine.health_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/predict")
def predict() -> Any:
    if "audio" not in request.files or not request.files["audio"].filename:
        return jsonify({"error": "multipart alan 'audio' (ses dosyası) gerekli"}), 400
    f = request.files["audio"]
    transcript = (request.form.get("transcript") or "").strip()

    ext = Path(f.filename or "x.wav").suffix.lower() or ".wav"
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        return jsonify({"error": f"desteklenmeyen uzantı: {ext}"}), 400

    try:
        _, bundle = engine.load_bundle()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        try:
            body = engine.predict_from_uploaded_file(tmp_path, transcript, ext)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except NotImplementedError as e:
        return jsonify({"error": str(e), "mode": bundle.get("mode")}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(body)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true"))
