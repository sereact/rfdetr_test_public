#!/usr/bin/env python3
"""Send one prompt (plus optional images) to Gemini and print the answer.

Fill in the INPUT block below, then run:
    python3 scripts/inference/gemini_request.py

The images go first, in the listed order, then the prompt text. The API key is
read from the GOOGLE_API_KEY environment variable. Standard library only.
"""
# uv run python -c "from gemini_request import ask; print(ask('Say hello', [], 'gemini-robotics-er-2-preview'))"

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# ================================ INPUT =================================
PROMPT = """What do you see in this image."""

IMAGES = [                      # image files (.jpg / .png / .webp); [] for text only
    "path/to/image.jpg",
]

MODEL = "gemini-robotics-er-2-preview"      # e.g. "gemini-3.7-flash"
THINKING = None                 # None (model default), "minimal", "low", "medium" or "high"
# ========================================================================

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def ask(prompt: str, images: list[str], model: str, thinking: str | None = None,
        retries: int = 4) -> str:
    parts = [{"inline_data": {"mime_type": MIME[Path(p).suffix.lower()],
                              "data": base64.b64encode(Path(p).read_bytes()).decode()}}
             for p in images]
    parts.append({"text": prompt})
    config = {"temperature": 0.0}
    if thinking:
        config["thinkingConfig"] = {"thinkingLevel": thinking}
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": config}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GOOGLE_API_KEY"]}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=180) as r:
                out = json.load(r)
            return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"]
                           if not p.get("thought"))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429 or attempt == retries - 1:
                raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()[:500]}")
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise SystemExit(f"request failed: {exc}")
        time.sleep(5 * (attempt + 1))
    raise SystemExit("request failed")


if __name__ == "__main__":
    print(ask(PROMPT, IMAGES, MODEL, THINKING))
