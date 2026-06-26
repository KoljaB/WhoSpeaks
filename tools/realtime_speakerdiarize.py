"""Realtime YouTube WASAPI transcription with embedding-only speaker labels.

The browser plays a YouTube iframe, RealtimeSTT captures system audio through
WASAPI loopback, and this script assigns each completed sentence to a stable
incremental speaker memory using one voice-embedding provider.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000
INT16_MAX_ABS_VALUE = 32767.0
DEFAULT_EMBEDDING_PROVIDER = "speechbrain_ecapa"
DEFAULT_SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealtimeSTT Speaker Diarization</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07090d;
      --panel: #101620;
      --panel-2: #151b25;
      --panel-3: #0d131c;
      --line: #263242;
      --text: #f4f7fb;
      --muted: #93a0b4;
      --green: #39e58e;
      --cyan: #37d5f7;
      --blue: #65a8ff;
      --yellow: #f2d35f;
      --pink: #ff79a6;
      --red: #ff5b70;
      --orange: #f59a43;
      --violet: #b48cff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }

    .app {
      height: 100vh;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(240px, 36vh) minmax(0, 1fr);
      gap: 14px;
      padding: 18px;
      overflow: hidden;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }

    input, button {
      border: 1px solid var(--line);
      border-radius: 7px;
      font: inherit;
      min-height: 44px;
    }

    input {
      width: 100%;
      background: #0b1018;
      color: var(--text);
      padding: 0 14px;
      outline: none;
    }

    input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 2px rgba(55, 213, 247, 0.16);
    }

    button {
      background: var(--green);
      color: #03120a;
      padding: 0 18px;
      font-weight: 400;
      cursor: pointer;
    }

    button.secondary {
      background: #1c2636;
      color: var(--text);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .main {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(260px, 340px);
      gap: 14px;
      min-height: 0;
    }

    .video, .status, .text-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .video {
      background: #000;
      min-height: 280px;
      position: relative;
    }

    #player, .placeholder {
      width: 100%;
      height: 100%;
      min-height: 280px;
    }

    .placeholder {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 18px;
    }

    .status, .text-panel {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      background: var(--panel);
    }

    .text-panel {
      background: var(--panel-2);
    }

    h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--cyan);
      font-size: 14px;
      font-weight: 400;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    #statusLog {
      margin: 0;
      padding: 12px 14px;
      color: #d5e0f5;
      overflow: auto;
      white-space: pre-wrap;
      font: 13px/1.45 Consolas, "Cascadia Mono", monospace;
    }

    .detected-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      min-height: 0;
    }

    #sentenceList {
      padding: 12px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: var(--panel-3);
    }

    .sentence-item {
      border: 1px solid #2b3849;
      border-radius: 8px;
      padding: 10px 11px;
      background: #111923;
    }

    .sentence-item.pending {
      opacity: 0.72;
    }

    .sentence-top {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      margin-bottom: 7px;
    }

    .speaker-badge {
      flex: 0 0 auto;
      min-width: 42px;
      padding: 3px 7px;
      border-radius: 6px;
      background: #233146;
      color: #fff;
      text-align: center;
      font-size: 12px;
      line-height: 1.35;
    }

    .speaker-new {
      background: #6b4d12;
      color: #fff2c0;
    }

    .speaker-unknown {
      background: #343a46;
      color: #d8dfec;
    }

    .sentence-meta {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .sentence-text {
      color: #f3f6fb;
      font-size: 15px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .probabilities {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .prob-label {
      border: 1px solid #354256;
      border-radius: 6px;
      color: #dce5f5;
      background: #0d1420;
      padding: 2px 6px;
      font-size: 12px;
      line-height: 1.35;
    }

    .prob-label.unknown {
      color: #f3d18c;
      border-color: #654b20;
      background: #21180a;
    }

    #transcript {
      padding: 18px;
      overflow: auto;
      font-size: 24px;
      line-height: 1.35;
      white-space: pre-wrap;
      background: var(--panel-2);
    }

    .final-speaker-1 { color: var(--blue); }
    .final-speaker-2 { color: var(--yellow); }
    .final-speaker-3 { color: var(--pink); }
    .final-speaker-4 { color: var(--green); }
    .final-speaker-5 { color: var(--orange); }
    .final-speaker-6 { color: var(--violet); }
    .final-unknown { color: #a9b3c4; }
    .stable { color: #ffffff; }
    .realtime { color: #8a93a5; }
    .error { color: var(--red); }

    @media (max-width: 900px) {
      body {
        height: auto;
        min-height: 100vh;
        overflow: auto;
      }

      .app {
        height: auto;
        min-height: 100vh;
        overflow: visible;
        grid-template-rows: auto minmax(360px, auto) minmax(520px, 1fr);
      }

      .toolbar, .main, .detected-grid {
        grid-template-columns: 1fr;
      }

      #transcript {
        font-size: 21px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <input id="urlInput" type="url" autocomplete="off" spellcheck="false" placeholder="https://www.youtube.com/watch?v=...">
      <button id="startButton">Start</button>
      <button id="stopButton" class="secondary">Stop</button>
    </div>

    <main class="main">
      <section class="video">
        <div id="player"><div class="placeholder">YouTube video</div></div>
      </section>

      <section class="status">
        <h2>Status</h2>
        <pre id="statusLog"></pre>
      </section>
    </main>

    <section class="detected-grid">
      <section class="text-panel">
        <h2>Detected Sentences</h2>
        <div id="sentenceList"></div>
      </section>
      <section class="text-panel">
        <h2>Detected Text</h2>
        <div id="transcript"></div>
      </section>
    </section>
  </div>

  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    const urlInput = document.getElementById("urlInput");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const statusLog = document.getElementById("statusLog");
    const transcript = document.getElementById("transcript");
    const sentenceList = document.getElementById("sentenceList");

    let player = null;
    let ytReady = false;
    let pendingVideoId = null;
    let activeSessionId = null;
    let timePing = null;
    let finalTexts = [];
    let sentences = [];
    let stableText = "";
    let realtimeText = "";
    let postedFirstPlaying = false;

    function snippet(text, limit = 260) {
      text = text || "";
      return text.length <= limit ? text : `${text.slice(0, limit)}...`;
    }

    function debugLog(stage, payload) {
      const body = JSON.stringify({
        stage,
        session_id: activeSessionId,
        browser_time_ms: performance.now(),
        payload: payload || {}
      });
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([body], {type: "application/json"});
          navigator.sendBeacon("/api/debug-log", blob);
          return;
        }
      } catch (_) {}
      fetch("/api/debug-log", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body,
        keepalive: true
      }).catch(() => {});
    }

    function appendStatus(message, cssClass) {
      const line = document.createElement("div");
      if (cssClass) {
        line.className = cssClass;
      }
      const timestamp = new Date().toLocaleTimeString();
      line.textContent = `[${timestamp}] ${message}`;
      statusLog.appendChild(line);
      statusLog.scrollTop = statusLog.scrollHeight;
    }

    function resetDisplay() {
      statusLog.textContent = "";
      finalTexts = [];
      sentences = [];
      stableText = "";
      realtimeText = "";
      postedFirstPlaying = false;
      stopTimePings();
      renderTranscript();
      renderSentences();
    }

    function speakerClass(speaker) {
      if (!speaker) {
        return "final-unknown";
      }
      const number = Number(String(speaker).replace(/[^\d]/g, "")) || 0;
      if (!number) {
        return "final-unknown";
      }
      return `final-speaker-${((number - 1) % 6) + 1}`;
    }

    function renderTranscript() {
      const fragment = document.createDocumentFragment();
      finalTexts.forEach((item) => {
        const span = document.createElement("span");
        span.className = speakerClass(item.speaker);
        span.textContent = `${item.text.trim()} `;
        fragment.appendChild(span);
      });

      if (stableText) {
        const stable = document.createElement("span");
        stable.className = "stable";
        stable.textContent = stableText;
        fragment.appendChild(stable);
      }

      if (realtimeText) {
        const realtime = document.createElement("span");
        realtime.className = "realtime";
        realtime.textContent = realtimeText;
        fragment.appendChild(realtime);
      }

      transcript.replaceChildren(fragment);
      transcript.scrollTop = transcript.scrollHeight;
      requestAnimationFrame(() => {
        transcript.scrollTop = transcript.scrollHeight;
      });
    }

    function probabilityKeys(probabilities) {
      return Object.keys(probabilities || {}).sort((left, right) => {
        if (left === "unknown") return -1;
        if (right === "unknown") return 1;
        const leftNumber = Number(left.replace(/[^\d]/g, "")) || 0;
        const rightNumber = Number(right.replace(/[^\d]/g, "")) || 0;
        return leftNumber - rightNumber || left.localeCompare(right);
      });
    }

    function formatProb(value) {
      return `${Math.round(Number(value || 0) * 100)}%`;
    }

    function sentenceBadgeText(item) {
      if (item.pending) {
        return "...";
      }
      if (!item.assigned_speaker) {
        return "UNK";
      }
      return item.created_speaker ? `${item.assigned_speaker} new` : item.assigned_speaker;
    }

    function renderSentences() {
      const fragment = document.createDocumentFragment();
      sentences.forEach((item) => {
        const row = document.createElement("article");
        row.className = `sentence-item${item.pending ? " pending" : ""}`;

        const top = document.createElement("div");
        top.className = "sentence-top";

        const badge = document.createElement("div");
        badge.className = "speaker-badge";
        if (!item.assigned_speaker) {
          badge.classList.add("speaker-unknown");
        }
        if (item.created_speaker) {
          badge.classList.add("speaker-new");
        }
        badge.textContent = sentenceBadgeText(item);
        top.appendChild(badge);

        const meta = document.createElement("div");
        meta.className = "sentence-meta";
        if (item.pending) {
          meta.textContent = "embedding pending";
        } else {
          const parts = [];
          if (item.duration_seconds) {
            parts.push(`${Number(item.duration_seconds).toFixed(2)}s`);
          }
          if (item.top_similarity !== null && item.top_similarity !== undefined) {
            parts.push(`sim ${Number(item.top_similarity).toFixed(2)}`);
          }
          if (item.margin !== null && item.margin !== undefined) {
            parts.push(`margin ${Number(item.margin).toFixed(2)}`);
          }
          if (item.assignment_source === "context") {
            parts.push("context");
          }
          meta.textContent = parts.join("  ");
        }
        top.appendChild(meta);
        row.appendChild(top);

        const text = document.createElement("div");
        text.className = "sentence-text";
        text.textContent = item.text || "";
        row.appendChild(text);

        const probs = document.createElement("div");
        probs.className = "probabilities";
        probabilityKeys(item.probabilities).forEach((key) => {
          const label = document.createElement("span");
          label.className = `prob-label${key === "unknown" ? " unknown" : ""}`;
          label.textContent = `${key} ${formatProb(item.probabilities[key])}`;
          probs.appendChild(label);
        });
        row.appendChild(probs);

        fragment.appendChild(row);
      });

      sentenceList.replaceChildren(fragment);
      sentenceList.scrollTop = sentenceList.scrollHeight;
      requestAnimationFrame(() => {
        sentenceList.scrollTop = sentenceList.scrollHeight;
      });
    }

    function upsertFinal(index, text) {
      const existing = finalTexts.find((item) => item.index === index);
      if (existing) {
        existing.text = text;
      } else {
        finalTexts.push({index, text, speaker: null});
      }
      renderTranscript();
    }

    function upsertSentence(data) {
      const existing = sentences.find((item) => item.index === data.index);
      if (existing) {
        Object.assign(existing, data);
      } else {
        sentences.push(data);
      }
      sentences.sort((left, right) => Number(left.index) - Number(right.index));

      const finalItem = finalTexts.find((item) => item.index === data.index);
      if (finalItem && data.assigned_speaker) {
        finalItem.speaker = data.assigned_speaker;
      }
      renderSentences();
      renderTranscript();
    }

    function consumeFinalFromRealtime(finalText) {
      const finalClean = (finalText || "").trim();
      if (!finalClean || (!stableText && !realtimeText)) {
        return;
      }
      const current = `${stableText || ""}${realtimeText || ""}`;
      const leading = current.length - current.trimStart().length;
      const comparable = current.trimStart().toLowerCase();
      if (comparable.startsWith(finalClean.toLowerCase())) {
        stableText = "";
        realtimeText = current.slice(leading + finalClean.length).trimStart();
      }
    }

    function splitRealtime(displayText, stable, unstable) {
      displayText = displayText || "";
      stable = stable || "";
      unstable = unstable || "";
      stableText = stable;
      if (unstable) {
        realtimeText = unstable;
      } else if (stable && displayText.toLowerCase().startsWith(stable.toLowerCase())) {
        realtimeText = displayText.slice(stable.length);
      } else {
        realtimeText = displayText;
      }
      renderTranscript();
      debugLog("realtimeRender", {
        stable: snippet(stableText),
        realtime: snippet(realtimeText)
      });
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function playerCurrentTime() {
      try {
        return player ? player.getCurrentTime() || 0 : 0;
      } catch (_) {
        return 0;
      }
    }

    function sendVideoTime(path) {
      if (!activeSessionId || !player) {
        return;
      }
      postJson(path, {
        session_id: activeSessionId,
        current_time: playerCurrentTime()
      }).catch((error) => appendStatus(error.message, "error"));
    }

    function startTimePings() {
      if (timePing) {
        clearInterval(timePing);
      }
      timePing = setInterval(() => {
        if (player && player.getPlayerState && player.getPlayerState() === YT.PlayerState.PLAYING) {
          sendVideoTime("/api/video-time");
        }
      }, 500);
    }

    function stopTimePings() {
      if (timePing) {
        clearInterval(timePing);
        timePing = null;
      }
    }

    function createPlayer(videoId) {
      pendingVideoId = null;
      if (player && player.destroy) {
        player.destroy();
      }
      player = new YT.Player("player", {
        width: "100%",
        height: "100%",
        videoId,
        playerVars: {
          autoplay: 1,
          controls: 1,
          rel: 0,
          modestbranding: 1,
          playsinline: 1
        },
        events: {
          onReady: (event) => {
            appendStatus("YouTube player ready. Starting playback.");
            event.target.playVideo();
          },
          onStateChange: (event) => {
            if (event.data === YT.PlayerState.PLAYING) {
              if (!postedFirstPlaying) {
                postedFirstPlaying = true;
                appendStatus("YouTube reports PLAYING. WASAPI capture is armed.");
                sendVideoTime("/api/video-playing");
              }
              startTimePings();
            } else if (
              event.data === YT.PlayerState.PAUSED ||
              event.data === YT.PlayerState.ENDED
            ) {
              stopTimePings();
            }
          }
        }
      });
    }

    window.onYouTubeIframeAPIReady = function() {
      ytReady = true;
      if (pendingVideoId) {
        createPlayer(pendingVideoId);
      }
    };

    function loadVideo(videoId) {
      if (!ytReady) {
        pendingVideoId = videoId;
        appendStatus("Waiting for YouTube player API.");
        return;
      }
      createPlayer(videoId);
    }

    const events = new EventSource("/events");
    events.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "");
    });
    events.addEventListener("error-status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "Error", "error");
    });
    events.addEventListener("capture-ready", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus("Capture ready. Loading YouTube video.");
      loadVideo(data.video_id);
    });
    events.addEventListener("realtime", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      splitRealtime(
        data.display_text || data.text || "",
        data.stable_text || "",
        data.unstable_text || ""
      );
    });
    events.addEventListener("final", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      const text = (data.text || "").trim();
      if (text) {
        upsertFinal(data.index, text);
        consumeFinalFromRealtime(text);
        renderTranscript();
      }
    });
    events.addEventListener("sentence", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      upsertSentence(data);
    });

    startButton.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (!url) {
        appendStatus("Paste a YouTube URL first.", "error");
        return;
      }

      startButton.disabled = true;
      resetDisplay();
      appendStatus("Started.");

      try {
        const data = await postJson("/api/start", {url});
        activeSessionId = data.session_id;
      } catch (error) {
        appendStatus(error.message, "error");
      } finally {
        startButton.disabled = false;
      }
    });

    stopButton.addEventListener("click", async () => {
      stopTimePings();
      if (player && player.pauseVideo) {
        player.pauseVideo();
      }
      try {
        await postJson("/api/stop", {session_id: activeSessionId});
        appendStatus("Stop requested.");
      } catch (error) {
        appendStatus(error.message, "error");
      }
    });
  </script>
</body>
</html>
"""


def configure_embedding_env() -> None:
    cache = ROOT / ".cache"
    env_defaults = {
        "HF_HOME": cache / "huggingface",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache,
        "NUMBA_CACHE_DIR": cache / "numba",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def default_embedding_python() -> Path:
    candidate = ROOT / ".venv-voice-embeddings" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def choose_torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def audio_to_float_mono(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)

    if isinstance(value, np.ndarray):
        audio = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) % 4 == 0:
            audio = np.frombuffer(raw, dtype=np.float32)
        elif len(raw) % 2 == 0:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
        else:
            audio = np.empty(0, dtype=np.float32)
    else:
        audio = np.asarray(value)

    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype.kind in {"i", "u"}:
        audio = audio.astype(np.float32) / INT16_MAX_ABS_VALUE
    else:
        audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio.reshape(-1), -1.0, 1.0).astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < int(sample_rate * 0.25):
        return audio

    frame = max(1, int(sample_rate * 0.03))
    hop = max(1, int(sample_rate * 0.01))
    if len(audio) < frame:
        return audio

    rms = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        chunk = audio[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not rms:
        return audio

    peak = max(rms)
    threshold = max(peak * 0.08, 0.003)
    active = [index for index, value in enumerate(rms) if value >= threshold]
    if not active:
        return audio

    pad = int(sample_rate * 0.10)
    start = max(0, active[0] * hop - pad)
    end = min(len(audio), active[-1] * hop + frame + pad)
    return audio[start:end]


def pad_audio(audio: np.ndarray, minimum_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    minimum_samples = int(max(0.0, minimum_seconds) * sample_rate)
    if len(audio) >= minimum_samples:
        return audio
    return np.pad(audio, (0, minimum_samples - len(audio))).astype(np.float32)


@dataclass
class TranscriptPart:
    text: str
    audio: np.ndarray
    start_seconds: float
    end_seconds: float
    word_start_seconds: float
    word_end_seconds: float
    word_count: int
    split_reason: str
    part_index: int
    part_count: int

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def timed_word_text(words: list[dict[str, Any]]) -> str:
    pieces = [str(word.get("word", "")) for word in words if str(word.get("word", "")).strip()]
    if not pieces:
        return ""
    if any(piece.startswith(" ") for piece in pieces):
        text = "".join(pieces)
    else:
        text = " ".join(piece.strip() for piece in pieces)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def normalize_timed_words(
    metadata: dict[str, Any] | None,
    duration_seconds: float,
    max_word_seconds: float | None = None,
) -> list[dict[str, Any]]:
    words = []
    for item in (metadata or {}).get("words") or []:
        text = str(item.get("word", ""))
        if not text.strip():
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, min(duration_seconds, start))
        end = max(0.0, min(duration_seconds, end))
        if end <= start:
            continue
        words.append({"word": text, "start": start, "end": end})
    words.sort(key=lambda word: (word["start"], word["end"]))
    if max_word_seconds is not None and max_word_seconds > 0.0:
        for index, word in enumerate(words):
            start = float(word["start"])
            max_end = min(duration_seconds, start + max_word_seconds)
            if index + 1 < len(words):
                next_start = float(words[index + 1]["start"])
                if next_start > start:
                    max_end = min(max_end, next_start)
            if float(word["end"]) > max_end:
                word["end"] = max_end
    return words


def _word_ends_sentence(word_text: str) -> bool:
    return bool(re.search(r"[.!?]+[\"')\]]*$", word_text.strip()))


def _word_ends_soft_boundary(word_text: str) -> bool:
    return bool(re.search(r"[,;:]+[\"')\]]*$", word_text.strip()))


def split_transcript_by_timestamps(
    text: str,
    audio: np.ndarray,
    sample_rate: int,
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[TranscriptPart]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_seconds = float(len(audio)) / float(sample_rate or SAMPLE_RATE)
    fallback = [
        TranscriptPart(
            text=text.strip(),
            audio=audio,
            start_seconds=0.0,
            end_seconds=duration_seconds,
            word_start_seconds=0.0,
            word_end_seconds=duration_seconds,
            word_count=0,
            split_reason="fallback",
            part_index=0,
            part_count=1,
        )
    ]
    if not text.strip() or not getattr(args, "split_final_transcripts", True):
        return fallback

    words = normalize_timed_words(
        metadata,
        duration_seconds,
        max_word_seconds=float(args.max_word_timestamp_seconds),
    )
    if not words:
        return fallback

    groups: list[tuple[list[dict[str, Any]], str]] = []
    current: list[dict[str, Any]] = []
    max_seconds = max(0.0, float(args.max_timestamp_split_seconds))
    gap_seconds = max(0.0, float(args.word_split_gap_seconds))
    split_on_soft = bool(getattr(args, "split_on_soft_punctuation", False))

    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            if gap_seconds > 0.0 and gap >= gap_seconds:
                groups.append((current, "gap"))
                current = []

        current.append(word)
        if len(current) < max(1, int(args.min_timestamp_split_words)):
            continue

        span = float(current[-1]["end"]) - float(current[0]["start"])
        should_split = False
        reason = ""
        if _word_ends_sentence(str(word.get("word", ""))):
            should_split = True
            reason = "sentence"
        elif split_on_soft and _word_ends_soft_boundary(str(word.get("word", ""))):
            should_split = True
            reason = "punctuation"
        elif max_seconds > 0.0 and span >= max_seconds:
            should_split = True
            reason = "max_duration"

        if should_split:
            groups.append((current, reason))
            current = []

    if current:
        groups.append((current, "final"))

    if not groups:
        return fallback

    padding = max(0.0, float(args.split_audio_padding_seconds))
    parts: list[TranscriptPart] = []
    raw_bounds = [
        (float(group[0][0]["start"]), float(group[0][-1]["end"]))
        for group in groups
    ]
    for index, ((group, reason), (raw_start, raw_end)) in enumerate(zip(groups, raw_bounds)):
        start = max(0.0, raw_start - padding)
        end = min(duration_seconds, raw_end + padding)
        if index > 0:
            previous_end = raw_bounds[index - 1][1]
            boundary = (previous_end + raw_start) / 2.0
            start = max(start, boundary)
        if index + 1 < len(raw_bounds):
            next_start = raw_bounds[index + 1][0]
            boundary = (raw_end + next_start) / 2.0
            end = min(end, boundary)
        start_sample = max(0, min(len(audio), int(start * sample_rate)))
        end_sample = max(start_sample, min(len(audio), int(math.ceil(end * sample_rate))))
        if end_sample <= start_sample:
            continue
        part_text = timed_word_text(group)
        if not part_text:
            continue
        parts.append(
            TranscriptPart(
                text=part_text,
                audio=audio[start_sample:end_sample].astype(np.float32, copy=False),
                start_seconds=round(float(start_sample) / float(sample_rate), 4),
                end_seconds=round(float(end_sample) / float(sample_rate), 4),
                word_start_seconds=round(float(raw_start), 4),
                word_end_seconds=round(float(raw_end), 4),
                word_count=len(group),
                split_reason=reason,
                part_index=index,
                part_count=0,
            )
        )

    if not parts:
        return fallback
    if len(parts) == 1:
        parts[0].text = text.strip()
    for index, part in enumerate(parts):
        part.part_index = index
        part.part_count = len(parts)
    return parts


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX_ABS_VALUE).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def load_audio_file(path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
        except ModuleNotFoundError:
            import torch
            import torchaudio.functional as audio_functional

            tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            audio = audio_functional.resample(
                tensor,
                orig_freq=int(source_rate),
                new_freq=int(sample_rate),
            ).cpu().numpy()
        source_rate = sample_rate
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(audio, dtype=np.float32), source_rate


def normalize_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Embedding provider returned an empty vector.")
    return (vector / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Embedding shape mismatch: {left.shape} vs {right.shape}")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        raise ValueError("Softmax temperature must be greater than zero.")
    max_value = max(values)
    exps = [math.exp((value - max_value) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class SpeechBrainEcapaProvider:
    def __init__(self, device: str, model_id: str = DEFAULT_SPEECHBRAIN_MODEL) -> None:
        configure_embedding_env()
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.device = choose_torch_device(device)
        savedir = ROOT / ".cache" / "speechbrain" / sanitize(model_id)
        self.model = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(savedir),
            run_opts={"device": self.device},
        )

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        waveform = self.torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            embedding = self.model.encode_batch(waveform, normalize=False)
        return normalize_vector(embedding)


def create_embedding_provider(provider: str, device: str) -> SpeechBrainEcapaProvider:
    if provider != DEFAULT_EMBEDDING_PROVIDER:
        raise ValueError(
            f"Unsupported embedding provider {provider!r}. "
            f"This tool currently supports only {DEFAULT_EMBEDDING_PROVIDER!r}."
        )
    return SpeechBrainEcapaProvider(device=device)


@dataclass
class SpeakerProfile:
    index: int
    centroid: np.ndarray
    sentence_count: int
    speech_seconds: float
    created_at: float
    last_seen_at: float

    @property
    def label(self) -> str:
        return f"S{self.index}"

    def update(self, embedding: np.ndarray, duration_seconds: float, weight: float) -> None:
        weight = clamp01(weight)
        centroid = (self.centroid * (1.0 - weight)) + (embedding * weight)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            self.centroid = (centroid / norm).astype(np.float32)
        self.sentence_count += 1
        self.speech_seconds += max(0.0, duration_seconds)
        self.last_seen_at = time.time()


@dataclass
class SpeakerDecision:
    assigned_speaker: str | None
    created_speaker: bool
    probabilities: dict[str, float]
    similarities: dict[str, float]
    unknown_probability: float
    top_similarity: float | None
    margin: float | None
    quality: float
    assignment_source: str = "embedding"


class SpeakerMemory:
    """Stable incremental speaker memory based only on voice embeddings."""

    def __init__(
        self,
        same_speaker_similarity: float,
        similarity_temperature: float,
        speaker_softmax_temperature: float,
        new_speaker_threshold: float,
        duplicate_profile_similarity: float,
        unknown_short_threshold: float,
        min_first_speaker_seconds: float,
        min_new_speaker_seconds: float,
        max_speakers: int,
        min_margin: float,
        margin_temperature: float,
        update_unknown_max: float,
    ) -> None:
        self.same_speaker_similarity = same_speaker_similarity
        self.similarity_temperature = similarity_temperature
        self.speaker_softmax_temperature = speaker_softmax_temperature
        self.new_speaker_threshold = new_speaker_threshold
        self.duplicate_profile_similarity = duplicate_profile_similarity
        self.unknown_short_threshold = unknown_short_threshold
        self.min_first_speaker_seconds = min_first_speaker_seconds
        self.min_new_speaker_seconds = min_new_speaker_seconds
        self.max_speakers = max_speakers
        self.min_margin = min_margin
        self.margin_temperature = margin_temperature
        self.update_unknown_max = update_unknown_max
        self._profiles: list[SpeakerProfile] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._profiles = []

    def classify(self, embedding: np.ndarray, duration_seconds: float) -> SpeakerDecision:
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)

        with self._lock:
            if not self._profiles:
                if duration_seconds < self.min_first_speaker_seconds:
                    return SpeakerDecision(
                        assigned_speaker=None,
                        created_speaker=False,
                        probabilities={"unknown": 1.0},
                        similarities={},
                        unknown_probability=1.0,
                        top_similarity=None,
                        margin=None,
                        quality=quality,
                    )
                profile = self._create_profile_locked(embedding, duration_seconds)
                return SpeakerDecision(
                    assigned_speaker=profile.label,
                    created_speaker=True,
                    probabilities={"unknown": 1.0},
                    similarities={},
                    unknown_probability=1.0,
                    top_similarity=None,
                    margin=None,
                    quality=quality,
                )

            return self._score_locked(
                embedding=embedding,
                duration_seconds=duration_seconds,
                quality=quality,
                allow_create=True,
                allow_update=True,
                force_assignment=False,
            )

    def score_existing(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        force_assignment: bool = False,
    ) -> SpeakerDecision:
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)

        with self._lock:
            if not self._profiles:
                return SpeakerDecision(
                    assigned_speaker=None,
                    created_speaker=False,
                    probabilities={"unknown": 1.0},
                    similarities={},
                    unknown_probability=1.0,
                    top_similarity=None,
                    margin=None,
                    quality=quality,
                )
            return self._score_locked(
                embedding=embedding,
                duration_seconds=duration_seconds,
                quality=quality,
                allow_create=False,
                allow_update=False,
                force_assignment=force_assignment,
            )

    def _score_locked(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        quality: float,
        allow_create: bool,
        allow_update: bool,
        force_assignment: bool,
    ) -> SpeakerDecision:
        similarities = [
            cosine_similarity(embedding, profile.centroid)
            for profile in self._profiles
        ]
        order = sorted(range(len(similarities)), key=lambda index: similarities[index], reverse=True)
        top_index = order[0]
        top_similarity = similarities[top_index]
        second_similarity = similarities[order[1]] if len(order) > 1 else -1.0
        margin = top_similarity - second_similarity if len(order) > 1 else 1.0

        same_probability = sigmoid(
            (top_similarity - self.same_speaker_similarity)
            / self.similarity_temperature
        )
        margin_probability = 1.0
        if len(self._profiles) > 1:
            margin_probability = sigmoid(
                (margin - self.min_margin) / self.margin_temperature
            )
        maturity = min(1.0, 0.45 + 0.55 * (self._profiles[top_index].speech_seconds / 8.0))
        quality_factor = 0.55 + 0.45 * quality
        known_mass = clamp01(
            same_probability
            * margin_probability
            * maturity
            * quality_factor
        )
        unknown_probability = 1.0 - known_mass
        speaker_distribution = softmax(similarities, self.speaker_softmax_temperature)

        probabilities = {"unknown": unknown_probability}
        similarities_by_label = {}
        for profile, similarity, speaker_probability in zip(
            self._profiles,
            similarities,
            speaker_distribution,
        ):
            key = f"speaker{profile.index}"
            probabilities[key] = known_mass * speaker_probability
            similarities_by_label[profile.label] = similarity

        created_speaker = False
        assigned_profile: SpeakerProfile | None = None
        if (
            allow_create
            and unknown_probability >= self.new_speaker_threshold
            and top_similarity < self.duplicate_profile_similarity
            and duration_seconds >= self.min_new_speaker_seconds
            and len(self._profiles) < self.max_speakers
        ):
            assigned_profile = self._create_profile_locked(embedding, duration_seconds)
            created_speaker = True
        elif (
            not force_assignment
            and unknown_probability >= self.unknown_short_threshold
            and duration_seconds < self.min_new_speaker_seconds
        ):
            assigned_profile = None
        else:
            assigned_profile = self._profiles[top_index]
            if allow_update and unknown_probability <= self.update_unknown_max and quality >= 0.35:
                update_weight = min(0.28, 0.08 + 0.18 * quality)
                update_weight /= max(1.0, float(assigned_profile.sentence_count) ** 0.35)
                assigned_profile.update(embedding, duration_seconds, update_weight)

        return SpeakerDecision(
            assigned_speaker=None if assigned_profile is None else assigned_profile.label,
            created_speaker=created_speaker,
            probabilities={key: round(float(value), 4) for key, value in probabilities.items()},
            similarities={key: round(float(value), 4) for key, value in similarities_by_label.items()},
            unknown_probability=round(float(unknown_probability), 4),
            top_similarity=round(float(top_similarity), 4),
            margin=round(float(margin), 4),
            quality=round(float(quality), 4),
        )

    def _create_profile_locked(self, embedding: np.ndarray, duration_seconds: float) -> SpeakerProfile:
        now = time.time()
        profile = SpeakerProfile(
            index=len(self._profiles) + 1,
            centroid=embedding.astype(np.float32),
            sentence_count=1,
            speech_seconds=max(0.0, duration_seconds),
            created_at=now,
            last_seen_at=now,
        )
        self._profiles.append(profile)
        return profile

    @staticmethod
    def _duration_quality(duration_seconds: float) -> float:
        # Embeddings from tiny utterances are noisy. Treat 2.2s+ as full quality.
        return max(0.25, min(1.0, (duration_seconds - 0.35) / (2.2 - 0.35)))


class EmbeddingSubprocessClient:
    def __init__(self, python: Path, provider: str, device: str) -> None:
        self.python = Path(python)
        self.provider = provider
        self.device = device
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._noise_lines: list[str] = []

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = [
            str(self.python),
            str(Path(__file__).resolve()),
            "--embedding-helper",
            "--embedding-provider",
            self.provider,
            "--embedding-device",
            self.device,
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        threading.Thread(
            target=self._collect_stderr,
            name="EmbeddingHelperStderr",
            daemon=True,
        ).start()

    def embed_wav(self, path: Path) -> np.ndarray:
        with self._lock:
            self.start()
            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Embedding helper did not start.")
            if self._process.poll() is not None:
                raise RuntimeError(
                    "Embedding helper exited before processing audio. "
                    + self._stderr_tail()
                )
            request = {"cmd": "embed", "path": str(path)}
            self._process.stdin.write(json_dumps(request) + "\n")
            self._process.stdin.flush()
            response = self._read_json_response()
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "Embedding failed"))
            return normalize_vector(response["embedding"])

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write(json_dumps({"cmd": "shutdown"}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self._process = None

    def _read_json_response(self) -> dict[str, Any]:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "Embedding helper closed stdout. "
                    + self._stderr_tail()
                )
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                self._noise_lines.append(stripped)
                self._noise_lines = self._noise_lines[-20:]

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.rstrip()
            if text:
                self._stderr_lines.append(text)
                self._stderr_lines = self._stderr_lines[-30:]

    def _stderr_tail(self) -> str:
        lines = self._stderr_lines[-8:] + self._noise_lines[-4:]
        if not lines:
            return ""
        return "Helper output: " + " | ".join(lines)


@dataclass
class SentenceJob:
    session_id: str
    index: int
    text: str
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    split_reason: str | None = None
    part_index: int | None = None
    part_count: int | None = None
    video_start_seconds: float | None = None
    video_end_seconds: float | None = None


@dataclass
class ProcessedSentenceRecord:
    session_id: str
    index: int
    text: str
    duration_seconds: float
    embedding: np.ndarray
    decision: SpeakerDecision
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    video_start_seconds: float | None = None
    video_end_seconds: float | None = None
    split_reason: str | None = None
    part_index: int | None = None
    part_count: int | None = None


def is_reassignment_candidate(
    args: argparse.Namespace,
    decision: SpeakerDecision,
    duration_seconds: float,
) -> bool:
    if not getattr(args, "reassign_uncertain_sentences", True):
        return False
    if decision.created_speaker:
        return False
    if duration_seconds > args.reassign_max_seconds:
        return False
    if decision.assigned_speaker is None:
        return True
    return decision.unknown_probability >= args.reassign_unknown_min


def is_reassignment_accepted(args: argparse.Namespace, decision: SpeakerDecision) -> bool:
    if not decision.assigned_speaker:
        return False
    similarity_ok = (
        decision.top_similarity is not None
        and decision.top_similarity >= args.reassign_min_similarity
    )
    unknown_ok = decision.unknown_probability <= args.reassign_unknown_max
    return similarity_ok or unknown_ok


def speaker_probability_key(speaker_label: str | None) -> str | None:
    if not speaker_label:
        return None
    match = re.fullmatch(r"S(\d+)", str(speaker_label).strip())
    if not match:
        return None
    return f"speaker{match.group(1)}"


def context_adjusted_decision(
    decision: SpeakerDecision,
    assigned_speaker: str,
    confidence: float,
) -> SpeakerDecision:
    confidence = clamp01(confidence)
    speaker_key = speaker_probability_key(assigned_speaker)
    probabilities = {"unknown": round(float(1.0 - confidence), 4)}
    if speaker_key is not None:
        probabilities[speaker_key] = round(float(confidence), 4)
    return replace(
        decision,
        assigned_speaker=assigned_speaker,
        created_speaker=False,
        probabilities=probabilities,
        unknown_probability=round(float(1.0 - confidence), 4),
        assignment_source="context",
    )


def is_context_assignment_candidate(
    args: argparse.Namespace,
    decision: SpeakerDecision,
    duration_seconds: float,
) -> bool:
    if not getattr(args, "context_assign_short_fragments", True):
        return False
    if duration_seconds > args.context_assign_max_seconds:
        return False
    if not decision.assigned_speaker:
        return True
    return decision.unknown_probability >= args.context_assign_candidate_unknown_min


def is_context_anchor(
    args: argparse.Namespace,
    record: ProcessedSentenceRecord,
) -> bool:
    decision = record.decision
    if not decision.assigned_speaker:
        return False
    if decision.assignment_source == "context":
        return False
    if record.duration_seconds < args.context_assign_stable_min_seconds:
        return False
    return decision.unknown_probability <= args.context_assign_stable_unknown_max


class RealtimeSpeakerEngine:
    def __init__(self, args: argparse.Namespace, bus: "EventBus") -> None:
        self.args = args
        self.bus = bus
        self.memory = self._new_memory()
        self.jobs: "queue.Queue[SentenceJob]" = queue.Queue()
        self.client = EmbeddingSubprocessClient(
            python=Path(args.embedding_python),
            provider=args.embedding_provider,
            device=args.embedding_device,
        )
        self._records_by_session: dict[str, dict[int, ProcessedSentenceRecord]] = {}
        self._records_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._session_id: str | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="RealtimeSpeakerEmbeddingWorker",
            daemon=True,
        )
        self._worker.start()

    def start_session(self, session_id: str) -> None:
        self._session_id = session_id
        self.memory = self._new_memory()
        with self._records_lock:
            self._records_by_session = {session_id: {}}
        self._status(session_id, f"Embedding provider: {self.args.embedding_provider}.")
        self._status(session_id, f"Embedding device: {self.args.embedding_device}.")
        self._status(session_id, f"Embedding helper Python: {self.args.embedding_python}.")
        self._status(
            session_id,
            f"RealtimeSTT post-speech silence: {self.args.post_speech_silence_duration:.2f}s.",
        )

    def submit(
        self,
        session_id: str,
        index: int,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        source_start_seconds: float | None = None,
        source_end_seconds: float | None = None,
        split_reason: str | None = None,
        part_index: int | None = None,
        part_count: int | None = None,
        video_start_seconds: float | None = None,
        video_end_seconds: float | None = None,
    ) -> None:
        duration_seconds = float(len(audio)) / float(sample_rate or SAMPLE_RATE)
        pending_payload = self._sentence_payload(
            session_id=session_id,
            index=index,
            text=text,
            duration_seconds=duration_seconds,
            source_start_seconds=source_start_seconds,
            source_end_seconds=source_end_seconds,
            split_reason=split_reason,
            part_index=part_index,
            part_count=part_count,
            video_start_seconds=video_start_seconds,
            video_end_seconds=video_end_seconds,
            pending=True,
            decision=None,
            error=None,
        )
        self.bus.emit("sentence", pending_payload)
        self.jobs.put(
            SentenceJob(
                session_id=session_id,
                index=index,
                text=text,
                audio=audio,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                source_start_seconds=source_start_seconds,
                source_end_seconds=source_end_seconds,
                split_reason=split_reason,
                part_index=part_index,
                part_count=part_count,
                video_start_seconds=video_start_seconds,
                video_end_seconds=video_end_seconds,
            )
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        self.client.shutdown()

    def _new_memory(self) -> SpeakerMemory:
        return SpeakerMemory(
            same_speaker_similarity=self.args.same_speaker_similarity,
            similarity_temperature=self.args.similarity_temperature,
            speaker_softmax_temperature=self.args.speaker_softmax_temperature,
            new_speaker_threshold=self.args.new_speaker_threshold,
            duplicate_profile_similarity=self.args.duplicate_profile_similarity,
            unknown_short_threshold=self.args.unknown_short_threshold,
            min_first_speaker_seconds=self.args.min_first_speaker_seconds,
            min_new_speaker_seconds=self.args.min_new_speaker_seconds,
            max_speakers=self.args.max_speakers,
            min_margin=self.args.min_margin,
            margin_temperature=self.args.margin_temperature,
            update_unknown_max=self.args.update_unknown_max,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if job.session_id != self._session_id:
                    continue
                audio = trim_silence(job.audio, job.sample_rate)
                audio = pad_audio(audio, self.args.min_embed_seconds, job.sample_rate)
                wav_path = self._write_temp_wav(job, audio)
                try:
                    embedding = self.client.embed_wav(wav_path)
                finally:
                    if not self.args.keep_segment_audio:
                        try:
                            wav_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                decision = self.memory.classify(embedding, job.duration_seconds)
                self._store_record(job, embedding, decision)
                self.bus.emit(
                    "sentence",
                    self._sentence_payload(
                        session_id=job.session_id,
                        index=job.index,
                        text=job.text,
                        duration_seconds=job.duration_seconds,
                        source_start_seconds=job.source_start_seconds,
                        source_end_seconds=job.source_end_seconds,
                        split_reason=job.split_reason,
                        part_index=job.part_index,
                        part_count=job.part_count,
                        video_start_seconds=job.video_start_seconds,
                        video_end_seconds=job.video_end_seconds,
                        pending=False,
                        decision=decision,
                        error=None,
                    ),
                )
                self._revisit_uncertain_records(job.session_id)
            except Exception as exc:
                self.bus.emit(
                    "sentence",
                    self._sentence_payload(
                        session_id=job.session_id,
                        index=job.index,
                        text=job.text,
                        duration_seconds=job.duration_seconds,
                        source_start_seconds=job.source_start_seconds,
                        source_end_seconds=job.source_end_seconds,
                        split_reason=job.split_reason,
                        part_index=job.part_index,
                        part_count=job.part_count,
                        video_start_seconds=job.video_start_seconds,
                        video_end_seconds=job.video_end_seconds,
                        pending=False,
                        decision=None,
                        error=str(exc),
                    ),
                )
                self._error(job.session_id, f"Embedding failed for sentence {job.index}: {exc}")
            finally:
                self.jobs.task_done()

    def _store_record(
        self,
        job: SentenceJob,
        embedding: np.ndarray,
        decision: SpeakerDecision,
    ) -> None:
        with self._records_lock:
            records = self._records_by_session.setdefault(job.session_id, {})
            records[job.index] = ProcessedSentenceRecord(
                session_id=job.session_id,
                index=job.index,
                text=job.text,
                duration_seconds=job.duration_seconds,
                embedding=embedding.astype(np.float32, copy=True),
                decision=decision,
                source_start_seconds=job.source_start_seconds,
                source_end_seconds=job.source_end_seconds,
                video_start_seconds=job.video_start_seconds,
                video_end_seconds=job.video_end_seconds,
                split_reason=job.split_reason,
                part_index=job.part_index,
                part_count=job.part_count,
            )

    def _revisit_uncertain_records(self, session_id: str) -> None:
        with self._records_lock:
            records = list(self._records_by_session.get(session_id, {}).values())

        records = sorted(records, key=lambda item: item.index)
        for record in records:
            old_decision = record.decision
            if not is_reassignment_candidate(
                self.args,
                old_decision,
                record.duration_seconds,
            ):
                continue
            candidate = self.memory.score_existing(
                record.embedding,
                record.duration_seconds,
                force_assignment=True,
            )
            if not is_reassignment_accepted(self.args, candidate):
                continue
            if old_decision.assigned_speaker == candidate.assigned_speaker:
                continue

            self._apply_record_revision(session_id, record, old_decision, candidate)

        for record in records:
            old_decision = record.decision
            if not is_context_assignment_candidate(
                self.args,
                old_decision,
                record.duration_seconds,
            ):
                continue
            candidate = self._context_assignment_candidate(record, records)
            if candidate is None:
                continue
            if old_decision.assigned_speaker == candidate.assigned_speaker:
                continue

            self._apply_record_revision(session_id, record, old_decision, candidate)

    def _apply_record_revision(
        self,
        session_id: str,
        record: ProcessedSentenceRecord,
        old_decision: SpeakerDecision,
        candidate: SpeakerDecision,
    ) -> None:
        with self._records_lock:
            current = self._records_by_session.get(session_id, {}).get(record.index)
            if current is None or current.decision is not old_decision:
                return
            current.decision = candidate
            record.decision = candidate
        self.bus.emit(
            "sentence",
            self._sentence_payload(
                session_id=session_id,
                index=record.index,
                text=record.text,
                duration_seconds=record.duration_seconds,
                source_start_seconds=record.source_start_seconds,
                source_end_seconds=record.source_end_seconds,
                split_reason=record.split_reason or "revision",
                part_index=record.part_index,
                part_count=record.part_count,
                video_start_seconds=record.video_start_seconds,
                video_end_seconds=record.video_end_seconds,
                pending=False,
                decision=candidate,
                error=None,
                revision=True,
            ),
        )

    def _context_assignment_candidate(
        self,
        record: ProcessedSentenceRecord,
        records: list[ProcessedSentenceRecord],
    ) -> SpeakerDecision | None:
        try:
            current_position = records.index(record)
        except ValueError:
            return None

        previous_speaker = None
        next_speaker = None
        window = max(1, int(self.args.context_assign_window))
        for position in range(current_position - 1, max(-1, current_position - window - 1), -1):
            candidate = records[position]
            if is_context_anchor(self.args, candidate):
                previous_speaker = candidate.decision.assigned_speaker
                break
        for position in range(current_position + 1, min(len(records), current_position + window + 1)):
            candidate = records[position]
            if is_context_anchor(self.args, candidate):
                next_speaker = candidate.decision.assigned_speaker
                break

        chosen_speaker = None
        confidence = 0.0
        if previous_speaker and next_speaker and previous_speaker == next_speaker:
            chosen_speaker = previous_speaker
            confidence = self.args.context_assign_same_speaker_confidence
        elif self.args.context_assign_one_sided and previous_speaker:
            chosen_speaker = previous_speaker
            confidence = self.args.context_assign_one_sided_confidence
        elif self.args.context_assign_one_sided and next_speaker:
            chosen_speaker = next_speaker
            confidence = max(0.0, self.args.context_assign_one_sided_confidence - 0.04)

        if not chosen_speaker:
            return None
        return context_adjusted_decision(record.decision, chosen_speaker, confidence)

    def _write_temp_wav(self, job: SentenceJob, audio: np.ndarray) -> Path:
        directory = Path(self.args.segment_audio_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_index = f"{job.index:05d}"
        if self.args.keep_segment_audio:
            path = directory / f"{safe_index}.wav"
        else:
            handle = tempfile.NamedTemporaryFile(
                suffix=f".{safe_index}.wav",
                prefix="realtime-speaker-",
                dir=str(directory),
                delete=False,
            )
            handle.close()
            path = Path(handle.name)
        write_wav(path, audio, job.sample_rate)
        return path

    @staticmethod
    def _sentence_payload(
        session_id: str,
        index: int,
        text: str,
        duration_seconds: float,
        source_start_seconds: float | None,
        source_end_seconds: float | None,
        split_reason: str | None,
        part_index: int | None,
        part_count: int | None,
        video_start_seconds: float | None,
        video_end_seconds: float | None,
        pending: bool,
        decision: SpeakerDecision | None,
        error: str | None,
        revision: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "index": index,
            "text": text,
            "duration_seconds": round(float(duration_seconds), 4),
            "source_start_seconds": (
                None if source_start_seconds is None else round(float(source_start_seconds), 4)
            ),
            "source_end_seconds": (
                None if source_end_seconds is None else round(float(source_end_seconds), 4)
            ),
            "split_reason": split_reason,
            "part_index": part_index,
            "part_count": part_count,
            "video_start_seconds": (
                None if video_start_seconds is None else round(float(video_start_seconds), 4)
            ),
            "video_end_seconds": (
                None if video_end_seconds is None else round(float(video_end_seconds), 4)
            ),
            "pending": pending,
            "revision": revision,
            "error": error,
            "assigned_speaker": None,
            "created_speaker": False,
            "probabilities": {"unknown": 1.0} if pending else {},
            "similarities": {},
            "unknown_probability": None,
            "top_similarity": None,
            "margin": None,
            "quality": None,
            "assignment_source": None,
        }
        if decision is not None:
            payload.update({
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
                "quality": decision.quality,
                "assignment_source": decision.assignment_source,
            })
        return payload

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit("error-status", {"session_id": session_id, "message": message})


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id

    path_match = re.search(r"/(?:embed|shorts|live)/([^/?#]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    raise ValueError("Could not extract a YouTube video id from the URL.")


class TraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        source: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "time": time.time(),
            "source": source,
            "event": event,
            "payload": payload or {},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


@dataclass
class VideoClock:
    current_time_seconds: float
    monotonic_seconds: float


def list_audio_input_devices() -> list[dict[str, Any]]:
    try:
        import pyaudiowpatch as pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyaudiowpatch is required for WASAPI loopback capture."
        ) from exc

    audio = pyaudio.PyAudio()
    devices: list[dict[str, Any]] = []
    try:
        host_api_names = {}
        for host_index in range(audio.get_host_api_count()):
            try:
                host_info = audio.get_host_api_info_by_index(host_index)
                host_api_names[host_index] = host_info.get("name", "")
            except Exception:
                host_api_names[host_index] = ""

        for index in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(index)
            except Exception:
                continue
            max_input_channels = int(info.get("maxInputChannels") or 0)
            if max_input_channels <= 0:
                continue
            host_api_index = int(info.get("hostApi") or 0)
            devices.append({
                "index": int(info.get("index", index)),
                "name": str(info.get("name", "")),
                "host_api": host_api_names.get(host_api_index, ""),
                "channels": max_input_channels,
                "sample_rate": int(float(info.get("defaultSampleRate") or 0)),
                "is_loopback": bool(info.get("isLoopbackDevice")),
            })
    finally:
        audio.terminate()

    return devices


def choose_wasapi_loopback_device(
    requested_index: int | None,
    allow_default_input: bool,
) -> tuple[int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    devices = list_audio_input_devices()
    if requested_index is not None:
        for device in devices:
            if device["index"] == requested_index:
                return requested_index, device, devices
        raise RuntimeError(f"Input device index {requested_index} is not available.")

    try:
        import pyaudiowpatch as pyaudio

        audio = pyaudio.PyAudio()
        try:
            default_loopback = audio.get_default_wasapi_loopback()
        finally:
            audio.terminate()
        if default_loopback and default_loopback.get("isLoopbackDevice"):
            index = int(default_loopback["index"])
            for device in devices:
                if device["index"] == index:
                    return index, device, devices
            return index, {
                "index": index,
                "name": str(default_loopback.get("name", "")),
                "host_api": "Windows WASAPI",
                "channels": int(default_loopback.get("maxInputChannels") or 2),
                "sample_rate": int(float(default_loopback.get("defaultSampleRate") or 48000)),
                "is_loopback": True,
            }, devices
    except Exception:
        pass

    loopbacks = [device for device in devices if device.get("is_loopback")]
    if loopbacks:
        return loopbacks[0]["index"], loopbacks[0], devices

    if allow_default_input:
        return None, None, devices

    device_lines = [
        f"{item['index']}: {item['name']} [{item['host_api']}, "
        f"{item['channels']}ch, {item['sample_rate']} Hz"
        f"{', loopback' if item.get('is_loopback') else ''}]"
        for item in devices
    ]
    raise RuntimeError(
        "No WASAPI loopback-like input device was detected. "
        "Rerun with --input-device-index N after choosing one of:\n"
        + "\n".join(device_lines)
    )


class EventBus:
    def __init__(self, trace: TraceLogger | None = None) -> None:
        self._subscribers: set[queue.Queue[tuple[str, str]]] = set()
        self._lock = threading.Lock()
        self._trace = trace

    def subscribe(self) -> queue.Queue[tuple[str, str]]:
        subscriber: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=300)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[tuple[str, str]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.write("backend_sse", event, payload)
        message = json_dumps(payload)
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait((event, message))
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait((event, message))
                except queue.Full:
                    pass


class YouTubeWasapiController:
    def __init__(self, args: argparse.Namespace, bus: EventBus) -> None:
        self.args = args
        self.bus = bus
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._recorder: Any = None
        self._audio_interface: Any = None
        self._audio_stream: Any = None
        self._final_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._video_ids: dict[str, str] = {}
        self._video_clocks: dict[str, VideoClock] = {}
        self._sentence_indices: dict[str, int] = {}
        self.speaker_engine = RealtimeSpeakerEngine(args, bus)

    def start(self, url: str) -> tuple[str, str]:
        video_id = extract_youtube_video_id(url)
        self.stop(emit=False)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._session_id = session_id
            self._stop_event = threading.Event()
            self._video_ids[session_id] = video_id
            self._video_clocks.pop(session_id, None)
            self._sentence_indices[session_id] = 0
        self.speaker_engine.start_session(session_id)
        self._status(session_id, "Started.")
        worker = threading.Thread(
            target=self._start_capture,
            args=(session_id,),
            name="YouTubeWasapiCaptureStarter",
            daemon=True,
        )
        worker.start()
        return session_id, video_id

    def stop(self, emit: bool = True) -> None:
        with self._lock:
            session_id = self._session_id
            recorder = self._recorder
            stream = self._audio_stream
            audio_interface = self._audio_interface
            stop_event = self._stop_event

        stop_event.set()
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
            except Exception:
                pass
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass
        if emit and session_id:
            self._status(session_id, "Stop requested.")

    def shutdown(self) -> None:
        self.stop(emit=False)
        self.speaker_engine.shutdown()

    @staticmethod
    def _coerce_video_time(value: Any) -> float | None:
        try:
            current_time = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(current_time) or current_time < 0:
            return None
        return current_time

    def _set_video_time(self, session_id: str | None, current_time: Any) -> float | None:
        if not session_id:
            return None
        video_time = self._coerce_video_time(current_time)
        if video_time is None:
            return None
        with self._lock:
            if session_id != self._session_id:
                return None
            self._video_clocks[session_id] = VideoClock(
                current_time_seconds=video_time,
                monotonic_seconds=time.monotonic(),
            )
        return video_time

    def _estimate_video_time(self, session_id: str, at_monotonic: float | None = None) -> float | None:
        with self._lock:
            clock = self._video_clocks.get(session_id)
        if clock is None:
            return None
        if at_monotonic is None:
            at_monotonic = time.monotonic()
        elapsed = max(0.0, float(at_monotonic) - clock.monotonic_seconds)
        return max(0.0, clock.current_time_seconds + elapsed)

    def mark_video_playing(self, session_id: str | None, current_time: Any) -> None:
        video_time = self._set_video_time(session_id, current_time)
        if not session_id:
            return
        if video_time is None:
            self._status(session_id, "YouTube playback confirmed.")
        else:
            self._status(session_id, f"YouTube playback confirmed at {video_time:.2f}s.")

    def update_video_time(self, session_id: str | None, current_time: Any) -> None:
        self._set_video_time(session_id, current_time)

    def _is_current(self, session_id: str) -> bool:
        with self._lock:
            return session_id == self._session_id

    def _next_sentence_index(self, session_id: str) -> int:
        with self._lock:
            index = self._sentence_indices.get(session_id, 0)
            self._sentence_indices[session_id] = index + 1
            return index

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit("error-status", {"session_id": session_id, "message": message})

    def _start_capture(self, session_id: str) -> None:
        try:
            self._status(session_id, "Selecting WASAPI loopback input.")
            input_device_index, device, devices = choose_wasapi_loopback_device(
                self.args.input_device_index,
                self.args.allow_default_input,
            )
            if device is not None:
                self._status(
                    session_id,
                    "Using input device "
                    f"{device['index']}: {device['name']} [{device['host_api']}].",
                )
            else:
                self._status(session_id, "Using the default input device.")
            if devices:
                self._status(
                    session_id,
                    "Available input devices: "
                    + "; ".join(
                        f"{item['index']}={item['name']} [{item['host_api']}]"
                        for item in devices
                    ),
                )

            recorder = self._create_recorder(session_id, input_device_index)
            audio_interface, stream = self._open_loopback_stream(
                session_id,
                input_device_index,
                recorder,
                device,
            )
            with self._lock:
                if session_id != self._session_id:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    try:
                        audio_interface.terminate()
                    except Exception:
                        pass
                    recorder.shutdown()
                    return
                self._recorder = recorder
                self._audio_interface = audio_interface
                self._audio_stream = stream

            recorder.start()
            self._status(session_id, "WASAPI capture started.")
            final_thread = threading.Thread(
                target=self._consume_final_text,
                args=(session_id, recorder, self._stop_event),
                name="YouTubeWasapiFinalConsumer",
                daemon=True,
            )
            with self._lock:
                self._final_thread = final_thread
            final_thread.start()
            video_id = self._video_ids.get(session_id)
            if video_id:
                self.bus.emit(
                    "capture-ready",
                    {"session_id": session_id, "video_id": video_id},
                )
        except Exception as exc:
            if self._is_current(session_id):
                self._error(session_id, str(exc))

    def _open_loopback_stream(
        self,
        session_id: str,
        input_device_index: int | None,
        recorder: Any,
        device: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        try:
            import pyaudiowpatch as pyaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyaudiowpatch and numpy are required for WASAPI loopback capture."
            ) from exc

        audio_interface = pyaudio.PyAudio()
        try:
            if input_device_index is None:
                device_info = audio_interface.get_default_wasapi_loopback()
                input_device_index = int(device_info["index"])
            else:
                device_info = audio_interface.get_device_info_by_index(input_device_index)

            if not device_info.get("isLoopbackDevice"):
                raise RuntimeError(f"Device {input_device_index} is not a WASAPI loopback device.")

            sample_rate = int(device_info["defaultSampleRate"])
            channels = int(device_info["maxInputChannels"])
            if channels <= 0:
                raise RuntimeError(f"Loopback device {input_device_index} has no input channels.")

            display_name = (
                device["name"]
                if device is not None
                else str(device_info.get("name", input_device_index))
            )
            self._status(
                session_id,
                f"Opening loopback stream: {display_name}, {channels}ch, {sample_rate} Hz.",
            )

            def feed(data: bytes, _frame_count: int, _time_info: Any, _status: Any):
                audio = np.frombuffer(data, np.int16)
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                recorder.feed_audio(audio, original_sample_rate=sample_rate)
                return (None, pyaudio.paContinue)

            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=max(1, sample_rate // 10),
                stream_callback=feed,
            )
            stream.start_stream()
            return audio_interface, stream
        except Exception:
            try:
                audio_interface.terminate()
            except Exception:
                pass
            raise

    def _create_recorder(self, session_id: str, input_device_index: int | None) -> Any:
        self._status(session_id, "Initializing RealtimeSTT.")

        if os.name == "nt":
            try:
                from torchaudio._extension.utils import _init_dll_path

                _init_dll_path()
            except Exception:
                pass

        from RealtimeSTT import AudioToTextRecorder

        def stabilization_update(event: Any) -> None:
            display_text = getattr(event, "display_text", None)
            if display_text is None:
                display_text = getattr(event, "raw_observation_text", "")
            stable_text = getattr(event, "stable_text", "") or ""
            unstable_text = getattr(event, "unstable_text", "") or ""
            self.bus.emit(
                "realtime",
                {
                    "session_id": session_id,
                    "display_text": display_text or "",
                    "stable_text": stable_text,
                    "unstable_text": unstable_text,
                },
            )

        recorder = AudioToTextRecorder(
            use_microphone=False,
            input_device_index=None,
            spinner=False,
            model=self.args.model,
            realtime_model_type=self.args.rt_model,
            language=self.args.language,
            device=self.args.device,
            compute_type=self.args.compute_type,
            download_root=self.args.download_root,
            enable_realtime_transcription=True,
            realtime_punctuation_split_marks=self.args.split_marks,
            realtime_processing_pause=self.args.realtime_processing_pause,
            init_realtime_after_seconds=0.0,
            on_realtime_transcription_update=None,
            on_realtime_text_stabilization_update=stabilization_update,
            batch_size=self.args.batch_size,
            realtime_batch_size=self.args.realtime_batch_size,
            min_length_of_recording=self.args.min_length_of_recording,
            min_gap_between_recordings=0,
            post_speech_silence_duration=self.args.post_speech_silence_duration,
            silero_sensitivity=self.args.silero_sensitivity,
            webrtc_sensitivity=self.args.webrtc_sensitivity,
            silero_deactivity_detection=True,
            realtime_transcription_use_syllable_boundaries=True,
            realtime_boundary_detector_sensitivity=0.6,
            realtime_boundary_followup_delays=(0.05, 0.2),
            beam_size=self.args.beam_size,
            beam_size_realtime=self.args.beam_size_realtime,
            no_log_file=True,
            faster_whisper_vad_filter=False,
            final_transcription_word_timestamps=self.args.final_word_timestamps,
        )
        self._status(session_id, "RealtimeSTT ready.")
        return recorder

    def _read_final_text(self, recorder: Any) -> str:
        if self.args.final_word_timestamps:
            try:
                return (recorder.text(word_timestamps=True) or "").strip()
            except TypeError:
                self._error(
                    self._session_id,
                    "RealtimeSTT does not expose text(word_timestamps=...). "
                    "Falling back to unsplit final transcripts.",
                )
        return (recorder.text() or "").strip()

    def _consume_final_text(
        self,
        session_id: str,
        recorder: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                text = self._read_final_text(recorder)
            except Exception as exc:
                if not stop_event.is_set():
                    self._error(session_id, f"Final transcription failed: {exc}")
                return
            if not text:
                continue

            audio = audio_to_float_mono(getattr(recorder, "last_transcription_bytes", None))
            if len(audio) == 0:
                self._error(session_id, "No audio samples available for final transcript.")
                continue
            metadata = getattr(recorder, "last_transcription_metadata", None) or {}
            parts = split_transcript_by_timestamps(
                text=text,
                audio=audio,
                sample_rate=SAMPLE_RATE,
                metadata=metadata,
                args=self.args,
            )
            if len(parts) > 1:
                self._status(
                    session_id,
                    f"Split final transcript into {len(parts)} timestamped parts.",
                )

            final_audio_duration = float(len(audio)) / float(SAMPLE_RATE)
            video_current = self._estimate_video_time(session_id)
            video_chunk_end = None
            if video_current is not None:
                video_chunk_end = max(
                    0.0,
                    video_current - max(0.0, float(self.args.final_video_latency_seconds)),
                )

            for part in parts:
                index = self._next_sentence_index(session_id)
                video_start_seconds = None
                video_end_seconds = None
                if video_chunk_end is not None:
                    video_chunk_start = max(0.0, video_chunk_end - final_audio_duration)
                    video_start_seconds = max(0.0, video_chunk_start + part.start_seconds)
                    video_end_seconds = max(video_start_seconds, video_chunk_start + part.end_seconds)
                self.bus.emit(
                    "final",
                    {
                        "session_id": session_id,
                        "index": index,
                        "text": part.text,
                        "source_start_seconds": part.start_seconds,
                        "source_end_seconds": part.end_seconds,
                        "word_start_seconds": part.word_start_seconds,
                        "word_end_seconds": part.word_end_seconds,
                        "split_reason": part.split_reason,
                        "part_index": part.part_index,
                        "part_count": part.part_count,
                        "video_start_seconds": (
                            None
                            if video_start_seconds is None
                            else round(float(video_start_seconds), 4)
                        ),
                        "video_end_seconds": (
                            None
                            if video_end_seconds is None else round(float(video_end_seconds), 4)
                        ),
                    },
                )
                self.speaker_engine.submit(
                    session_id=session_id,
                    index=index,
                    text=part.text,
                    audio=part.audio,
                    sample_rate=SAMPLE_RATE,
                    source_start_seconds=part.start_seconds,
                    source_end_seconds=part.end_seconds,
                    split_reason=part.split_reason,
                    part_index=part.part_index,
                    part_count=part.part_count,
                    video_start_seconds=video_start_seconds,
                    video_end_seconds=video_end_seconds,
                )


class RequestHandler(BaseHTTPRequestHandler):
    server: "GuiServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/events":
            self._serve_events()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                url = str(payload.get("url") or "").strip()
                if not url:
                    raise ValueError("Missing YouTube URL.")
                session_id, video_id = self.server.controller.start(url)
                self._send_json({"session_id": session_id, "video_id": video_id})
                return
            if self.path == "/api/video-playing":
                self.server.controller.mark_video_playing(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/video-time":
                self.server.controller.update_video_time(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/debug-log":
                self.server.trace.write(
                    "frontend",
                    str(payload.get("stage") or "debug"),
                    payload,
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/stop":
                self.server.controller.stop()
                self._send_json({"ok": True})
                return
            self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json_dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        subscriber = self.server.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event, payload = subscriber.get(timeout=10)
                    message = f"event: {event}\ndata: {payload}\n\n"
                except queue.Empty:
                    message = ": heartbeat\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self.server.bus.unsubscribe(subscriber)


class GuiServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        args: argparse.Namespace,
        bus: EventBus,
        trace: TraceLogger,
    ) -> None:
        super().__init__(address, RequestHandler)
        self.args = args
        self.bus = bus
        self.trace = trace
        self.controller = YouTubeWasapiController(args, bus)


def run_embedding_helper(args: argparse.Namespace) -> int:
    provider = create_embedding_provider(args.embedding_provider, args.embedding_device)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                print(json_dumps({"ok": True}), flush=True)
                return 0
            if command != "embed":
                raise ValueError(f"Unknown helper command: {command!r}")
            path = Path(str(request["path"]))
            audio, sample_rate = load_audio_file(path)
            audio = trim_silence(audio, sample_rate)
            audio = pad_audio(audio, 0.5, sample_rate)
            embedding = provider.embed(audio, sample_rate)
            print(
                json_dumps({
                    "ok": True,
                    "embedding": embedding.astype(float).tolist(),
                    "seconds": len(audio) / float(sample_rate),
                }),
                flush=True,
            )
        except Exception as exc:
            print(
                json_dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }),
                flush=True,
            )
    return 0


def read_canonical_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return [
            {
                "speaker": segment["speaker_id"],
                "start": segment["start_sec"],
                "end": segment["end_sec"],
                "text": segment["text"],
            }
            for segment in data["segments"]
        ]
    raise ValueError(f"Could not read canonical segments from {path}")


def embed_audio_with_client(
    client: EmbeddingSubprocessClient,
    audio: np.ndarray,
    sample_rate: int,
    directory: Path,
    suffix: str,
) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="realtime-speaker-validate-",
        dir=str(directory),
        delete=False,
    ) as handle:
        wav_path = Path(handle.name)
    try:
        write_wav(wav_path, audio, sample_rate)
        return client.embed_wav(wav_path)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def validate_cunk(args: argparse.Namespace) -> int:
    configure_embedding_env()
    client = EmbeddingSubprocessClient(
        python=Path(args.embedding_python),
        provider=args.embedding_provider,
        device=args.embedding_device,
    )
    memory = SpeakerMemory(
        same_speaker_similarity=args.same_speaker_similarity,
        similarity_temperature=args.similarity_temperature,
        speaker_softmax_temperature=args.speaker_softmax_temperature,
        new_speaker_threshold=args.new_speaker_threshold,
        duplicate_profile_similarity=args.duplicate_profile_similarity,
        unknown_short_threshold=args.unknown_short_threshold,
        min_first_speaker_seconds=args.min_first_speaker_seconds,
        min_new_speaker_seconds=args.min_new_speaker_seconds,
        max_speakers=args.max_speakers,
        min_margin=args.min_margin,
        margin_temperature=args.margin_temperature,
        update_unknown_max=args.update_unknown_max,
    )
    audio, sample_rate = load_audio_file(args.validation_audio)
    segments = read_canonical_segments(args.validation_canonical)

    rows: list[dict[str, Any]] = []
    validation_records: list[ProcessedSentenceRecord] = []
    try:
        for index, segment in enumerate(segments):
            start_sample = max(0, int(float(segment["start"]) * sample_rate))
            end_sample = min(len(audio), int(float(segment["end"]) * sample_rate))
            chunk = trim_silence(audio[start_sample:end_sample], sample_rate)
            chunk = pad_audio(chunk, args.min_embed_seconds, sample_rate)
            embedding = embed_audio_with_client(
                client,
                chunk,
                sample_rate,
                args.validation_output.parent,
                f".canonical-{index:04d}.wav",
            )
            duration_seconds = float(segment["end"]) - float(segment["start"])
            decision = memory.classify(embedding, duration_seconds)
            rows.append({
                "index": index,
                "start": segment["start"],
                "end": segment["end"],
                "duration_seconds": round(duration_seconds, 4),
                "text": segment.get("text", ""),
                "canonical_speaker": segment["speaker"],
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "reassigned": False,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
            })
            validation_records.append(
                ProcessedSentenceRecord(
                    session_id="validation",
                    index=index,
                    text=segment.get("text", ""),
                    duration_seconds=duration_seconds,
                    embedding=embedding.astype(np.float32, copy=True),
                    decision=decision,
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for record in validation_records:
        old_decision = record.decision
        if not is_reassignment_candidate(args, old_decision, record.duration_seconds):
            continue
        candidate = memory.score_existing(
            record.embedding,
            record.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        record.decision = candidate
        row = rows[record.index]
        row.update({
            "assigned_speaker": candidate.assigned_speaker,
            "created_speaker": candidate.created_speaker,
            "reassigned": True,
            "probabilities": candidate.probabilities,
            "similarities": candidate.similarities,
            "unknown_probability": candidate.unknown_probability,
            "top_similarity": candidate.top_similarity,
            "margin": candidate.margin,
        })
        reassigned_count += 1

    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        assigned = row["assigned_speaker"]
        if assigned:
            profile_speaker_durations[assigned][row["canonical_speaker"]] += row["duration_seconds"]
    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }

    correct_count = 0
    unknown_count = 0
    correct_duration = 0.0
    total_duration = 0.0
    for row in rows:
        total_duration += row["duration_seconds"]
        mapped = profile_map.get(row["assigned_speaker"])
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == row["canonical_speaker"])
        if not row["assigned_speaker"]:
            unknown_count += 1
        if row["matches_canonical"]:
            correct_count += 1
            correct_duration += row["duration_seconds"]

    summary = {
        "embedding_provider": args.embedding_provider,
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "segments": len(rows),
        "profiles": len(profile_map),
        "unknown_segments": unknown_count,
        "reassigned_segments": reassigned_count,
        "segment_accuracy": round(correct_count / max(1, len(rows)), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "profile_map": profile_map,
        "rows": rows,
    }

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Validation output: {args.validation_output}", flush=True)
    print(f"Provider: {summary['embedding_provider']}", flush=True)
    print(f"Segments: {summary['segments']}", flush=True)
    print(f"Profiles: {summary['profiles']} ({summary['profile_map']})", flush=True)
    print(f"Unknown short/noisy segments: {summary['unknown_segments']}", flush=True)
    print(f"Reassigned uncertain segments: {summary['reassigned_segments']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0


def canonical_overlap(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> tuple[str | None, float, Counter[str], float]:
    overlaps: Counter[str] = Counter()
    for segment in segments:
        left = max(start, float(segment["start"]))
        right = min(end, float(segment["end"]))
        if right <= left:
            continue
        overlaps[str(segment["speaker"])] += right - left
    if not overlaps:
        return None, 0.0, overlaps, 0.0
    dominant, dominant_seconds = overlaps.most_common(1)[0]
    total = sum(overlaps.values())
    return dominant, float(dominant_seconds), overlaps, float(total)


def summarize_canonical_gap_groups(
    segments: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.6, 0.4, 0.28, 0.2, 0.12),
) -> list[dict[str, Any]]:
    summaries = []
    if not segments:
        return summaries
    for threshold in thresholds:
        groups = []
        current = [segments[0]]
        for previous, segment in zip(segments, segments[1:]):
            gap = float(segment["start"]) - float(previous["end"])
            if gap <= threshold:
                current.append(segment)
            else:
                groups.append(current)
                current = [segment]
        groups.append(current)
        mixed = [
            group for group in groups
            if len({item["speaker"] for item in group}) > 1
        ]
        summaries.append({
            "threshold_seconds": threshold,
            "groups": len(groups),
            "mixed_groups": len(mixed),
            "mixed_duration_seconds": round(
                sum(float(group[-1]["end"]) - float(group[0]["start"]) for group in mixed),
                4,
            ),
            "max_group_seconds": round(
                max(float(group[-1]["end"]) - float(group[0]["start"]) for group in groups),
                4,
            ),
        })
    return summaries


def transcribe_validation_audio_with_realtimestt(
    args: argparse.Namespace,
    audio: np.ndarray,
) -> tuple[str, dict[str, Any], float]:
    if os.name == "nt":
        try:
            from torchaudio._extension.utils import _init_dll_path

            _init_dll_path()
        except Exception:
            pass

    from RealtimeSTT import AudioToTextRecorder

    recorder = AudioToTextRecorder(
        use_microphone=False,
        input_device_index=None,
        spinner=False,
        model=args.model,
        realtime_model_type=args.rt_model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.download_root,
        enable_realtime_transcription=False,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        no_log_file=True,
        faster_whisper_vad_filter=False,
        final_transcription_word_timestamps=True,
    )
    started = time.perf_counter()
    try:
        text = recorder.perform_final_transcription(
            audio,
            use_prompt=False,
            word_timestamps=True,
        )
        elapsed = time.perf_counter() - started
        metadata = getattr(recorder, "last_transcription_metadata", None) or {}
        return (text or "").strip(), metadata, elapsed
    finally:
        recorder.shutdown()


def validate_cunk_word_splits(args: argparse.Namespace) -> int:
    audio, sample_rate = load_audio_file(args.validation_audio)
    canonical_segments = read_canonical_segments(args.validation_canonical)
    text, metadata, transcription_seconds = transcribe_validation_audio_with_realtimestt(
        args,
        audio,
    )
    parts = split_transcript_by_timestamps(
        text=text,
        audio=audio,
        sample_rate=sample_rate,
        metadata=metadata,
        args=args,
    )

    configure_embedding_env()
    client = EmbeddingSubprocessClient(
        python=Path(args.embedding_python),
        provider=args.embedding_provider,
        device=args.embedding_device,
    )
    memory = SpeakerMemory(
        same_speaker_similarity=args.same_speaker_similarity,
        similarity_temperature=args.similarity_temperature,
        speaker_softmax_temperature=args.speaker_softmax_temperature,
        new_speaker_threshold=args.new_speaker_threshold,
        duplicate_profile_similarity=args.duplicate_profile_similarity,
        unknown_short_threshold=args.unknown_short_threshold,
        min_first_speaker_seconds=args.min_first_speaker_seconds,
        min_new_speaker_seconds=args.min_new_speaker_seconds,
        max_speakers=args.max_speakers,
        min_margin=args.min_margin,
        margin_temperature=args.margin_temperature,
        update_unknown_max=args.update_unknown_max,
    )

    rows: list[dict[str, Any]] = []
    validation_records: list[ProcessedSentenceRecord] = []
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, part in enumerate(parts):
            chunk = trim_silence(part.audio, sample_rate)
            chunk = pad_audio(chunk, args.min_embed_seconds, sample_rate)
            embedding = embed_audio_with_client(
                client,
                chunk,
                sample_rate,
                args.validation_output.parent,
                f".word-split-{index:04d}.wav",
            )
            decision = memory.classify(embedding, part.duration_seconds)
            canonical_speaker, dominant_overlap, overlaps, total_overlap = canonical_overlap(
                canonical_segments,
                part.word_start_seconds,
                part.word_end_seconds,
            )
            overlap_speakers = {
                speaker: round(float(seconds), 4)
                for speaker, seconds in overlaps.items()
                if seconds >= args.mixed_overlap_min_seconds
            }
            rows.append({
                "index": index,
                "text": part.text,
                "audio_start": part.start_seconds,
                "audio_end": part.end_seconds,
                "word_start": part.word_start_seconds,
                "word_end": part.word_end_seconds,
                "duration_seconds": round(part.duration_seconds, 4),
                "word_count": part.word_count,
                "split_reason": part.split_reason,
                "canonical_speaker": canonical_speaker,
                "canonical_overlap_seconds": round(float(dominant_overlap), 4),
                "canonical_total_overlap_seconds": round(float(total_overlap), 4),
                "canonical_speaker_overlaps": overlap_speakers,
                "mixed_canonical_speakers": len(overlap_speakers) > 1,
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "reassigned": False,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
            })
            validation_records.append(
                ProcessedSentenceRecord(
                    session_id="word_split_validation",
                    index=index,
                    text=part.text,
                    duration_seconds=part.duration_seconds,
                    embedding=embedding.astype(np.float32, copy=True),
                    decision=decision,
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for record in validation_records:
        old_decision = record.decision
        if not is_reassignment_candidate(args, old_decision, record.duration_seconds):
            continue
        candidate = memory.score_existing(
            record.embedding,
            record.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        record.decision = candidate
        row = rows[record.index]
        row.update({
            "assigned_speaker": candidate.assigned_speaker,
            "created_speaker": candidate.created_speaker,
            "reassigned": True,
            "probabilities": candidate.probabilities,
            "similarities": candidate.similarities,
            "unknown_probability": candidate.unknown_probability,
            "top_similarity": candidate.top_similarity,
            "margin": candidate.margin,
        })
        reassigned_count += 1

    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        assigned = row["assigned_speaker"]
        canonical_speaker = row["canonical_speaker"]
        if assigned and canonical_speaker:
            profile_speaker_durations[assigned][canonical_speaker] += row[
                "canonical_overlap_seconds"
            ]
    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }

    correct_count = 0
    scored_count = 0
    unknown_count = 0
    correct_duration = 0.0
    total_duration = 0.0
    mixed_count = 0
    mixed_duration = 0.0
    for row in rows:
        if row["assigned_speaker"] is None:
            unknown_count += 1
        mapped = profile_map.get(row["assigned_speaker"])
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == row["canonical_speaker"])
        score_duration = max(
            0.0,
            float(row["word_end"]) - float(row["word_start"]),
        )
        if row["canonical_speaker"]:
            scored_count += 1
            total_duration += score_duration
            if row["matches_canonical"]:
                correct_count += 1
                correct_duration += score_duration
        if row["mixed_canonical_speakers"]:
            mixed_count += 1
            mixed_duration += score_duration

    word_count = len((metadata or {}).get("words") or [])
    summary = {
        "embedding_provider": args.embedding_provider,
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "transcription_model": args.model,
        "transcription_seconds": round(transcription_seconds, 4),
        "transcript_characters": len(text),
        "words": word_count,
        "split_parts": len(parts),
        "average_part_seconds": round(
            sum(part.duration_seconds for part in parts) / max(1, len(parts)),
            4,
        ),
        "profiles": len(profile_map),
        "unknown_segments": unknown_count,
        "reassigned_segments": reassigned_count,
        "mixed_segments": mixed_count,
        "mixed_duration_seconds": round(mixed_duration, 4),
        "segment_accuracy": round(correct_count / max(1, scored_count), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "profile_map": profile_map,
        "canonical_gap_group_baseline": summarize_canonical_gap_groups(canonical_segments),
        "transcript": text,
        "rows": rows,
    }

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Validation output: {args.validation_output}", flush=True)
    print(f"Transcription model: {summary['transcription_model']}", flush=True)
    print(f"Transcription seconds: {summary['transcription_seconds']:.2f}", flush=True)
    print(f"Words with timestamps: {summary['words']}", flush=True)
    print(f"Timestamped split parts: {summary['split_parts']}", flush=True)
    print(f"Average part seconds: {summary['average_part_seconds']:.3f}", flush=True)
    print(f"Mixed canonical-speaker parts: {summary['mixed_segments']}", flush=True)
    print(f"Mixed duration seconds: {summary['mixed_duration_seconds']:.3f}", flush=True)
    print(f"Profiles: {summary['profiles']} ({summary['profile_map']})", flush=True)
    print(f"Unknown short/noisy segments: {summary['unknown_segments']}", flush=True)
    print(f"Reassigned uncertain segments: {summary['reassigned_segments']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def normalized_match_text(text: str) -> str:
    return " ".join(text_tokens(text))


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0] * (len(right) + 1)
        for index, right_item in enumerate(right, 1):
            if left_item == right_item:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def lcs_alignment(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    if not left or not right:
        return []

    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index, left_item in enumerate(left, 1):
        row = table[left_index]
        previous_row = table[left_index - 1]
        for right_index, right_item in enumerate(right, 1):
            if left_item == right_item:
                row[right_index] = previous_row[right_index - 1] + 1
            else:
                row[right_index] = max(previous_row[right_index], row[right_index - 1])

    pairs: list[tuple[int, int]] = []
    left_index = len(left)
    right_index = len(right)
    while left_index > 0 and right_index > 0:
        if left[left_index - 1] == right[right_index - 1]:
            pairs.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif table[left_index - 1][right_index] >= table[left_index][right_index - 1]:
            left_index -= 1
        else:
            right_index -= 1
    pairs.reverse()
    return pairs


def lcs_speaker_matches_by_final(
    finals: dict[int, dict[str, Any]],
    canonical_segments: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    live_tokens: list[str] = []
    live_token_final_indices: list[int] = []
    live_final_token_counts: Counter[int] = Counter()
    for final_index, final in sorted(finals.items()):
        tokens = text_tokens(str(final.get("text") or ""))
        live_final_token_counts[final_index] = len(tokens)
        for token in tokens:
            live_tokens.append(token)
            live_token_final_indices.append(final_index)

    canonical_tokens: list[str] = []
    canonical_token_speakers: list[str] = []
    for segment in canonical_segments:
        speaker = str(segment.get("speaker") or "")
        if not speaker:
            continue
        for token in text_tokens(str(segment.get("text") or "")):
            canonical_tokens.append(token)
            canonical_token_speakers.append(speaker)

    matches: dict[int, Counter[str]] = defaultdict(Counter)
    for live_token_index, canonical_token_index in lcs_alignment(live_tokens, canonical_tokens):
        final_index = live_token_final_indices[live_token_index]
        speaker = canonical_token_speakers[canonical_token_index]
        matches[final_index][speaker] += 1

    result: dict[int, dict[str, Any]] = {}
    for final_index, speaker_counts in matches.items():
        if not speaker_counts:
            continue
        speaker, count = speaker_counts.most_common(1)[0]
        token_count = max(1, live_final_token_counts[final_index])
        result[final_index] = {
            "speaker": speaker,
            "score": count / float(token_count),
            "matched_tokens": int(sum(speaker_counts.values())),
            "speaker_token_counts": dict(speaker_counts),
        }
    return result


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(text_tokens(left))
    right_tokens = set(text_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def best_canonical_text_match(
    text: str,
    canonical_segments: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    normalized = normalized_match_text(text)
    if not normalized:
        return 0.0, None
    best_score = 0.0
    best_segment = None
    for segment in canonical_segments:
        segment_text = str(segment.get("text") or "")
        segment_normalized = normalized_match_text(segment_text)
        if not segment_normalized:
            continue
        ratio = difflib.SequenceMatcher(None, normalized, segment_normalized).ratio()
        score = 0.65 * ratio + 0.35 * token_jaccard(text, segment_text)
        if score > best_score:
            best_score = score
            best_segment = segment
    return best_score, best_segment


def analyze_trace_against_canonical(
    records: list[dict[str, Any]],
    canonical_segments: list[dict[str, Any]],
    match_mode: str = "auto",
) -> dict[str, Any]:
    finals: dict[int, dict[str, Any]] = {}
    sentences: dict[int, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload") or {}
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        if record.get("event") == "final":
            finals[index] = payload
        elif record.get("event") == "sentence" and not payload.get("pending"):
            sentences[index] = payload

    canonical_text = " ".join(str(segment.get("text") or "") for segment in canonical_segments)
    live_text = " ".join(str(final.get("text") or "") for _, final in sorted(finals.items()))
    canonical_tokens = text_tokens(canonical_text)
    live_tokens = text_tokens(live_text)
    common_tokens = lcs_length(live_tokens, canonical_tokens)

    rows: list[dict[str, Any]] = []
    timestamped_rows = 0
    text_matched_rows = 0
    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    profile_speaker_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lcs_text_matches = (
        {}
        if match_mode == "timestamp"
        else lcs_speaker_matches_by_final(finals, canonical_segments)
    )

    for index, final in sorted(finals.items()):
        sentence = sentences.get(index) or {}
        assigned = sentence.get("assigned_speaker") or final.get("assigned_speaker")
        text = str(final.get("text") or sentence.get("text") or "")
        video_start = final.get("video_start_seconds")
        video_end = final.get("video_end_seconds")
        if video_start is None:
            video_start = sentence.get("video_start_seconds")
        if video_end is None:
            video_end = sentence.get("video_end_seconds")

        canonical_speaker = None
        canonical_overlap_seconds = 0.0
        canonical_total_overlap_seconds = 0.0
        canonical_match_score = None
        row_match_mode = "unmatched"
        duration_seconds = float(sentence.get("duration_seconds") or 0.0)
        try:
            if match_mode != "text" and video_start is not None and video_end is not None:
                video_start = float(video_start)
                video_end = float(video_end)
                if math.isfinite(video_start) and math.isfinite(video_end) and video_end > video_start:
                    canonical_speaker, canonical_overlap_seconds, overlaps, canonical_total_overlap_seconds = (
                        canonical_overlap(canonical_segments, video_start, video_end)
                    )
                    duration_seconds = max(duration_seconds, video_end - video_start)
                    timestamped_rows += 1
                    row_match_mode = "timestamp"
                else:
                    video_start = None
                    video_end = None
        except (TypeError, ValueError):
            video_start = None
            video_end = None

        if canonical_speaker is None and match_mode != "timestamp":
            lcs_match = lcs_text_matches.get(index)
            if lcs_match is not None and float(lcs_match.get("score") or 0.0) >= 0.34:
                canonical_speaker = str(lcs_match["speaker"])
                canonical_match_score = float(lcs_match["score"])
                text_matched_rows += 1
                row_match_mode = "text_lcs"
            else:
                canonical_match_score, segment = best_canonical_text_match(text, canonical_segments)
                if segment is not None and canonical_match_score >= 0.45:
                    canonical_speaker = str(segment["speaker"])
                    text_matched_rows += 1
                    row_match_mode = "text"

        if assigned and canonical_speaker:
            weight = max(0.001, float(duration_seconds))
            profile_speaker_durations[str(assigned)][canonical_speaker] += weight
            profile_speaker_counts[str(assigned)][canonical_speaker] += 1

        rows.append({
            "index": index,
            "text": text,
            "assigned_speaker": assigned,
            "video_start_seconds": video_start,
            "video_end_seconds": video_end,
            "duration_seconds": round(float(duration_seconds), 4),
            "canonical_speaker": canonical_speaker,
            "canonical_overlap_seconds": round(float(canonical_overlap_seconds), 4),
            "canonical_total_overlap_seconds": round(float(canonical_total_overlap_seconds), 4),
            "canonical_text_match_score": canonical_match_score,
            "canonical_text_lcs_match": lcs_text_matches.get(index),
            "match_mode": row_match_mode,
            "probabilities": sentence.get("probabilities") or {},
            "similarities": sentence.get("similarities") or {},
            "assignment_source": sentence.get("assignment_source"),
        })

    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }
    if not profile_map:
        profile_map = {
            profile: counter.most_common(1)[0][0]
            for profile, counter in profile_speaker_counts.items()
            if counter
        }

    scored_count = 0
    correct_count = 0
    total_duration = 0.0
    correct_duration = 0.0
    unknown_count = 0
    for row in rows:
        assigned = row.get("assigned_speaker")
        if not assigned:
            unknown_count += 1
        canonical_speaker = row.get("canonical_speaker")
        if not canonical_speaker:
            row["mapped_speaker"] = None
            row["matches_canonical"] = False
            continue
        mapped = profile_map.get(str(assigned)) if assigned else None
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == canonical_speaker)
        scored_count += 1
        duration = float(row.get("duration_seconds") or 0.0)
        total_duration += duration
        if row["matches_canonical"]:
            correct_count += 1
            correct_duration += duration

    return {
        "match_mode": match_mode,
        "final_segments": len(finals),
        "resolved_segments": len(sentences),
        "timestamped_segments": timestamped_rows,
        "text_matched_segments": text_matched_rows,
        "unknown_segments": unknown_count,
        "canonical_words": len(canonical_tokens),
        "live_final_words": len(live_tokens),
        "lcs_words": common_tokens,
        "text_recall": round(common_tokens / max(1, len(canonical_tokens)), 4),
        "text_precision": round(common_tokens / max(1, len(live_tokens)), 4),
        "profile_map": profile_map,
        "assigned_counts": dict(Counter(str(row.get("assigned_speaker") or "UNKNOWN") for row in rows)),
        "segment_accuracy": round(correct_count / max(1, scored_count), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "rows": rows,
    }


def analyze_trace(
    path: Path,
    canonical_path: Path | None = None,
    output_path: Path | None = None,
    summary_only: bool = False,
    match_mode: str = "auto",
) -> int:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No trace records found in {path}")
        return 1

    start = records[0].get("time", 0)
    print(f"Trace: {path}")
    print(f"Records: {len(records)}")
    if canonical_path is not None and canonical_path.exists():
        canonical_segments = read_canonical_segments(canonical_path)
        summary = analyze_trace_against_canonical(
            records,
            canonical_segments,
            match_mode=match_mode,
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Trace analysis output: {output_path}")
        print(f"Match mode: {summary['match_mode']}")
        print(f"Final segments: {summary['final_segments']}")
        print(f"Resolved segments: {summary['resolved_segments']}")
        print(f"Timestamped segments: {summary['timestamped_segments']}")
        print(f"Text-matched fallback segments: {summary['text_matched_segments']}")
        print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}")
        print(f"Text recall/precision by LCS: {summary['text_recall']:.3f} / {summary['text_precision']:.3f}")
        print(f"Assigned counts: {summary['assigned_counts']}")
        print(f"Profile map: {summary['profile_map']}")
        print(f"Unknown segments: {summary['unknown_segments']}")
        print(f"Live segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}")
        print(f"Live duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}")
        if summary_only:
            return 0
    elif summary_only:
        return 0
    for record in records:
        event = record.get("event")
        payload = record.get("payload") or {}
        if event not in {"sentence", "final", "realtime", "capture-ready", "error-status"}:
            continue
        elapsed = float(record.get("time", start) or start) - float(start or 0)
        if event == "sentence":
            video_range = ""
            if payload.get("video_start_seconds") is not None:
                video_range = (
                    f" video={payload.get('video_start_seconds')}-"
                    f"{payload.get('video_end_seconds')}"
                )
            print(
                f"{elapsed:8.3f}s sentence "
                f"idx={payload.get('index')} "
                f"speaker={payload.get('assigned_speaker')} "
                f"new={payload.get('created_speaker')} "
                f"unknown={payload.get('unknown_probability')} "
                f"{video_range} "
                f"text={(payload.get('text') or '')[:120]!r}"
            )
        elif event == "final":
            video_range = ""
            if payload.get("video_start_seconds") is not None:
                video_range = (
                    f" video={payload.get('video_start_seconds')}-"
                    f"{payload.get('video_end_seconds')}"
                )
            print(
                f"{elapsed:8.3f}s final "
                f"idx={payload.get('index')} "
                f"{video_range} "
                f"text={(payload.get('text') or '')[:120]!r}"
            )
        else:
            print(f"{elapsed:8.3f}s {event} {payload}")
    return 0


def validate_cunk_realtime_replay(args: argparse.Namespace) -> int:
    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            ROOT
            / "tools"
            / ".realtime_speakerdiarize_traces"
            / f"trace-replay-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    controller = YouTubeWasapiController(args, bus)
    session_id = "replay"
    with controller._lock:
        controller._session_id = session_id
        controller._stop_event = threading.Event()
        controller._video_ids[session_id] = "local-replay"
        controller._sentence_indices[session_id] = 0
    controller.speaker_engine.start_session(session_id)

    audio, sample_rate = load_audio_file(args.validation_audio)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX_ABS_VALUE).astype(np.int16)
    chunk_samples = max(1, int(sample_rate * args.replay_chunk_seconds))
    replay_speed = max(0.1, float(args.replay_speed))
    stop_event = threading.Event()
    recorder = None
    final_thread = None
    started_at = time.monotonic()
    try:
        recorder = controller._create_recorder(session_id, None)
        recorder.start()
        controller._set_video_time(session_id, 0.0)
        final_thread = threading.Thread(
            target=controller._consume_final_text,
            args=(session_id, recorder, stop_event),
            name="ReplayFinalConsumer",
            daemon=True,
        )
        final_thread.start()

        controller._status(
            session_id,
            f"Replay feeding {args.validation_audio} at {replay_speed:.1f}x.",
        )
        for start in range(0, len(audio_int16), chunk_samples):
            end = min(len(audio_int16), start + chunk_samples)
            chunk = audio_int16[start:end]
            recorder.feed_audio(chunk, original_sample_rate=sample_rate)
            controller._set_video_time(session_id, end / float(sample_rate))
            if args.replay_sleep:
                time.sleep((len(chunk) / float(sample_rate)) / replay_speed)

        silence_seconds = max(0.5, float(args.replay_trailing_silence_seconds))
        silence = np.zeros(int(sample_rate * silence_seconds), dtype=np.int16)
        for start in range(0, len(silence), chunk_samples):
            end = min(len(silence), start + chunk_samples)
            chunk = silence[start:end]
            recorder.feed_audio(chunk, original_sample_rate=sample_rate)
            controller._set_video_time(
                session_id,
                (len(audio_int16) + end) / float(sample_rate),
            )
            if args.replay_sleep:
                time.sleep((len(chunk) / float(sample_rate)) / replay_speed)

        controller._status(session_id, "Replay audio feed complete; draining final transcripts.")
        time.sleep(max(0.0, float(args.replay_drain_seconds)))
    finally:
        stop_event.set()
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass
        if final_thread is not None:
            final_thread.join(timeout=3.0)
        drain_deadline = time.monotonic() + max(0.0, float(args.replay_embedding_drain_seconds))
        while (
            getattr(controller.speaker_engine.jobs, "unfinished_tasks", 0) > 0
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.1)
        controller.speaker_engine.shutdown()

    elapsed = time.monotonic() - started_at
    records = []
    with trace.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    canonical_segments = read_canonical_segments(args.validation_canonical)
    analysis_match_mode = "timestamp" if abs(replay_speed - 1.0) < 0.05 else "text"
    summary = analyze_trace_against_canonical(
        records,
        canonical_segments,
        match_mode=analysis_match_mode,
    )
    summary.update({
        "trace": str(trace.path),
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "elapsed_seconds": round(elapsed, 4),
        "replay_speed": replay_speed,
    })
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Replay trace: {trace.path}", flush=True)
    print(f"Replay analysis output: {args.validation_output}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Match mode: {summary['match_mode']}", flush=True)
    print(f"Final segments: {summary['final_segments']}", flush=True)
    print(f"Resolved segments: {summary['resolved_segments']}", flush=True)
    print(f"Timestamped segments: {summary['timestamped_segments']}", flush=True)
    print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}", flush=True)
    print(
        "Text recall/precision by LCS: "
        f"{summary['text_recall']:.3f} / {summary['text_precision']:.3f}",
        flush=True,
    )
    print(f"Assigned counts: {summary['assigned_counts']}", flush=True)
    print(f"Profile map: {summary['profile_map']}", flush=True)
    print(f"Unknown segments: {summary['unknown_segments']}", flush=True)
    print(f"Live segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Live duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding-only realtime speaker diarization for a YouTube WASAPI capture."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--allow-default-input", action="store_true")

    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--rt-model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--split-marks", default="sentence")
    parser.add_argument("--realtime-processing-pause", type=float, default=0.1)
    parser.add_argument("--post-speech-silence-duration", type=float, default=1.0)
    parser.add_argument("--final-video-latency-seconds", type=float, default=0.8)
    parser.add_argument("--min-length-of-recording", type=float, default=0.0)
    parser.add_argument("--silero-sensitivity", type=float, default=0.05)
    parser.add_argument("--webrtc-sensitivity", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--beam-size-realtime", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Final faster-whisper batch size. 0 uses the regular decoder; "
            "RealtimeSTT's default 16 is faster but dropped words on the Cunk clip."
        ),
    )
    parser.add_argument(
        "--realtime-batch-size",
        type=int,
        default=0,
        help="Realtime-preview faster-whisper batch size. 0 uses the regular decoder.",
    )
    parser.add_argument(
        "--no-final-word-timestamps",
        dest="final_word_timestamps",
        action="store_false",
    )
    parser.add_argument(
        "--no-split-final-transcripts",
        dest="split_final_transcripts",
        action="store_false",
    )
    parser.add_argument("--word-split-gap-seconds", type=float, default=0.32)
    parser.add_argument("--max-timestamp-split-seconds", type=float, default=2.0)
    parser.add_argument("--max-word-timestamp-seconds", type=float, default=1.2)
    parser.add_argument("--min-timestamp-split-words", type=int, default=1)
    parser.add_argument("--split-audio-padding-seconds", type=float, default=0.04)
    parser.add_argument("--split-on-soft-punctuation", action="store_true")

    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-python", type=Path, default=default_embedding_python())
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--same-speaker-similarity", type=float, default=0.45)
    parser.add_argument("--similarity-temperature", type=float, default=0.07)
    parser.add_argument("--speaker-softmax-temperature", type=float, default=0.075)
    parser.add_argument("--new-speaker-threshold", type=float, default=0.58)
    parser.add_argument("--duplicate-profile-similarity", type=float, default=0.40)
    parser.add_argument("--unknown-short-threshold", type=float, default=0.86)
    parser.add_argument("--min-first-speaker-seconds", type=float, default=1.2)
    parser.add_argument("--min-new-speaker-seconds", type=float, default=2.0)
    parser.add_argument("--min-embed-seconds", type=float, default=0.5)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--margin-temperature", type=float, default=0.035)
    parser.add_argument("--update-unknown-max", type=float, default=0.55)
    parser.add_argument(
        "--no-reassign-uncertain-sentences",
        dest="reassign_uncertain_sentences",
        action="store_false",
    )
    parser.add_argument("--reassign-max-seconds", type=float, default=2.2)
    parser.add_argument("--reassign-unknown-min", type=float, default=0.7)
    parser.add_argument("--reassign-unknown-max", type=float, default=0.82)
    parser.add_argument("--reassign-min-similarity", type=float, default=0.42)
    parser.add_argument(
        "--no-context-assign-short-fragments",
        dest="context_assign_short_fragments",
        action="store_false",
        help="Disable nearby-speaker context assignment for very short uncertain fragments.",
    )
    parser.add_argument("--context-assign-max-seconds", type=float, default=1.0)
    parser.add_argument("--context-assign-candidate-unknown-min", type=float, default=0.9)
    parser.add_argument("--context-assign-window", type=int, default=4)
    parser.add_argument("--context-assign-stable-unknown-max", type=float, default=0.7)
    parser.add_argument("--context-assign-stable-min-seconds", type=float, default=0.5)
    parser.add_argument("--context-assign-same-speaker-confidence", type=float, default=0.92)
    parser.add_argument("--context-assign-one-sided-confidence", type=float, default=0.82)
    parser.add_argument(
        "--no-context-assign-one-sided",
        dest="context_assign_one_sided",
        action="store_false",
        help="Only context-assign short fragments when stable speakers on both sides agree.",
    )
    parser.add_argument(
        "--segment-audio-dir",
        type=Path,
        default=ROOT / ".cache" / "realtime_speakerdiarize_segments",
    )
    parser.add_argument("--keep-segment-audio", action="store_true")

    parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help="Optional JSONL trace path for backend/frontend UI events.",
    )
    parser.add_argument(
        "--analyze-trace",
        type=Path,
        default=None,
        help="Analyze a previously captured JSONL trace and exit.",
    )
    parser.add_argument(
        "--trace-analysis-output",
        type=Path,
        default=None,
        help="Write structured trace-vs-canonical analysis JSON when --analyze-trace is used.",
    )
    parser.add_argument("--trace-summary-only", action="store_true")
    parser.add_argument(
        "--trace-match-mode",
        choices=("auto", "timestamp", "text"),
        default="auto",
        help="How --analyze-trace maps trace rows to canonical speakers.",
    )
    parser.add_argument("--validate-cunk", action="store_true")
    parser.add_argument("--validate-cunk-word-splits", action="store_true")
    parser.add_argument("--validate-cunk-realtime-replay", action="store_true")
    parser.add_argument(
        "--validation-audio",
        type=Path,
        default=ROOT / "output_pyannote_cunk" / "cunk_on_earth_clip.mp3",
    )
    parser.add_argument(
        "--validation-canonical",
        type=Path,
        default=ROOT / "output_elevenlabs_cunk" / "cunk_on_earth_clip.canonical_diarization.json",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=ROOT / "tools" / ".realtime_speakerdiarize_validation" / "latest.json",
    )
    parser.add_argument("--mixed-overlap-min-seconds", type=float, default=0.05)
    parser.add_argument("--replay-speed", type=float, default=8.0)
    parser.add_argument("--replay-chunk-seconds", type=float, default=0.1)
    parser.add_argument("--replay-trailing-silence-seconds", type=float, default=2.0)
    parser.add_argument("--replay-drain-seconds", type=float, default=25.0)
    parser.add_argument("--replay-embedding-drain-seconds", type=float, default=15.0)
    parser.add_argument(
        "--no-replay-sleep",
        dest="replay_sleep",
        action="store_false",
        help="Feed replay audio as fast as possible instead of wall-clock pacing.",
    )
    parser.set_defaults(replay_sleep=True)
    parser.add_argument("--embedding-helper", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.embedding_helper:
        return run_embedding_helper(args)
    if args.analyze_trace is not None:
        return analyze_trace(
            args.analyze_trace,
            canonical_path=args.validation_canonical,
            output_path=args.trace_analysis_output,
            summary_only=args.trace_summary_only,
            match_mode=args.trace_match_mode,
        )
    if args.validate_cunk_realtime_replay:
        return validate_cunk_realtime_replay(args)
    if args.validate_cunk_word_splits:
        return validate_cunk_word_splits(args)
    if args.validate_cunk:
        return validate_cunk(args)

    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            Path(__file__).resolve().parent
            / ".realtime_speakerdiarize_traces"
            / f"trace-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    server = GuiServer((args.host, args.port), args, bus, trace)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Serving realtime speaker diarization GUI at {url}", flush=True)
    print(f"Trace log: {trace.path}", flush=True)
    print(f"Embedding helper: {args.embedding_python}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.controller.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
