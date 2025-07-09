#!/usr/bin/env python3
"""
Kyutai PyTorch-TTS Flask app with voice labeling
────────────────────────────────────────────────────────────────────────
• GET  /             – HTML UI (text box + voice dropdown + label editor)
• GET  /voices       – JSON list of local *.wav voices
• GET  /labels       – JSON mapping voice→label
• POST /labels      – {"voice":"...","label":"..."} → save label
• POST /synthesize   – JSON {"text":"...","voice":"...","fmt":"wav|ogg"} → WAV/OGG
────────────────────────────────────────────────────────────────────────
Dependencies:
  pip install flask torch sphn moshi==0.2.8
  system: ffmpeg (for OGG)
"""
import os, io, json, wave, subprocess, threading, traceback
from pathlib import Path

import numpy as np
import torch
import sphn
from flask import (
    Flask, request, jsonify, send_file, render_template_string
)
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

# ─────────── CONFIG ───────────
HF_TTS_REPO = os.getenv("HF_TTS_REPO", DEFAULT_DSM_TTS_REPO)
VOICES_ROOT = Path(os.getenv("VOICES_ROOT", "/root/tts-voices")).expanduser()
LABELS_FILE = VOICES_ROOT / "voice_labels.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────

app = Flask(__name__)

# discover voices (*.wav)
VOICE_LIST = sorted(p.relative_to(VOICES_ROOT).as_posix()
                    for p in VOICES_ROOT.rglob("*.wav"))
if not VOICE_LIST:
    raise RuntimeError(f"No *.wav voices in {VOICES_ROOT}")
DEFAULT_VOICE = VOICE_LIST[0]

# load or init labels
if LABELS_FILE.exists():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = {}
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

# load model once
print(f"[init] loading TTS model {HF_TTS_REPO} on {DEVICE} …")
ckpt      = CheckpointInfo.from_hf_repo(HF_TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=DEVICE)
SR = tts_model.mimi.sample_rate
print(f"[init] model ready (SR={SR})")

MIMI_LOCK = threading.Lock()

def pcm_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    pcm_i16 = (pcm * 32767).astype(np.int16)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()

def wav_to_ogg_bytes(wav_bytes: bytes) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-ac", "1", "-ar", "48000",
        "-c:a", "libopus", "-f", "ogg", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, check=True)
    return proc.stdout

# ╔════════ Routes ═════════╗
@app.get("/voices")
def get_voices():
    return jsonify(VOICE_LIST)

@app.get("/labels")
def get_labels():
    return jsonify(labels)

@app.post("/labels")
def set_label():
    data = request.get_json(force=True)
    voice = data.get("voice")
    label = data.get("label", "").strip()
    if voice not in VOICE_LIST:
        return jsonify(error="Unknown voice"), 400
    labels[voice] = label
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    return jsonify({voice: label})

@app.get("/")
def index():
    voices_js = json.dumps(VOICE_LIST)
    labels_js = json.dumps(labels)
    return render_template_string(f"""
<!doctype html><html><head><title>TTS Demo</title></head>
<body style="font-family:sans-serif">
<h3>Kyutai TTS with Labels</h3>
<textarea id="txt" rows="4" cols="60">Hello world!</textarea><br>
<select id="voice"></select>
<button onclick="editLabel()">Edit Label</button><br>
<label>New label: <input id="labelInput" /></label>
<button onclick="saveLabel()">Save Label</button>
<br><br>
<button onclick="speak('wav')">Speak WAV</button>
<button onclick="speak('ogg')">Speak OGG</button>
<button onclick="downloadFile()">Download</button>
<button onclick="copyClipboard()">Copy to Clipboard</button>
<audio id="player" controls style="display:block;margin-top:1em"></audio>

<script>
const voices = {voices_js};
let labels = {labels_js};
const sel = document.getElementById("voice");
const lblInput = document.getElementById("labelInput");
function refreshDropdown(){{
  sel.innerHTML = "";
  voices.forEach(v => {{
    const display = labels[v] ? labels[v] + " (" + v + ")" : v;
    const o = document.createElement("option");
    o.value = o.text = display;
    o.dataset.voice = v;
    sel.appendChild(o);
  }});
}}
refreshDropdown();

function editLabel(){{
  const v = sel.selectedOptions[0].dataset.voice;
  lblInput.value = labels[v] || "";
}}

async function saveLabel(){{
  const opt = sel.selectedOptions[0];
  const voice = opt.dataset.voice;
  const label = lblInput.value.trim();
  const res = await fetch("/labels", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{voice, label}})
  }});
  if (!res.ok) return alert(await res.text());
  labels[voice] = label;
  refreshDropdown();
}}

let lastBlob=null, lastFmt='wav';
async function speak(fmt){{
  lastFmt = fmt;
  const opt = sel.selectedOptions[0];
  const voice = opt.dataset.voice;
  const res = await fetch("/synthesize?fmt=" + fmt, {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{
      text: txt.value,
      voice: voice
    }})
  }});
  if (!res.ok) return alert(await res.text());
  lastBlob = await res.blob();
  player.src = URL.createObjectURL(lastBlob);
  player.play();
}}

function downloadFile(){{
  if (!lastBlob) return alert("Generate audio first");
  const url = URL.createObjectURL(lastBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "speech." + (lastFmt==='ogg'?'ogg':'wav');
  document.body.appendChild(a);
  a.click();
  a.remove();
}}

async function copyClipboard(){{
  if (!lastBlob) return alert("Generate audio first");
  try {{
    const item = new ClipboardItem({{[lastBlob.type]: lastBlob}});
    await navigator.clipboard.write([item]);
    alert("Audio copied to clipboard!");
  }} catch(e) {{
    alert("Clipboard API not supported or denied; downloading instead");
    downloadFile();
  }}
}}
</script>
</body></html>
""")

@app.post("/synthesize")
def synthesize():
    fmt = request.args.get("fmt", "wav")
    try:
        data = request.get_json(force=True)
    except:
        return jsonify(error="invalid JSON"), 400

    text = str(data.get("text","")).strip()
    voice = str(data.get("voice", DEFAULT_VOICE)).strip()
    if not text:
        return jsonify(error="empty text"), 400
    if voice not in VOICE_LIST:
        return jsonify(error="voice not found"), 400

    wav_path = VOICES_ROOT / voice
    if not wav_path.is_file():
        return jsonify(error="wav missing"), 500

    try:
        with torch.no_grad():
            embed   = tts_model.get_voice_path(str(wav_path))
            entries = tts_model.prepare_script([text], padding_between=1)
            cond    = tts_model.make_condition_attributes([embed], cfg_coef=2.0)
            result  = tts_model.generate([entries], [cond])

        with MIMI_LOCK, torch.no_grad(), tts_model.mimi.streaming(1):
            pcm = np.concatenate([
                tts_model.mimi.decode(fr[:,1:,:]).cpu().numpy()[0,0]
                for fr in result.frames[tts_model.delay_steps:]
            ]).astype(np.float32)

        peak = np.max(np.abs(pcm))
        if peak > 1.0:
            pcm *= 0.95 / peak

        wav_bytes = pcm_to_wav_bytes(pcm, SR)
        if fmt == "ogg":
            ogg_bytes = wav_to_ogg_bytes(wav_bytes)
            return send_file(
                io.BytesIO(ogg_bytes),
                mimetype="audio/ogg",
                as_attachment=True,
                download_name="speech.ogg"
            )
        else:
            return send_file(
                io.BytesIO(wav_bytes),
                mimetype="audio/wav",
                as_attachment=True,
                download_name="speech.wav"
            )

    except subprocess.CalledProcessError:
        traceback.print_exc()
        return jsonify(error="ffmpeg failed"), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
