#!/usr/bin/env python3
"""
Kyutai PyTorch-TTS Flask app — Improved WAV quality
────────────────────────────────────────────────────────────────────────
• GET  /            – HTML UI (text box + voice dropdown)
• GET  /voices      – JSON list of local *.wav voices
• POST /synthesize  – JSON {"text": "...", "voice": "..."} → 16-bit WAV
Produces high-quality WAV with improved sampling and normalization.
"""

import os
import io
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from flask import (
    Flask, request, jsonify, send_file, render_template_string
)

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

# ───────── CONFIG ─────────
HF_TTS_REPO = os.getenv("HF_TTS_REPO", DEFAULT_DSM_TTS_REPO)
VOICES_ROOT = Path(os.getenv("VOICES_ROOT", "/root/tts-voices")).expanduser()
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "false").lower() == "true"
# ──────────────────────────

app = Flask(__name__)

# ─── discover voices (.wav) ───
VOICE_LIST = sorted(p.relative_to(VOICES_ROOT).as_posix()
                    for p in VOICES_ROOT.rglob("*.wav"))
if not VOICE_LIST:
    raise RuntimeError(f"No *.wav voices in {VOICES_ROOT}")
DEFAULT_VOICE = VOICE_LIST[0]

# ─── load model once ───
print(f"[init] loading {HF_TTS_REPO} on {DEVICE} …")
ckpt      = CheckpointInfo.from_hf_repo(HF_TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=DEVICE)
print("[init] model ready.")
SR = tts_model.mimi.sample_rate
print(f"[init] Model sample rate: {SR}")
print(f"[init] Expected sample rate: {tts_model.mimi.sample_rate}")
# ─────────────────────────

# ╔════════ ROUTES ════════╗
@app.get("/voices")
def voices(): 
    return jsonify(VOICE_LIST)

@app.get("/")
def index():
    return render_template_string(
f"""<!doctype html><html><head><title>TTS - Improved Audio Quality</title></head><body>
<h1>Text-to-Speech Generator</h1>
<div style="margin: 20px 0;">
    <label for="txt">Text to synthesize:</label><br>
    <textarea id="txt" rows="4" cols="60" placeholder="Enter your text here...">Hello world!</textarea>
</div>
<div style="margin: 20px 0;">
    <label for="voice">Voice:</label><br>
    <select id="voice">{"".join(f'<option value="{v}">{v}</option>' for v in VOICE_LIST)}</select>
</div>
<button onclick="speak()" style="padding: 10px 20px; font-size: 16px;">🔊 Speak</button>
<div id="status" style="margin: 10px 0; font-style: italic;"></div>
<audio id="player" controls style="display:block;margin-top:1em;width:100%;"></audio>
<script>
async function speak() {{
    const statusDiv = document.getElementById('status');
    const player = document.getElementById('player');
    const text = document.getElementById('txt').value.trim();
    const voice = document.getElementById('voice').value;
    
    if (!text) {{
        alert('Please enter some text to synthesize.');
        return;
    }}
    
    statusDiv.textContent = 'Generating speech...';
    
    try {{
        const response = await fetch("/synthesize", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{text: text, voice: voice}})
        }});
        
        if (!response.ok) {{
            const error = await response.text();
            alert('Error: ' + error);
            statusDiv.textContent = 'Error occurred.';
            return;
        }}
        
        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        player.src = audioUrl;
        player.play();
        statusDiv.textContent = 'Speech generated successfully!';
        
        // Clean up old URLs
        setTimeout(() => URL.revokeObjectURL(audioUrl), 60000);
        
    }} catch (error) {{
        alert('Network error: ' + error.message);
        statusDiv.textContent = 'Network error occurred.';
    }}
}}

// Auto-focus on text area
document.getElementById('txt').focus();
</script>
</body></html>"""
    )

@app.post("/synthesize")
def synthesize():
    """Synthesize speech with improved audio quality"""
    # Validate input
    try: 
        data = request.get_json(force=True)
    except: 
        return jsonify(error="Bad JSON format"), 400
    
    text = str(data.get("text", "")).strip()
    voice = str(data.get("voice", DEFAULT_VOICE)).strip()
    
    if not text:  
        return jsonify(error="Empty text provided"), 400
    if voice not in VOICE_LIST: 
        return jsonify(error="Voice not found in available voices"), 400
    
    wav_path = VOICES_ROOT / voice
    if not wav_path.is_file(): 
        return jsonify(error="Voice file missing from filesystem"), 500

    try:
        print(f"[synthesis] Processing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"[synthesis] Using voice: {voice}")
        
        with torch.no_grad():
            # Load voice embedding
            embed = tts_model.get_voice_path(str(wav_path))
            
            # Prepare text
            entries = tts_model.prepare_script([text], padding_between=1)
            
            # Create conditioning
            cond = tts_model.make_condition_attributes([embed], cfg_coef=2.0)
            
            # Generate audio
            result = tts_model.generate([entries], [cond])
            
            print(f"[synthesis] Generated {len(result.frames)} frames")
            print(f"[synthesis] Delay steps: {tts_model.delay_steps}")

            # Improved decode logic - collect all PCM chunks
            with tts_model.mimi.streaming(1):
                pcms = []
                for i, fr in enumerate(result.frames[tts_model.delay_steps:]):
                    pcm = tts_model.mimi.decode(fr[:,1:,:]).cpu().numpy()[0,0]
                    # Don't apply aggressive clipping here
                    pcms.append(pcm)
                    
                    if DEBUG_AUDIO and i < 3:  # Debug first few frames
                        print(f"[debug] Frame {i}: shape={pcm.shape}, range=[{pcm.min():.4f}, {pcm.max():.4f}]")
                        
            if not pcms: 
                raise RuntimeError("No audio frames generated")
            
            # Concatenate all PCM data
            pcm_all = np.concatenate(pcms, axis=-1)
            
            # Audio quality improvements
            print(f"[synthesis] Raw PCM: {pcm_all.shape} samples, range=[{pcm_all.min():.4f}, {pcm_all.max():.4f}]")
            
            # Apply soft normalization instead of hard clipping
            max_amplitude = np.max(np.abs(pcm_all))
            if max_amplitude > 1.0:
                # Normalize to 95% to prevent clipping artifacts
                normalization_factor = 0.95 / max_amplitude
                pcm_all = pcm_all * normalization_factor
                print(f"[synthesis] Normalized by factor {normalization_factor:.4f}")
            elif max_amplitude < 0.1:
                # Boost very quiet audio
                boost_factor = 0.5 / max_amplitude
                pcm_all = pcm_all * boost_factor
                print(f"[synthesis] Boosted by factor {boost_factor:.4f}")
            
            # Final audio stats
            duration = len(pcm_all) / SR
            print(f"[synthesis] Final PCM: {len(pcm_all)} samples, {duration:.2f}s duration")
            print(f"[synthesis] Final range: [{pcm_all.min():.4f}, {pcm_all.max():.4f}]")
            
            # Save debug file if enabled
            if DEBUG_AUDIO:
                debug_filename = f"debug_audio_{int(time.time())}.wav"
                sf.write(debug_filename, pcm_all, SR, subtype='PCM_16')
                print(f"[debug] Saved debug audio to {debug_filename}")

        # Write high-quality WAV using soundfile
        buf = io.BytesIO()
        sf.write(buf, pcm_all, SR, format='WAV', subtype='PCM_16')
        buf.seek(0)
        
        print(f"[synthesis] WAV buffer size: {len(buf.getvalue())} bytes")
        
    except Exception as e:
        print(f"[error] Synthesis failed: {str(e)}")
        traceback.print_exc()
        return jsonify(error=f"Synthesis error: {str(e)}"), 500

    return send_file(
        buf, 
        mimetype="audio/wav",
        as_attachment=False,
        download_name=f"speech_{int(time.time())}.wav"
    )

@app.get("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "device": str(DEVICE),
        "sample_rate": SR,
        "available_voices": len(VOICE_LIST)
    })

@app.get("/info")
def info():
    """System information endpoint"""
    return jsonify({
        "model_repo": HF_TTS_REPO,
        "voices_root": str(VOICES_ROOT),
        "device": str(DEVICE),
        "sample_rate": SR,
        "voice_count": len(VOICE_LIST),
        "default_voice": DEFAULT_VOICE,
        "debug_mode": DEBUG_AUDIO
    })

# ╚════════════════════════╝

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-quality TTS Flask Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--debug-audio", action="store_true", help="Enable audio debugging")
    
    args = parser.parse_args()
    
    if args.debug_audio:
        os.environ["DEBUG_AUDIO"] = "true"
    
    print(f"[startup] Starting T
