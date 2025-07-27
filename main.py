# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import librosa
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import threading
import time
import requests
import soundfile as sf
# --- Load Models ---
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import webview

# Get current directory for proper path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)

# Create a writable temp directory
TEMP_DIR = os.path.join(CURRENT_DIR, "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)
print(f"📁 Audio temp dir: {TEMP_DIR}")

# --- 1. Whisper Tiny (STT) ---
print("Loading Whisper Tiny...")
stt_model = whisper.load_model("tiny")

# --- 2. TinyLlama (LLM) ---
print("Loading TinyLlama...")
llm_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# --- 3. SpeechT5 + HiFiGAN (TTS) ---
print("Loading SpeechT5 TTS and HiFiGAN...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(llm_model.device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(llm_model.device)

# Dummy speaker embedding
speaker_embeddings = torch.zeros((1, 512)).to(tts_model.device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    input_path = os.path.join(TEMP_DIR, "input.wav")
    audio_file.save(input_path)

    # Load and resample
    audio, sr = librosa.load(input_path, sr=16000)
    audio = librosa.util.normalize(audio)
    normalized_path = os.path.join(TEMP_DIR, "input_norm.wav")
    sf.write(normalized_path, audio, sr)

    # Transcribe
    result = stt_model.transcribe(os.path.join(TEMP_DIR, "input_norm.wav"))
    text = result["text"].strip()
    print(f"✅ Transcribed: {text}")
    return jsonify({"text": text})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_input = data.get("text", "").strip()
    if not user_input:
        return jsonify({"reply": "I didn't catch that. Could you repeat?"})

    prompt = f"<|system|>\nYou are a joyful, proud AI assistant.</|system|>\n<|user|>\n{user_input}</|user|>\n<|assistant|>\n"
    response = llm_pipeline(prompt)[0]["generated_text"]

    try:
        reply = response.split("<|assistant|>")[1]
        if "<|" in reply:
            reply = reply.split("<|")[0]
        reply = reply.strip()
    except:
        reply = response[len(prompt):].strip()

    print(f"💬 AI Response: {reply}")
    return jsonify({"reply": reply})

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text for TTS"}), 400

    inputs = processor(text=text, return_tensors="pt").to(tts_model.device)
    with torch.no_grad():
        spectrogram = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        waveform = vocoder(spectrogram.unsqueeze(0)).cpu().numpy().squeeze()

    output_path = os.path.join(TEMP_DIR, "output.wav")
    wavfile.write(output_path, rate=16000, data=np.int16(waveform * 32767))
    print(f"🔊 Audio saved: {output_path}")
    return jsonify({"audio_url": "/audio/output.wav"})

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(TEMP_DIR, filename)

if __name__ == "__main__":
    print(f"🌈 Pride AI Assistant is starting...")
    
    # Start Flask app in a separate thread
    def run_flask():
        try:
            app.run(debug=False, host="127.0.0.1", port=8181, use_reloader=False)
        except Exception as e:
            print(f"❌ Flask error: {e}")
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Give Flask time to start
    print("⏳ Starting Flask server...")
    time.sleep(3)
    
    # Test if Flask is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:8181", timeout=5)
        print("✅ Flask server is running")
    except Exception as e:
        print(f"⚠️ Flask server check failed: {e}")
        print("🔄 Continuing anyway...")
    
    print(f"🌈 Opening Pride AI Assistant in desktop window...")
    try:
        # Create webview window
        webview.create_window(
            title="🎙️ Voice AI Assistant", 
            url="http://127.0.0.1:8181",
            width=900,
            height=800,
            resizable=True,
            min_size=(700, 600),
            maximized=False
        )
        webview.start(debug=True)
    except Exception as e:
        print(f"❌ Webview error: {e}")
        print("🌐 You can access the app at: http://127.0.0.1:8181")
        # Keep Flask running if webview fails
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("👋 Shutting down...")