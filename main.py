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
from collections import deque
import json
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
    max_new_tokens=400,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# --- 3. SpeechT5 + HiFiGAN (TTS) ---
print("Loading SpeechT5 TTS and HiFiGAN...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(llm_model.device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(llm_model.device)
# print out the directoeries of the models

# Dummy speaker embedding
speaker_embeddings = torch.zeros((1, 512)).to(tts_model.device)

# --- Conversation Context Management ---
# TinyLlama has 2048 token context window, we'll use ~1600 tokens for safety
MAX_CONTEXT_TOKENS = 1600
SYSTEM_PROMPT = "You are a joyful, proud AI assistant called Chatty who celebrates diversity and spreads positivity."

class ConversationManager:
    def __init__(self, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.conversation_history = deque()
        self.system_prompt = SYSTEM_PROMPT
        
    def add_exchange(self, user_input, ai_response):
        """Add a user-AI exchange to the conversation history"""
        exchange = {
            "user": user_input.strip(),
            "assistant": ai_response.strip()
        }
        self.conversation_history.append(exchange)
        self._trim_context()
    
    def _trim_context(self):
        """Trim conversation history to stay within token limit"""
        while len(self.conversation_history) > 1:  # Keep at least 1 exchange if possible
            # Build current context
            context = self._build_context()
            tokens = self.tokenizer.encode(context, return_tensors="pt")
            
            if tokens.shape[1] <= self.max_tokens:
                break
                
            # Remove oldest exchange
            self.conversation_history.popleft()
    
    def _build_context(self):
        """Build the full context string with system prompt and history"""
        context_parts = [f"<|system|>\n{self.system_prompt}<|system|>"]
        
        for exchange in self.conversation_history:
            context_parts.append(f"<|user|>\n{exchange['user']}<|user|>")
            context_parts.append(f"<|assistant|>\n{exchange['assistant']}<|assistant|>")
        
        return "\n".join(context_parts)
    
    def get_prompt_for_input(self, user_input):
        """Get the full prompt including context for a new user input"""
        context = self._build_context()
        return f"{context}\n<|user|>\n{user_input}<|user|>\n<|assistant|>\n"
    
    def get_history_summary(self):
        """Get a summary of the conversation history"""
        return {
            "total_exchanges": len(self.conversation_history),
            "estimated_tokens": len(self.tokenizer.encode(self._build_context(), return_tensors="pt")[0]),
            "recent_topics": [exchange["user"][:50] + "..." if len(exchange["user"]) > 50 else exchange["user"] 
                            for exchange in list(self.conversation_history)[-3:]]
        }
    
    def clear_history(self):
        """Clear all conversation history"""
        self.conversation_history.clear()

# Initialize conversation manager
conversation_manager = ConversationManager(tokenizer)

print(f"🧠 Conversation Context: {MAX_CONTEXT_TOKENS} tokens max, TinyLlama context window: 2048 tokens")
print(f"💭 System Prompt: {SYSTEM_PROMPT}")

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
    print(f"DEBUG: /generate called with user_input: '{user_input}'")  # Add this line
    if not user_input:
        return jsonify({"reply": "I didn't catch that. Could you repeat?"})

    # Get prompt with conversation context
    prompt = conversation_manager.get_prompt_for_input(user_input)
    
    # Generate response
    response = llm_pipeline(prompt)[0]["generated_text"]

    # Extract assistant reply
    try:
        reply = response.split("<|assistant|>")[-1]  # Get the last assistant response
        if "<|" in reply:
            reply = reply.split("<|")[0]
        reply = reply.strip()
    except:
        reply = response[len(prompt):].strip()

    # Add this exchange to conversation history
    conversation_manager.add_exchange(user_input, reply)
    
    # Get conversation stats for debugging
    history_info = conversation_manager.get_history_summary()
    print(f"💬 User: {user_input}")
    print(f"🤖 AI: {reply}")
    print(f"📊 Context: {history_info['total_exchanges']} exchanges, ~{history_info['estimated_tokens']} tokens")
    
    return jsonify({
        "reply": reply,
        "context_info": history_info  # Optional: include in response for debugging
    })


@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text for TTS"}), 400

    max_chunk = 300
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    waveforms = []
    print(f"🔉 TTS: Splitting text into {len(chunks)} chunk(s) of up to {max_chunk} chars each.")
    for idx, chunk in enumerate(chunks):
        print(f"  - Generating audio for chunk {idx+1}/{len(chunks)}: '{chunk[:40]}{'...' if len(chunk)>40 else ''}'")
        inputs = processor(text=chunk, return_tensors="pt").to(tts_model.device)
        with torch.no_grad():
            spectrogram = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings)
            waveform = vocoder(spectrogram.unsqueeze(0)).cpu().numpy().squeeze()
            waveforms.append(waveform)

    # Concatenate all waveforms
    if len(waveforms) == 1:
        final_waveform = waveforms[0]
    else:
        # Pad to same dtype and concatenate
        final_waveform = np.concatenate([np.array(w, dtype=np.float32) for w in waveforms])

    output_path = os.path.join(TEMP_DIR, "output.wav")
    wavfile.write(output_path, rate=16000, data=np.int16(final_waveform * 32767))
    print(f"🔊 Audio saved: {output_path}")
    return jsonify({"audio_url": "/audio/output.wav"})

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(TEMP_DIR, filename)

@app.route("/conversation/history", methods=["GET"])
def get_conversation_history():
    """Get the current conversation history"""
    history = list(conversation_manager.conversation_history)
    summary = conversation_manager.get_history_summary()
    return jsonify({
        "history": history,
        "summary": summary
    })

@app.route("/conversation/clear", methods=["POST"])
def clear_conversation():
    """Clear the conversation history"""
    conversation_manager.clear_history()
    print("🗑️ Conversation history cleared")
    return jsonify({"status": "cleared", "message": "Conversation history has been cleared"})

@app.route("/conversation/context", methods=["GET"])
def get_context_info():
    """Get information about current context usage"""
    summary = conversation_manager.get_history_summary()
    context_text = conversation_manager._build_context()
    return jsonify({
        "summary": summary,
        "max_tokens": MAX_CONTEXT_TOKENS,
        "context_preview": context_text[:1024] + "..." if len(context_text) > 1024 else context_text
    })

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