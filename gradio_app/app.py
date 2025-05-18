import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import time
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('voice_chatbot_frontend')

# Path to store chat history
HISTORY_PATH = os.path.join(tempfile.gettempdir(), "voice_chat_history.json")
API_URL = "http://localhost:8000/voice-chat"
REQUEST_TIMEOUT = 300  # Timeout in seconds

# Load existing chat history or create empty one
def load_chat_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return []
    return []

# Save chat history
def save_chat_history(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

# Voice chat function with improved error handling
def voice_chat(audio, history, progress=gr.Progress()):
    if audio is None:
        return None, history, "⚠️ Mohon rekam suara terlebih dahulu"
    
    # Update timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Processing voice request at {timestamp}")
    
    # Add progress updates
    progress(0, desc="Memproses suara...")
    
    try:
        sr, audio_data = audio
        
        # Log audio details for debugging
        logger.info(f"Audio sample rate: {sr}, shape: {audio_data.shape}")
        
        # Save as .wav with unique filename
        audio_filename = f"input_{int(time.time())}.wav"
        audio_path = os.path.join(tempfile.gettempdir(), audio_filename)
        
        scipy.io.wavfile.write(audio_path, sr, audio_data)
        logger.info(f"Saved input audio to: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"Failed to save audio file at {audio_path}")
            return None, history, "⚠️ Gagal menyimpan file audio"
            
        progress(0.3, desc="Mengirim ke server...")
        
        # Send to FastAPI endpoint with timeout
        try:
            logger.info(f"Sending request to {API_URL}")
            with open(audio_path, "rb") as f:
                files = {"file": (audio_filename, f, "audio/wav")}
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=REQUEST_TIMEOUT
                )
            
            logger.info(f"Response status: {response.status_code}, Content length: {len(response.content) if response.content else 0}")
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            error_msg = "🕒 Waktu permintaan habis. Server membutuhkan waktu terlalu lama untuk merespons."
            return None, history + [[error_msg, None, timestamp]], error_msg
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error")
            error_msg = "🔌 Tidak dapat terhubung ke server. Pastikan server berjalan di http://localhost:8000"
            return None, history + [[error_msg, None, timestamp]], error_msg
            
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            error_msg = f"🔴 Error: {str(e)}"
            return None, history + [[error_msg, None, timestamp]], error_msg
        
        progress(0.7, desc="Mendapatkan balasan...")
        
        if response.status_code == 200:
            logger.info("Request successful, processing response")
            
            # Verify content type and length
            content_type = response.headers.get('Content-Type', '')
            logger.info(f"Response Content-Type: {content_type}")
            
            if not response.content:
                logger.error("Response content is empty")
                error_msg = "⚠️ Server mengembalikan respons kosong"
                return None, history + [[error_msg, None, timestamp]], error_msg
            
            # Save response audio with unique timestamp to avoid caching issues
            output_audio_path = os.path.join(tempfile.gettempdir(), f"tts_output_{int(time.time())}.wav")
            
            try:
                with open(output_audio_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved response audio to: {output_audio_path}")
                
                # Verify if file exists and has content
                if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
                    logger.error(f"Output file doesn't exist or is empty: {output_audio_path}")
                    error_msg = "⚠️ File audio respons kosong atau tidak valid"
                    return None, history + [[error_msg, None, timestamp]], error_msg
                
            except Exception as e:
                logger.error(f"Failed to save response audio: {e}")
                error_msg = f"⚠️ Gagal menyimpan file audio respons: {str(e)}"
                return None, history + [[error_msg, None, timestamp]], error_msg
            
            # Add successful interaction to history
            user_message = "🎤 Pesan Suara"
            ai_message = "🔊 Balasan Suara"
            new_history = history + [[user_message, ai_message, timestamp]]
            save_chat_history(new_history)
            
            progress(1.0, desc="Selesai!")
            return output_audio_path, new_history, "✅ Berhasil mendapatkan respons"
        else:
            logger.error(f"Server returned error status: {response.status_code}")
            try:
                error_content = response.json() if response.content else {}
                error_detail = error_content.get('message', f"Kode status: {response.status_code}")
            except:
                error_detail = f"Kode status: {response.status_code}"
                
            error_msg = f"⚠️ Server Error: {error_detail}"
            return None, history + [[error_msg, None, timestamp]], error_msg
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        error_msg = f"⚠️ Terjadi kesalahan: {str(e)}"
        return None, history + [[error_msg, None, timestamp]], error_msg

# Clear history function
def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    return [], "🗑️ Riwayat percakapan telah dihapus"

# Format chat history for display with simpler styling
def format_chat_history(history):
    if not history:
        return "<div class='empty-history'>Belum ada percakapan. Mulai dengan merekam suara Anda.</div>"
        
    html = "<div class='chat-container'>"
    for entry in history:
        user_msg, ai_msg, timestamp = entry
        
        # User message
        html += f"""
        <div class="chat-row">
            <div class="chat-bubble user-bubble">
                <div class="chat-content">
                    <div class="chat-icon">👤</div>
                    <div class="chat-message">{user_msg}</div>
                </div>
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """
        
        # AI message if exists
        if ai_msg:
            html += f"""
            <div class="chat-row">
                <div class="chat-bubble assistant-bubble">
                    <div class="chat-content">
                        <div class="chat-icon">🤖</div>
                        <div class="chat-message">{ai_msg}</div>
                    </div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
            """
    
    html += "</div>"
    return html

# Simplified CSS
custom_css = """
body {
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    background-color: #f5f5f5;
    color: #333;
    margin: 0;
    padding: 0;
    line-height: 1.5;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

/* Simplified Header */
.header {
    background-color: #4b6cb7;
    padding: 20px;
    border-radius: 0 0 10px 10px;
    margin-bottom: 20px;
    text-align: center;
}

.app-title {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    color: white;
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1rem;
    margin-top: 10px;
}

/* Basic panel styling */
.panel {
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.panel-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}

.panel-icon {
    margin-right: 10px;
    font-size: 1.2rem;
}

.panel-title {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
}

/* Chat styling */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 8px;
    background-color: #fafafa;
    border: 1px solid #eee;
}

.chat-row {
    margin-bottom: 15px;
    display: flex;
    flex-direction: column;
}

.chat-bubble {
    padding: 10px 15px;
    max-width: 85%;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.user-bubble {
    background-color: #3b5998;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
}

.assistant-bubble {
    background-color: #f0f0f0;
    color: #333;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}

.chat-content {
    display: flex;
    align-items: flex-start;
}

.chat-icon {
    margin-right: 8px;
}

.timestamp {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 5px;
    text-align: right;
}

.empty-history {
    text-align: center;
    color: #777;
    padding: 20px;
    font-style: italic;
    background-color: #fafafa;
    border-radius: 8px;
    border: 1px dashed #ddd;
}

/* Recording indicator */
.recording-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    margin: 10px 0;
    border-radius: 8px;
    background-color: #fafafa;
    border: 1px solid #eee;
}

.pulse-recording {
    display: flex;
    align-items: center;
    color: #e53935;
    font-weight: 600;
}

.record-icon {
    margin-right: 10px;
}

/* Status message */
.status-message {
    margin-top: 10px;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    font-size: 0.9rem;
}

.status-error {
    background-color: #ffebee;
    color: #c62828;
    border-left: 3px solid #c62828;
}

.status-success {
    background-color: #e8f5e9;
    color: #2e7d32;
    border-left: 3px solid #2e7d32;
}

.status-warning {
    background-color: #fff8e1;
    color: #f9a825;
    border-left: 3px solid #f9a825;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 20px;
    padding: 15px;
    color: #777;
    background-color: #f0f0f0;
    border-radius: 8px 8px 0 0;
}

/* Responsive */
@media (max-width: 768px) {
    .app-title {
        font-size: 1.5rem;
    }
    
    .panel {
        padding: 10px;
    }
    
    .chat-bubble {
        max-width: 95%;
    }
}
"""

# Create a simplified theme
theme = gr.themes.Base().set(
    body_background_fill="#f5f5f5",
    body_text_color="#333",
    block_background_fill="#FFFFFF",
    block_border_color="#e0e0e0",
    input_background_fill="#FFFFFF",
    button_primary_background_fill="#4b6cb7",
    button_primary_background_fill_hover="#3b5998",
    button_primary_text_color="#FFFFFF"
)

# Recording indicator state function
def recording_state(recording=False):
    if recording:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# UI with Gradio Blocks - Simplified version
with gr.Blocks(theme=theme, css=custom_css) as demo:
    # Initialize state
    history_state = gr.State(load_chat_history())
    
    # Simple Header
    gr.HTML("""
    <div class="header">
        <h1 class="app-title">AI-Speech Response App</h1>
        <p class="app-subtitle">Asisten Suara Bahasa Indonesia</p>
    </div>
    """)
    
    # Main content area
    with gr.Row():
        # Left column for chat history
        with gr.Column(scale=3):
            with gr.Group(elem_classes="panel"):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">💬</div>
                    <h2 class="panel-title">Riwayat Percakapan</h2>
                </div>
                """)
                # Chat history display
                chat_display = gr.HTML(elem_classes="chat-history")
        
        # Right column for input and output
        with gr.Column(scale=2):
            # Voice input panel
            with gr.Group(elem_classes="panel"):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">🎤</div>
                    <h2 class="panel-title">Rekam Suara</h2>
                </div>
                """)
                
                # Recording indicators
                with gr.Group(elem_classes="recording-indicator"):
                    ready_indicator = gr.HTML("""
                    <div>
                        <span class="record-icon">⚪</span> Siap merekam
                    </div>
                    """, visible=True)
                    
                    recording_active = gr.HTML("""
                    <div class="pulse-recording">
                        <span class="record-icon">🔴</span> Sedang merekam...
                    </div>
                    """, visible=False)
                
                # Audio input with microphone
                audio_input = gr.Audio(
                    sources="microphone",
                    type="numpy",
                    elem_id="voice-input",
                    streaming=False
                )
                
                # Buttons
                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Hapus Riwayat", 
                        variant="secondary"
                    )
                    submit_btn = gr.Button(
                        "🚀 Kirim", 
                        variant="primary"
                    )
                
                # Status message display
                status_msg = gr.HTML(
                    """<div class="status-message">Siap menerima pertanyaan</div>"""
                )
            
            # Voice output panel
            with gr.Group(elem_classes="panel"):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">🔊</div>
                    <h2 class="panel-title">Balasan Asisten</h2>
                </div>
                """)
                
                # Audio output
                audio_output = gr.Audio(
                    type="filepath",
                    elem_id="voice-output",
                    show_label=False
                )
    
    # Simple Footer
    gr.HTML("""
    <div class="footer">
        <p>SUARA AI © 2025 - Platform AI Berbasis Suara Bahasa Indonesia</p>
    </div>
    """)
    
    # Define event handlers
    def update_status(message, is_error=False, is_warning=False):
        if is_error:
            return f'<div class="status-message status-error">{message}</div>'
        elif is_warning:
            return f'<div class="status-message status-warning">{message}</div>'
        else:
            return f'<div class="status-message status-success">{message}</div>'
    
    # Recording start event
    audio_input.start_recording(
        fn=lambda: recording_state(True),
        outputs=[recording_active, ready_indicator]
    )
    
    # Recording stop event
    audio_input.stop_recording(
        fn=lambda: recording_state(False),
        outputs=[recording_active, ready_indicator]
    )
    
    # Submit button click
    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, history_state],
        outputs=[audio_output, history_state, status_msg]
    ).then(
        fn=format_chat_history,
        inputs=[history_state],
        outputs=[chat_display]
    )
    
    # Clear history button
    clear_btn.click(
        fn=clear_history,
        outputs=[history_state, status_msg]
    ).then(
        fn=format_chat_history,
        inputs=[history_state],
        outputs=[chat_display]
    )
    
    # Load history on start
    demo.load(
        fn=format_chat_history,
        inputs=[history_state],
        outputs=[chat_display]
    )

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Voice Chatbot Frontend")
    demo.launch()