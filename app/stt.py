import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder whisper.cpp
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# Path ke binary whisper-cli
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli")

# Path ke file model Whisper
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin")

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan whisper.cpp CLI

    Args:
        file_bytes (bytes): Isi file audio
        file_ext (str): Ekstensi file audio, default ".wav"

    Returns:
        str: Teks hasil transkripsi, atau pesan error diawali dengan [ERROR]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Simpan audio sementara
            audio_filename = f"{uuid.uuid4()}{file_ext}"
            audio_path = os.path.join(tmpdir, audio_filename)
            with open(audio_path, "wb") as f:
                f.write(file_bytes)

            # Output tanpa ekstensi, Whisper akan menambahkan .txt otomatis
            output_base = os.path.join(tmpdir, "transcription")

            # Jalankan whisper.cpp CLI
            cmd = [
                WHISPER_BINARY,
                "-m", WHISPER_MODEL_PATH,
                "-f", audio_path,
                "-otxt",
                "-of", output_base
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return f"[ERROR] Whisper failed: {result.stderr.strip()}"

            # Baca hasil transkripsi
            transcription_file = output_base + ".txt"
            if not os.path.exists(transcription_file):
                return "[ERROR] Transcription file not found"

            with open(transcription_file, "r", encoding="utf-8") as result_file:
                return result_file.read().strip()

        except Exception as e:
            return f"[ERROR] Exception occurred during STT: {str(e)}"
