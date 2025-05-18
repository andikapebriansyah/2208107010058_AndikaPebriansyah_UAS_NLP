import os
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

# Path untuk file .env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, '.env')

# Coba load menggunakan dotenv
print(f"Loading .env from: {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

MODEL = "gemini-2.0-flash"

# Coba dapatkan API key dari variabel lingkungan
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env. Pastikan file .env berisi GEMINI_API_KEY=your_api_key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Prompt sistem yang digunakan untuk membimbing gaya respons LLM
system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant designed for Indonesian language interaction.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

IMPORTANT: Your response must ONLY be the IPA phonetic transcription of what you would say in Indonesian. Do not include any regular text.

Your answers must:
- Be based on polite and easily understandable Indonesian.
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.
- ONLY output the IPA phonetic transcription, nothing else.

Examples:
User: What's the weather like today?
Assistant: hari ini t͡ʃuat͡ʃaɲa t͡ʃərah di səbaɡian bəsar wilaʝah, dəŋan suhu səkitar tiɡa puluh dərat͡ʃat.

User: Do you know who is the president of Indonesia?
Assistant: prɛsidɛn indonɛsia saʔat ini adalah d͡ʒɔkɔ widɔdɔ.

User: Tell me about Jakarta
Assistant: d͡ʒakarta adalah ibu kɔta indɔnɛsia ʝaŋ tərlɛtak di pulau d͡ʒawa. kɔta ini mərupakan pusat pəmərintahan dəŋan pɔpulasi səkitar səpuluh d͡ʒuta d͡ʒiwa.

If you're unsure about an answer, respond with the IPA transcription of "Maaf, saya tidak tahu jawaban untuk pertanyaan tersebut."
"""

# TODO: Inisialisasi klien Gemini dan konfigurasi prompt
# Gunakan genai.Client(api_key=...) untuk membuat klien.
# Gunakan types.GenerateContentConfig(system_instruction=...) untuk membuat konfigurasi awal.
# Jika ingin melihat contoh implementasi, baca dokumentasi resmi Gemini:
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb
client = genai.Client(api_key=GOOGLE_API_KEY)
chat_config = types.GenerateContentConfig(system_instruction=system_instruction)
history_adapter = TypeAdapter(list[types.Content])

# Fungsi untuk menyimpan/memuat riwayat chat
def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")

def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return client.chats.create(model=MODEL, config=chat_config)
    
    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model=MODEL, config=chat_config)

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return client.chats.create(model=MODEL, config=chat_config)

    try:
        history = history_adapter.validate_json(json_str)
        return client.chats.create(model=MODEL, config=chat_config, history=history)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        return client.chats.create(model=MODEL, config=chat_config)

# Inisialisasi sesi chat saat aplikasi dimulai
chat = load_chat_history()

# Kirim prompt ke LLM dan kembalikan respons teks
def generate_response(prompt: str) -> str:
    try:
        response = chat.send_message(prompt)
        save_chat_history(chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"