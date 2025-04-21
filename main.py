import sys
import threading
from typing import Optional
import webbrowser
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import atexit
import json
import socket
import os

# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# --- TRAY APP IMPORTS ---
try:
    import pystray
    from pystray import Menu, MenuItem
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None

# Set your Sonar device here (replace with your device name or index)
sd.default.device = 'SteelSeries Sonar - Aux (SteelS, MME'  # or the correct index

_silent_stream = None

def start_silent_stream(samplerate=96000, channels=8):
    global _silent_stream
    if _silent_stream is not None:
        return
    def callback(outdata, frames, time, status):
        outdata[:] = np.zeros((frames, channels), dtype=np.float32)
    _silent_stream = sd.OutputStream(
        samplerate=samplerate,
        channels=channels,
        dtype='float32',
        callback=callback
    )
    _silent_stream.start()

def stop_silent_stream():
    global _silent_stream
    if _silent_stream is not None:
        _silent_stream.stop()
        _silent_stream.close()
        _silent_stream = None

atexit.register(stop_silent_stream)
start_silent_stream()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/sounds", StaticFiles(directory="sounds"), name="sounds")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sounds_dir = Path("sounds")
sounds_dir.mkdir(exist_ok=True)

FAVOURITES_FILE = sounds_dir / "favourites.json"
ORDER_FILE = sounds_dir / "order.json"

def load_favourites():
    if FAVOURITES_FILE.exists():
        with open(FAVOURITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_favourites(favs):
    with open(FAVOURITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f)

def load_order():
    if ORDER_FILE.exists():
        with open(ORDER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_order(order):
    with open(ORDER_FILE, "w", encoding="utf-8") as f:
        json.dump(order, f)

# Keep track of active players (threads and stop events)
active_players = []

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse("static/index.html")

@app.get("/sounds")
async def list_sounds():
    return [
        f.name
        for f in sounds_dir.glob("*.wav")
        if f.is_file()
    ]

@app.get("/favourites")
async def get_favourites():
    return load_favourites()

@app.post("/favourites")
async def set_favourites(request: Request):
    favs = await request.json()
    save_favourites(favs)
    return {"status": "ok"}

@app.get("/order")
async def get_order():
    return load_order()

@app.post("/order")
async def set_order(request: Request):
    order = await request.json()
    save_order(order)
    return {"status": "ok"}

def play_audio_file(path, stop_event):
    try:
        data, samplerate = sf.read(str(path), dtype='float32')
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        stream = sd.OutputStream(samplerate=samplerate, channels=data.shape[1])
        stream.start()
        blocksize = 1024
        idx = 0
        while idx < len(data) and not stop_event.is_set():
            end_idx = min(idx + blocksize, len(data))
            stream.write(data[idx:end_idx])
            idx = end_idx
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"Error playing sound: {e}")

@app.post("/play")
async def play_sound(request: Request):
    data = await request.json()
    sound_name = data["name"]
    sound_path = sounds_dir / sound_name

    if not sound_path.exists():
        return {"error": "sound not found"}

    stop_event = threading.Event()
    t = threading.Thread(target=play_audio_file, args=(sound_path, stop_event), daemon=True)
    active_players.append((t, stop_event))
    t.start()
    return {"status": "playing"}

TARGET_SR = 96000

@app.post("/upload")
async def upload_sound(file: UploadFile = File(...), image: Optional[UploadFile] = File(None)):
    # Save original file temporarily
    temp_path = sounds_dir / ("_temp_" + file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and resample with pydub
    audio = AudioSegment.from_file(str(temp_path))
    audio = audio.set_frame_rate(TARGET_SR)
    audio = audio.set_channels(2)  # or 1 if you want mono

    # Save as .wav with the original name but .wav extension
    out_name = Path(file.filename).stem + ".wav"
    out_path = sounds_dir / out_name
    audio.export(str(out_path), format="wav")

    temp_path.unlink()  # Remove temp file

    if image:
        image_path = sounds_dir / image.filename
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    return {"status": "uploaded", "filename": out_name}

@app.post("/stop")
async def stop_all():
    # Signal all players to stop
    for t, stop_event in active_players[:]:
        stop_event.set()
        if t.is_alive():
            t.join(timeout=0.1)
        active_players.remove((t, stop_event))
    return {"status": "stopped"}

@app.delete("/delete/{sound_name}")
async def delete_sound(sound_name: str):
    sound_path = sounds_dir / sound_name
    image_path = sounds_dir / (Path(sound_name).stem + ".png")
    if not sound_path.exists():
        raise HTTPException(status_code=404, detail="Sound not found")
    sound_path.unlink()
    if image_path.exists():
        image_path.unlink()
    return {"status": "deleted"}

# --- TRAY APP FUNCTIONALITY ---
def create_tray_icon(url):
    if not pystray:
        print("pystray and pillow are required for tray functionality.")
        return

    def create_image():
        # Load .ico file as tray icon
        try:
            from PIL import Image
            return Image.open("static/soundboard_icon.ico")
        except Exception as e:
            print("Failed to load .ico, using fallback icon:", e)
            img = Image.new('RGB', (64, 64), color=(0, 119, 204))
            d = ImageDraw.Draw(img)
            d.ellipse((8, 8, 56, 56), fill=(79, 195, 247))
            return img

    def on_open(icon, item):
        webbrowser.open(url)

    def on_quit(icon, item):
        icon.stop()
        sys.exit(0)

    menu = Menu(
        MenuItem(f"Open Soundboard ({url})", on_open),
        MenuItem("Quit", on_quit)
    )
    icon = pystray.Icon("Soundboard", create_image(), "Soundboard", menu)
    icon.run()

def get_local_ip():
    """Get the local IP address of the computer."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    import uvicorn
    import time

    HOST = get_local_ip()
    PORT = 8000
    URL = f"http://{HOST}:{PORT}"

    # Start FastAPI server in a thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": HOST, "port": PORT, "log_level": "info"},
        daemon=True
    )
    server_thread.start()

    # Wait for server to start
    time.sleep(1)

    # Start tray icon
    create_tray_icon(URL)