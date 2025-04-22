import sys
import os
import threading
import time
import platform
import multiprocessing
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
import json
import socket
import uvicorn

# Logging imports
import logging
import logging.config
import logging.handlers # For file logging

# Tray icon imports
try:
    import pystray
    from pystray import Menu, MenuItem
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None
    print("Warning: pystray or Pillow not found. Tray icon functionality will be disabled.")
    print("Install them with: pip install pystray Pillow")

# --- Helper function to determine resource paths ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in PyInstaller bundle
        base_path = os.path.abspath(os.path.dirname(__file__)) # Use script's directory

    return os.path.join(base_path, relative_path)

def get_persistent_data_dir():
    """ Get the directory for persistent user data (sounds, config, logs). """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as bundled exe, use directory containing the exe
        return Path(os.path.dirname(sys.executable))
    else:
        # Running as script, use script's directory
        return Path(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
APP_NAME = "Soundboard"
STATIC_DIR = Path(resource_path("static"))
# Persistent data goes next to the executable or script
PERSISTENT_DATA_DIR = get_persistent_data_dir()
SOUNDS_DIR = PERSISTENT_DATA_DIR / "sounds"
SOUNDS_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists, including parents

FAVOURITES_FILE = SOUNDS_DIR / "favourites.json"
ORDER_FILE = SOUNDS_DIR / "order.json"
ICON_FILE = "icon.png" # Assumes icon is in 'static' folder relative to script/bundle
VOLUMES_FILE = SOUNDS_DIR / "volumes.json"
SETTINGS_FILE = PERSISTENT_DATA_DIR / "settings.json"

# --- File Logging Setup ---
LOG_FILE = PERSISTENT_DATA_DIR / "soundboard_app.log"

# Ensure the log directory exists
try:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    # Handle potential permission errors if needed
    print(f"Warning: Could not create log directory {LOG_FILE.parent}: {e}")

# Define the logging configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Keep FastAPI's and others' loggers
    "formatters": {
        "file_formatter": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file_formatter",
            "filename": str(LOG_FILE), # Convert Path object to string
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 3,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": { # Root logger
            "handlers": ["file_handler"],
            "level": "INFO", # Log INFO and above everywhere by default
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "WARNING", # Reduce noise from access logs
            "handlers": ["file_handler"],
            "propagate": False,
        },
         "fastapi": {
            "level": "INFO",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        # Add other libraries if they become too noisy
        # "pydub": {
        #     "level": "WARNING",
        #     "handlers": ["file_handler"],
        #     "propagate": False,
        # },
    },
}

# Apply the logging configuration early, so even startup messages are logged
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__) # Get a logger for this module

# --- FastAPI setup ---
app = FastAPI()

# Ensure static dir exists before mounting
if not STATIC_DIR.exists():
     logger.error(f"Static directory not found at: {STATIC_DIR}")
     # Decide how to handle this - exit or continue without static files?
     # sys.exit(f"Error: Static directory not found at {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/sounds", StaticFiles(directory=SOUNDS_DIR), name="sounds")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of active audio player threads
active_players = []

# --- Favourites & Order Logic ---
def load_favourites():
    if FAVOURITES_FILE.exists():
        try:
            with open(FAVOURITES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {FAVOURITES_FILE}. Starting empty.", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error loading favourites: {e}", exc_info=True)
            return []
    return []

def save_favourites(favs):
    try:
        with open(FAVOURITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving favourites: {e}", exc_info=True)

def load_order():
    if ORDER_FILE.exists():
        try:
            with open(ORDER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {ORDER_FILE}. Starting empty.", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error loading order: {e}", exc_info=True)
            return []
    return []

def save_order(order):
    try:
        with open(ORDER_FILE, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving order: {e}", exc_info=True)

def load_volumes():
    if VOLUMES_FILE.exists():
        try:
            with open(VOLUMES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_volumes(volumes):
    with open(VOLUMES_FILE, "w", encoding="utf-8") as f:
        json.dump(volumes, f, indent=2)

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
         logger.error(f"index.html not found at {index_path}")
         raise HTTPException(status_code=404, detail=f"index.html not found")
    return FileResponse(index_path)

@app.get("/sounds")
async def list_sounds():
    try:
        return [
            f.name
            for f in SOUNDS_DIR.glob("*.wav")
            if f.is_file()
        ]
    except Exception as e:
        logger.error(f"Error listing sounds in {SOUNDS_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing sounds")

@app.get("/favourites")
async def get_favourites():
    return load_favourites()

@app.post("/favourites")
async def set_favourites(request: Request):
    try:
        favs = await request.json()
        if not isinstance(favs, list):
            raise HTTPException(status_code=400, detail="Invalid favourites format, must be a list")
        save_favourites(favs)
        logger.info("Favourites updated.")
        return {"status": "ok"}
    except json.JSONDecodeError:
        logger.warning("Invalid JSON received for favourites update.")
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error setting favourites: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving favourites")

@app.get("/order")
async def get_order():
    return load_order()

@app.post("/order")
async def set_order(request: Request):
    try:
        order = await request.json()
        if not isinstance(order, list):
            raise HTTPException(status_code=400, detail="Invalid order format, must be a list")
        save_order(order)
        logger.info("Sound order updated.")
        return {"status": "ok"}
    except json.JSONDecodeError:
        logger.warning("Invalid JSON received for order update.")
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error setting order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving order")

@app.get("/volumes")
async def get_volumes():
    return load_volumes()

@app.post("/volumes")
async def set_volumes(request: Request):
    try:
        volumes = await request.json()
        if not isinstance(volumes, dict):
            raise HTTPException(status_code=400, detail="Invalid volumes format, must be a dict")
        save_volumes(volumes)
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

@app.get("/settings")
async def get_settings():
    return load_settings()

@app.post("/settings")
async def set_settings(request: Request):
    try:
        data = await request.json()
        settings = load_settings()
        if "dark_mode" in data:
            settings["dark_mode"] = data["dark_mode"]
            save_settings(settings)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error saving settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save settings")

# --- Audio Playback ---
def play_audio_file(path: Path, stop_event: threading.Event):
    try:
        volumes = load_volumes()
        base_name = path.name
        volume = float(volumes.get(base_name, 1.0))  # Default to 1.0 (100%)
        data, samplerate = sf.read(str(path), dtype='float32', always_2d=True)
        if data.shape[1] > 2:
            data = data[:, :2]
        elif data.shape[1] == 1:
            data = np.column_stack((data[:, 0], data[:, 0]))
        # Apply volume
        data = data * volume
        with sd.OutputStream(samplerate=samplerate, channels=data.shape[1], dtype='float32') as stream:
            blocksize = 2048
            idx = 0
            while idx < len(data) and not stop_event.is_set():
                end_idx = min(idx + blocksize, len(data))
                chunk = data[idx:end_idx]
                stream.write(chunk)
                idx = end_idx
    except Exception as e:
        logger.error(f"Error playing sound {path.name}: {type(e).__name__} - {e}", exc_info=True)

@app.post("/play")
async def play_sound(request: Request):
    global active_players
    try:
        data = await request.json()
        sound_name = data.get("name")
        if not sound_name:
            logger.warning("Play request missing sound name.")
            raise HTTPException(status_code=400, detail="Missing 'name' in request body")

        # Sanitize name slightly
        sound_name = Path(sound_name).name
        sound_path = SOUNDS_DIR / sound_name

        if not sound_path.exists() or not sound_path.is_file():
             logger.warning(f"Play request for non-existent sound: {sound_path}")
             raise HTTPException(status_code=404, detail=f"Sound '{sound_name}' not found")

        # Prune completed threads before starting a new one
        active_players = [(t, e) for t, e in active_players if t.is_alive()]

        stop_event = threading.Event()
        # Give the thread a descriptive name
        thread_name = f"AudioPlayer_{sound_name[:20]}" # Truncate long names
        t = threading.Thread(target=play_audio_file, args=(sound_path, stop_event), daemon=True, name=thread_name)
        active_players.append((t, stop_event))
        t.start()
        logger.info(f"Started playback thread for {sound_name}")
        return {"status": "playing", "sound": sound_name}

    except json.JSONDecodeError:
        logger.warning("Invalid JSON received for play request.")
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error handling play request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during play request")


# --- Audio Upload & Processing ---
TARGET_SR = 48000 # Target sample rate (48kHz is common)

@app.post("/upload")
async def upload_sound(file: UploadFile = File(...), image: Optional[UploadFile] = File(None)):
    if not file.filename:
         logger.warning("Upload request with no filename.")
         raise HTTPException(status_code=400, detail="File has no filename")

    # Sanitize filename to prevent path traversal and invalid chars
    safe_filename = Path(file.filename).name
    # Replace potentially problematic characters if needed (more robust sanitization could be added)
    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in ('.', '_', '-')).strip()
    if not safe_filename:
        logger.warning("Upload filename became empty after sanitization.")
        raise HTTPException(status_code=400, detail="Invalid filename provided")

    temp_path = SOUNDS_DIR / ("_temp_" + safe_filename)
    out_name = Path(safe_filename).stem + ".wav" # Ensure .wav extension
    out_path = SOUNDS_DIR / out_name

    logger.info(f"Receiving upload: '{file.filename}' -> saving as '{out_name}'")

    try:
        # Save original file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temporary file saved to: {temp_path}")

        # --- Load and process with pydub ---
        # No need to set ffmpeg path explicitly if it's global
        logger.info(f"Processing '{safe_filename}' with pydub...")
        audio = AudioSegment.from_file(str(temp_path))
        logger.debug(f"Original SR: {audio.frame_rate}, Ch: {audio.channels}, Len: {len(audio)/1000:.2f}s")

        # Resample and convert to stereo
        audio = audio.set_frame_rate(TARGET_SR)
        audio = audio.set_channels(2)
        logger.debug(f"Converted to SR: {audio.frame_rate}, Ch: {audio.channels}")

        # Export as .wav
        audio.export(str(out_path), format="wav")
        logger.info(f"Successfully processed and saved to: {out_path}")

    except FileNotFoundError as fnf_err:
        # This might indicate ffmpeg wasn't found by pydub
        logger.error(f"FileNotFoundError during audio processing (ffmpeg installed globally?): {fnf_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio processing failed. Ensure ffmpeg is installed and in PATH.")
    except Exception as e:
        logger.error(f"Error processing upload '{safe_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {e}")
    finally:
        # Ensure temp file is removed
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Removed temp file: {temp_path}")
            except OSError as e:
                 logger.warning(f"Could not remove temp file {temp_path}: {e}")


    # Handle optional image upload
    image_saved_path_str = None
    if image and image.filename:
        safe_image_filename = Path(image.filename).name # Sanitize
        allowed_exts = {'.png', '.jpg', '.jpeg', '.gif', '.webp'} # Allow common image types
        img_ext = Path(safe_image_filename).suffix.lower()

        if img_ext in allowed_exts:
            # Save image with the same base name as the sound file, using the original image extension
            image_savename = Path(out_name).stem + img_ext
            image_path = SOUNDS_DIR / image_savename
            try:
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                image_saved_path_str = str(image_path)
                logger.info(f"Saved associated image to: {image_path}")
            except Exception as e:
                logger.error(f"Error saving image '{safe_image_filename}': {e}", exc_info=True)
                # Continue without the image
        else:
            logger.warning(f"Skipping image upload '{safe_image_filename}': unsupported extension '{img_ext}'.")

    # Return success status
    return {"status": "uploaded", "filename": out_name, "image_saved": image_saved_path_str}

# --- Stop & Delete ---
@app.post("/stop")
async def stop_all():
    global active_players
    logger.info("Received request to stop all sounds.")
    stopped_count = 0
    # Iterate over a copy for safe removal while iterating
    players_to_stop = active_players[:]
    active_players.clear() # Clear the main list immediately

    for t, stop_event in players_to_stop:
        if not stop_event.is_set():
            stop_event.set()
            stopped_count += 1
            logger.debug(f"Signaled thread {t.name} to stop.")
        # Don't join here in async context - let them terminate
        # if t.is_alive():
        #     t.join(timeout=0.1) # Avoid blocking async loop

    # sd.stop() # Force stop all sounddevice streams if needed (can be abrupt)

    logger.info(f"Signaled {stopped_count} active player(s) to stop.")
    # Give threads a moment to potentially react before next request
    # await asyncio.sleep(0.1) # Optional short async sleep
    return {"status": "stopped", "stopped_count": stopped_count}


@app.delete("/delete/{sound_name}")
async def delete_sound(sound_name: str):
    logger.info(f"Received request to delete sound: {sound_name}")
    # Sanitize input
    sound_name = Path(sound_name).name
    if not sound_name.endswith(".wav"):
         logger.warning(f"Delete request for invalid sound name format: {sound_name}")
         raise HTTPException(status_code=400, detail="Invalid sound name format, must end with .wav")

    sound_path = SOUNDS_DIR / sound_name
    if not sound_path.exists() or not sound_path.is_file():
        logger.warning(f"Delete request for non-existent sound file: {sound_path}")
        raise HTTPException(status_code=404, detail="Sound file not found")

    deleted_files = []
    try:
        sound_path.unlink()
        deleted_files.append(str(sound_path.name))
        logger.info(f"Deleted sound file: {sound_path}")
    except Exception as e:
         logger.error(f"Failed to delete sound file {sound_path}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Failed to delete sound file: {e}")

    # Try deleting associated images (.png, .jpg, .jpeg, .gif, .webp)
    sound_stem = Path(sound_name).stem
    for img_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        image_path = SOUNDS_DIR / (sound_stem + img_ext)
        if image_path.exists() and image_path.is_file():
            try:
                image_path.unlink()
                deleted_files.append(str(image_path.name))
                logger.info(f"Deleted associated image: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to delete associated image {image_path}: {e}", exc_info=True)
                # Don't fail the whole request if image deletion fails

    # Update favourites and order
    try:
        favs = load_favourites()
        if sound_name in favs:
            favs.remove(sound_name)
            save_favourites(favs)
            logger.info(f"Removed '{sound_name}' from favourites.")

        order = load_order()
        if sound_name in order:
            order.remove(sound_name)
            save_order(order)
            logger.info(f"Removed '{sound_name}' from order.")
    except Exception as e:
         logger.error(f"Error updating config files after deleting '{sound_name}': {e}", exc_info=True)
         # Continue returning success for the deletion itself

    return {"status": "deleted", "deleted_files": deleted_files}

# --- Tray Icon Logic ---
tray_icon = None # Global variable to hold the tray icon instance
server_stop_event = threading.Event() # Event to signal server shutdown

def create_tray_icon(url):
    if not pystray:
        logger.warning("pystray not available, cannot create tray icon.")
        return # Skip if library not available

    global tray_icon

    def create_image():
        # Load icon from the static folder using resource_path
        icon_path_str = resource_path(f"static/{ICON_FILE}")
        try:
            image = Image.open(icon_path_str)
            logger.debug(f"Loaded tray icon from {icon_path_str}")
            return image
        except Exception as e:
            logger.warning(f"Could not load icon '{icon_path_str}'. Using fallback. Error: {e}", exc_info=True)
            # Fallback drawing code
            img = Image.new('RGB', (64, 64), color=(0, 119, 204)) # Blue
            d = ImageDraw.Draw(img)
            d.text((12, 8), "S B", fill=(255,255,255)) # Simple 'SB' text
            return img

    def on_open(icon, item):
        logger.info(f"Opening URL from tray: {url}")
        webbrowser.open(url)

    def on_quit(icon, item):
        logger.info("Quit requested from tray icon.")
        server_stop_event.set() # Signal the main thread/server to stop
        icon.stop() # Stop the tray icon loop

    def on_settings(icon, item):
        threading.Thread(target=show_settings_window, daemon=True).start()

    menu = Menu(
        MenuItem(f"Open Soundboard ({url})", on_open, default=True),
        MenuItem("Settings...", on_settings),
        Menu.SEPARATOR,
        MenuItem("Quit", on_quit)
    )

    icon_image = create_image()
    tray_icon = pystray.Icon(APP_NAME.lower().replace(" ", "_"), icon_image, APP_NAME, menu)

    logger.info("Starting tray icon...")
    try:
        # Run the icon loop in the main thread (it blocks)
        tray_icon.run()
    except Exception as e:
         logger.error(f"Error running tray icon: {e}", exc_info=True)
    finally:
        logger.info("Tray icon run loop finished.")
        # Ensure main thread knows to exit if tray stops unexpectedly or normally
        server_stop_event.set()

# --- Settings Window ---
def show_settings_window():
    import tkinter as tk
    from tkinter import ttk, messagebox
    import sounddevice as sd

    # Load current settings or defaults
    settings = load_settings()
    current_host = settings.get("host", get_local_ip())
    current_port = str(settings.get("port", 8000))
    current_device = settings.get("audio_device", None)

    # Query only playback devices (those shown in Windows sound tab)
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        # Find Windows WASAPI or MME hostapis
        valid_hostapis = [i for i, h in enumerate(hostapis) if h['name'] in ('Windows WASAPI', 'MME', 'Windows DirectSound')]
        device_names = [
            f"{i}: {d['name']}"
            for i, d in enumerate(devices)
            if d['max_output_channels'] > 0 and d['hostapi'] in valid_hostapis
        ]
    except Exception as e:
        device_names = []
        print("Could not query audio devices:", e)

    root = tk.Tk()
    root.title("Soundboard Settings")
    root.geometry("400x260")
    root.resizable(False, False)

    # Set window icon to the same .ico as the tray and project
    try:
        icon_path = resource_path(f"static/{ICON_FILE}")
        root.iconbitmap(icon_path)
    except Exception as e:
        print(f"Could not set window icon: {e}")

    tk.Label(root, text="Host Address:").pack(anchor="w", padx=20, pady=(18,0))
    host_entry = tk.Entry(root)
    host_entry.insert(0, current_host)
    host_entry.pack(fill="x", padx=20)

    tk.Label(root, text="Server Port:").pack(anchor="w", padx=20, pady=(12,0))
    port_entry = tk.Entry(root)
    port_entry.insert(0, current_port)
    port_entry.pack(fill="x", padx=20)

    tk.Label(root, text="Audio Output Device:").pack(anchor="w", padx=20, pady=(12,0))
    device_var = tk.StringVar()
    device_combo = ttk.Combobox(root, textvariable=device_var, values=device_names, state="readonly")
    if current_device:
        for name in device_names:
            if name.startswith(str(current_device)):
                device_var.set(name)
                break
    device_combo.pack(fill="x", padx=20)

    # Dark mode checkbox
    dark_mode_var = tk.BooleanVar(value=settings.get("dark_mode", False))
    dark_mode_check = tk.Checkbutton(root, text="Enable Dark Mode", variable=dark_mode_var)
    dark_mode_check.pack(anchor="w", padx=20, pady=(16, 0))

    def save_and_close():
        host = host_entry.get().strip()
        port = port_entry.get().strip()
        device = device_var.get().split(":")[0] if device_var.get() else None
        try:
            port = int(port)
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer.")
            return
        settings["host"] = host
        settings["port"] = port
        settings["audio_device"] = int(device) if device is not None else None
        settings["dark_mode"] = dark_mode_var.get()
        save_settings(settings)
        messagebox.showinfo("Settings Saved", "Settings saved. Please restart the app for host/port changes to take effect.")
        root.destroy()

    tk.Button(root, text="Save", command=save_and_close).pack(pady=18)
    root.mainloop()

# --- Server Startup Logic ---
def get_local_ip():
    """Tries various methods to get a non-localhost IP"""
    ip = '127.0.0.1' # Default
    try:
        # Method 1: Connect to external non-blocking socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.1)
            s.connect(('8.8.8.8', 1)) # Connect to Google DNS
            ip = s.getsockname()[0]
            if not ip or ip.startswith('127.'):
                 raise ValueError("Connected to loopback or got invalid IP")
            logger.debug(f"IP found via external connect: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"IP discovery method 1 failed: {e}")

    try:
         # Method 2: Get hostname and resolve it
         hostname = socket.gethostname()
         ip = socket.gethostbyname(hostname)
         if not ip or ip.startswith('127.'):
              raise ValueError("Resolved to loopback or invalid IP")
         logger.debug(f"IP found via hostname resolution: {ip}")
         return ip
    except Exception as e:
        logger.debug(f"IP discovery method 2 failed: {e}")

    # Method 3 (More complex, OS-specific - could use psutil if added as dependency)
    # ...

    logger.warning(f"Could not determine non-localhost IP, falling back to {ip}")
    return ip


# --- Main Execution ---
if __name__ == "__main__":
    # PyInstaller compatibility for multiprocessing if used (not strictly needed here now)
    # multiprocessing.freeze_support()

    # Set working directory to script location (useful for relative paths if not frozen)
    # if not getattr(sys, 'frozen', False):
    #    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger.info(f"--- Starting {APP_NAME} ---")
    logger.info(f"Running from: {os.path.abspath(__file__ if '__file__' in locals() else sys.argv[0])}")
    logger.info(f"Persistent data directory: {PERSISTENT_DATA_DIR}")
    logger.info(f"Sounds directory: {SOUNDS_DIR}")
    logger.info(f"Static files source directory: {STATIC_DIR}")
    logger.info(f"Logging to file: {LOG_FILE}")

    # Determine host and port
    settings = load_settings()
    HOST = settings.get("host", get_local_ip())
    PORT = int(settings.get("port", 8000))
    AUDIO_DEVICE = settings.get("audio_device", None)
    if AUDIO_DEVICE is not None:
        try:
            sd.default.device = int(AUDIO_DEVICE)
        except Exception as e:
            logger.warning(f"Could not set audio device: {e}")
    URL = f"http://{HOST}:{PORT}"
    logger.info(f"Server will attempt to run at: {URL}")

    # --- Configure Uvicorn ---
    config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        log_config=LOGGING_CONFIG, # Use our custom file logging config
        # loop="asyncio", # Default, but can be explicit
        # lifespan="on", # Handle startup/shutdown events if defined in FastAPI app
    )
    server = uvicorn.Server(config)

    # Start Uvicorn in a separate thread
    server_thread = threading.Thread(target=server.run, daemon=True, name="UvicornServerThread")
    server_thread.start()
    logger.info("Uvicorn server thread started.")

    # Allow server startup time
    time.sleep(2) # Give uvicorn a couple of seconds to bind port etc.

    # Check if server thread is alive before proceeding
    if not server_thread.is_alive():
        logger.critical("Uvicorn server thread failed to start. Check logs. Exiting.")
        sys.exit(1)

    logger.info(f"Access the soundboard web UI at: {URL}")

    # --- Start Tray Icon or Fallback ---
    if pystray:
        # This blocks the main thread until the tray icon is quit
        create_tray_icon(URL)
        # --- Shutdown sequence after tray icon exits ---
        logger.info("Tray icon exited. Initiating server shutdown.")
    else:
        # Fallback if pystray is not installed - keep running until Ctrl+C
        print("\npystray not found. Running without tray icon.")
        print(f"Access the soundboard at: {URL}")
        print("Press Ctrl+C to stop the server.")
        try:
            while not server_stop_event.is_set():
                time.sleep(1) # Keep main thread alive
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Initiating server shutdown.")
            server_stop_event.set() # Signal shutdown

    # --- Cleanup & Exit ---
    logger.info("Requesting Uvicorn server to shut down...")
    # Uvicorn's server.run() running in a daemon thread doesn't have a direct clean
    # shutdown method easily callable from here. Setting should_exit on the Server
    # instance before starting the thread *might* work, but is complex.
    # For a simple tray app, abruptly exiting after tray closes is often acceptable.
    # If cleaner shutdown is vital, investigate uvicorn's programmatic shutdown more.
    # server.should_exit = True # This might work depending on uvicorn version and loop

    # Optionally wait a moment for server thread to potentially exit
    # server_thread.join(timeout=2)
    # if server_thread.is_alive():
    #    logger.warning("Server thread did not exit cleanly.")

    logger.info(f"--- {APP_NAME} finished ---")
    os._exit(0) # Force exit if necessary, common for tray apps