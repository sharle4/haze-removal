"""
Serveur API FastAPI pour l'application de suppression de brume.

Ce module expose les points d'accès (endpoints) web pour :
- Servir l'interface utilisateur (frontend).
- Recevoir une image et des paramètres, puis lancer le traitement.
- Diffuser en temps réel les journaux (logs) du traitement.
"""

import os
import uuid
import asyncio
import base64
import io as BytesIO
from typing import Dict, Any
from queue import Queue

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import numpy as np
from PIL import Image
import yaml

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dark_channel_prior import core as dcp_core
from src.dark_channel_prior import io as dcp_io
from src.dark_channel_prior import visualization as dcp_vis

# --- Configuration de l'application FastAPI ---
app = FastAPI(
    title="API de Suppression de Brume (Dark Channel Prior)",
    description="Une API pour appliquer l'algorithme DCP sur des images via une interface web.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variables globales et gestion d'état ---
log_queues: Dict[str, Queue] = {}
UPLOADS_DIR = "temp_uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Fonctions utilitaires de l'API ---
def get_config_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit le dictionnaire de configuration à partir des données du formulaire.

    Args:
        form_data (Dict[str, Any]): Données brutes reçues du formulaire web.

    Returns:
        Dict[str, Any]: Dictionnaire de configuration structuré.
    """
    return {
        'algorithm': {
            'patch_size': int(form_data['patch_size']),
            'omega': float(form_data['omega']),
            'atmospheric_light_percentile': float(form_data['atmospheric_light_percentile']),
            't0': float(form_data['t0'])
        },
        'refinement': {
            'method': "guided_filter",
            'guided_filter': {
                'radius': int(form_data['gf_radius']),
                'epsilon': float(form_data['gf_epsilon'])
            }
        }
    }

def image_to_base64(img_np: np.ndarray) -> str:
    """
    Convertit une image (tableau NumPy) en une chaîne de caractères base64.

    Args:
        img_np (np.ndarray): L'image à convertir.

    Returns:
        str: La chaîne base64 préfixée pour l'affichage en HTML.
    """
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = BytesIO.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# --- Définition des Endpoints de l'API ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Sert le fichier principal de l'interface web.

    Returns:
        FileResponse: Le fichier index.html.
    """
    return FileResponse('public/index.html')

@app.get("/default-config")
async def get_default_config():
    """
    Sert la configuration par défaut depuis le fichier YAML.

    Returns:
        Dict[str, Any]: Le contenu du fichier de configuration.
    """
    try:
        with open('configs/default.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichier de configuration par défaut non trouvé.")

@app.post("/process-image/")
async def process_image_endpoint(
    image: UploadFile = File(...),
    patch_size: int = Form(...),
    omega: float = Form(...),
    atmospheric_light_percentile: float = Form(...),
    t0: float = Form(...),
    gf_radius: int = Form(...),
    gf_epsilon: float = Form(...)
):
    """
    Endpoint pour recevoir une image et lancer le traitement.
    """
    job_id = str(uuid.uuid4())
    log_queues[job_id] = Queue()

    contents = await image.read()
    try:
        hazy_image_pil = Image.open(BytesIO.BytesIO(contents)).convert('RGB')
        hazy_image_np = np.array(hazy_image_pil, dtype=np.float32) / 255.0
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide.")

    form_data = {
        "patch_size": patch_size, "omega": omega, "atmospheric_light_percentile": atmospheric_light_percentile, "t0": t0,
        "gf_radius": gf_radius, "gf_epsilon": gf_epsilon
    }
    config = get_config_from_form(form_data)

    asyncio.create_task(
        run_processing_and_cleanup(job_id, hazy_image_np, config)
    )

    return {"job_id": job_id}

async def run_processing_and_cleanup(job_id: str, hazy_image: np.ndarray, config: Dict):
    """
    Orchestre le traitement de l'image et le nettoyage des ressources.
    """
    log_queue = log_queues[job_id]
    
    def log_callback(message: str, data: Dict = None):
        """Fonction de rappel pour envoyer des messages au client."""
        payload = {"type": "log", "message": message}
        if data:
            payload.update(data)
        log_queue.put(payload)

    try:
        dcp_core.run_haze_removal_pipeline(
            hazy_image=hazy_image,
            config=config,
            log_callback=log_callback,
            image_to_base64_func=image_to_base64
        )
    except Exception as e:
        log_callback(f"ERREUR: Une erreur critique est survenue : {e}")
    finally:
        log_queue.put({"type": "done", "message": "Traitement terminé."})
        await asyncio.sleep(5) 
        if job_id in log_queues:
            del log_queues[job_id]


@app.get("/stream-logs/{job_id}")
async def stream_logs(job_id: str):
    """
    Endpoint pour diffuser les logs d'une tâche via Server-Sent Events (SSE).
    """
    if job_id not in log_queues:
        async def empty_stream():
            yield {"data": '{"type": "error", "message": "ID de tâche invalide ou expiré."}'}
        return EventSourceResponse(empty_stream())

    async def event_generator():
        """Générateur qui puise dans la file d'attente et envoie les messages."""
        log_queue = log_queues[job_id]
        try:
            while True:
                if not log_queue.empty():
                    data = log_queue.get()
                    yield {"data": f"{data}"}
                    if data.get("type") == "done":
                        break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"Client déconnecté pour le job {job_id}")
            raise

    return EventSourceResponse(event_generator())

app.mount("/", StaticFiles(directory="public", html=True), name="static") # Permet de charger les fichiers statiques css et js

# --- Point d'entrée pour lancer le serveur (avec Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    print("Serveur démarré. Accédez à l'application via http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
