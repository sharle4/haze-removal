"""
Serveur API FastAPI pour l'application de suppression de brume.

Ce module expose les points d'accès (endpoints) web pour :
- Servir l'interface utilisateur (frontend).
- Lancer un traitement unique avec un jeu de paramètres.
- Lancer une batterie d'expériences avec une grille de paramètres.
- Diffuser en temps réel les journaux (logs) et résultats du traitement via SSE.
"""

import os
import uuid
import asyncio
import base64
import io as BytesIO
import itertools
import json
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
from src.dark_channel_prior.algorithms import (
    get_dark_channel, estimate_atmospheric_light, estimate_initial_transmission,
    refine_transmission_guided_filter, recover_scene_radiance
)
from src.dark_channel_prior.preprocessing import convert_to_grayscale


# --- Configuration de l'application FastAPI ---
app = FastAPI(
    title="API de Suppression de Brume (Dark Channel Prior)",
    description="Une API pour appliquer l'algorithme DCP sur des images via une interface web.",
    version="2.0.0"
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
    if img_np.dtype != np.uint8:
        img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img_np
    
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

# --- Endpoint pour un traitement unique ---
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
    Endpoint pour recevoir une image et lancer un traitement unique.
    """
    job_id = str(uuid.uuid4())
    log_queues[job_id] = Queue()

    try:
        contents = await image.read()
        hazy_image_pil = Image.open(BytesIO.BytesIO(contents)).convert('RGB')
        hazy_image_np = np.array(hazy_image_pil, dtype=np.float32) / 255.0
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide.")

    form_data = {
        "patch_size": patch_size, "omega": omega, 
        "atmospheric_light_percentile": atmospheric_light_percentile, "t0": t0,
        "gf_radius": gf_radius, "gf_epsilon": gf_epsilon
    }
    config = get_config_from_form(form_data)

    asyncio.create_task(
        run_single_processing(job_id, hazy_image_np, config)
    )

    return {"job_id": job_id}

# --- Endpoint pour une batterie d'expériences ---
@app.post("/process-experiment/")
async def process_experiment_endpoint(
    image: UploadFile = File(...),
    parameter_grid: str = Form(...)
):
    """
    Endpoint pour lancer une batterie d'expériences basées sur une grille de paramètres.
    """
    job_id = str(uuid.uuid4())
    log_queues[job_id] = Queue()

    try:
        param_grid_dict = json.loads(parameter_grid)
        contents = await image.read()
        hazy_image_pil = Image.open(BytesIO.BytesIO(contents)).convert('RGB')
        hazy_image_np = np.array(hazy_image_pil, dtype=np.float32) / 255.0
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=400, detail="Grille de paramètres JSON invalide.")
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide.")

    asyncio.create_task(
        run_experiment_processing(job_id, hazy_image_np, param_grid_dict)
    )

    return {"job_id": job_id}


async def run_single_processing(job_id: str, hazy_image: np.ndarray, config: Dict):
    """Orchestre le traitement d'une seule image et nettoie les ressources."""
    log_queue = log_queues[job_id]
    
    def log_callback(message: str, data: Dict = None):
        """Fonction de rappel pour envoyer des messages au client."""
        payload = {"type": "log", "message": message}
        if data:
            data["type"] = "result_intermediate"
            payload.update(data)
        log_queue.put(payload)

    try:
        dcp_core.run_haze_removal_pipeline(
            hazy_image=hazy_image,
            config=config,
            log_callback=log_callback,
            image_to_base64_func=image_to_base64
        )
        log_queue.put({"type": "done", "message": "Traitement unique terminé."})
    except Exception as e:
        log_queue.put({"type": "error", "message": f"Erreur critique: {e}"})
    finally:
        await asyncio.sleep(5) 
        if job_id in log_queues:
            del log_queues[job_id]


async def run_experiment_processing(job_id: str, hazy_image: np.ndarray, param_grid: Dict):
    """Orchestre une batterie d'expériences."""
    log_queue = log_queues[job_id]
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    log_queue.put({"type": "experiment_start", "total_runs": len(combinations)})

    try:
        for i, combo in enumerate(combinations):
            run_params = dict(zip(param_names, combo))
            log_queue.put({"type": "log", "message": f"Calcul de la combinaison {i+1}/{len(combinations)}: {run_params}"})
            
            run_config = {
                'algorithm': {
                    'patch_size': run_params.get('patch_size', 15),
                    'omega': run_params.get('omega', 0.95),
                    'atmospheric_light_percentile': run_params.get('atmospheric_light_percentile', 0.001),
                    't0': run_params.get('t0', 0.1)
                },
                'refinement': {
                    'guided_filter': {
                        'radius': run_params.get('gf_radius', 60),
                        'epsilon': run_params.get('gf_epsilon', 1e-3)
                    }
                }
            }
            
            final_image = dcp_core.process_image_for_experiment(hazy_image, run_config)
            final_image_b64 = image_to_base64(final_image)

            log_queue.put({
                "type": "run_result",
                "image": final_image_b64,
                "params": run_params,
                "run_index": i
            })
            await asyncio.sleep(0.01)

        log_queue.put({"type": "experiment_done", "message": "Expérience terminée avec succès."})
    except Exception as e:
        log_queue.put({"type": "error", "message": f"Erreur critique durant l'expérience: {e}"})
    finally:
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
        log_queue = log_queues.get(job_id)
        if not log_queue: return
        try:
            while True:
                if not log_queue.empty():
                    data = log_queue.get()
                    yield {"data": json.dumps(data)}
                    
                    if data.get("type") in ["done", "experiment_done", "error"]:
                        break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"Client déconnecté pour le job {job_id}")
            raise

    return EventSourceResponse(event_generator())


# --- Service des fichiers statiques ---
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# --- Point d'entrée pour lancer le serveur (avec Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    print("Serveur démarré. Accédez à l'application via http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
