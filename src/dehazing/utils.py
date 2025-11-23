"""
Utilitaires pour les E/S, la configuration et le traitement d'image de base.
"""
import logging
import os
from typing import Any, Dict

import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis : {config_path}")
        return config or {}
    except FileNotFoundError:
        logger.error(f"Erreur : Fichier de configuration '{config_path}' introuvable.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Erreur lors du parsing YAML '{config_path}': {e}")
        raise

def setup_basic_logging(level_str: str = "INFO"):
    """Configure un logging basique sur stdout."""
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def read_image(image_path: str) -> np.ndarray:
    """
    Lit une image et la convertit en tableau numpy float [0, 1] RGB.
    Lève une exception en cas d'erreur.
    """
    if not os.path.exists(image_path):
         raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
        return img_np
    except Exception as e:
        logger.error(f"Erreur à la lecture de '{image_path}': {e}")
        raise

def save_image(image_np: np.ndarray, save_path: str):
    """Sauvegarde un tableau numpy float [0, 1] en fichier image uint8."""
    try:
        img_to_save = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_to_save)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        logger.info(f"Image sauvegardée : {save_path}")
    except Exception as e:
        logger.error(f"Erreur à la sauvegarde vers '{save_path}': {e}")
        raise

def convert_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Convertit une image RGB (float 0-1) en niveaux de gris (NTSC standard)."""
    return np.dot(image_rgb[...,:3], [0.2989, 0.5870, 0.1140])