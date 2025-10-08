"""
Fonctions pour la lecture et l'écriture d'images.
"""

import numpy as np
from PIL import Image
from typing import Optional

def read_image(image_path: str) -> Optional[np.ndarray]:
    """
    Lit une image depuis un fichier et la convertit en un tableau numpy (float, 0-1).

    Args:
        image_path (str): Chemin vers le fichier image.

    Returns:
        Optional[np.ndarray]: Tableau numpy représentant l'image en format RGB,
                              avec des valeurs flottantes dans [0, 1].
                              Retourne None si le fichier n'est pas trouvé.
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_np = np.array(img, dtype=np.float32) / 255.0
        return img_np
    except FileNotFoundError:
        print(f"Erreur: Fichier image introuvable à l'adresse '{image_path}'")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image '{image_path}': {e}")
        return None


def save_image(image_np: np.ndarray, save_path: str):
    """
    Sauvegarde un tableau numpy en tant que fichier image.

    Args:
        image_np (np.ndarray): Tableau numpy (float, 0-1) à sauvegarder.
        save_path (str): Chemin de destination pour l'image.
    """
    try:
        img_to_save = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_to_save)
        img.save(save_path)
        print(f"Image sauvegardée à l'adresse : {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image à '{save_path}': {e}")
