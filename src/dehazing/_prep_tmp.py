"""
Fonctions de prétraitement pour les images.
Dans notre projet de haze-removal, le prétraitement est minimal car l'algorithme s'applique directement.
"""

import numpy as np

def convert_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convertit une image RGB (numpy array, 0-1) en niveaux de gris.

    Args:
        image_rgb (np.ndarray): Image RGB.

    Returns:
        np.ndarray: Image en niveaux de gris.
    """
    return np.dot(image_rgb[...,:3], [0.2989, 0.5870, 0.1140]) # Standard NTSC
