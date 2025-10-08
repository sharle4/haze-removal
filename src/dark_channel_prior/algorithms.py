"""
Implémentation de l'algorithme "Dark Channel Prior".
"""

import numpy as np
import scipy.ndimage as ndimage

def get_dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Calcule le Dark Channel d'une image.

    Args:
        image (np.ndarray): Image d'entrée (RGB) normalisée entre 0 et 1.
                            Shape: (hauteur, largeur, 3).
        patch_size (int): Taille du patch carré pour le filtre minimum. Doit être impair.

    Returns:
        np.ndarray: Canal sombre de l'image. Shape: (hauteur, largeur).
    """
    if patch_size % 2 == 0:
        raise ValueError("La taille du patch (patch_size) doit être un entier impair.")
    
    min_channel_img = np.min(image, axis=2)
    
    dark_channel = ndimage.minimum_filter(min_channel_img, size=patch_size)
    
    return dark_channel


def estimate_atmospheric_light(hazy_image: np.ndarray, dark_channel: np.ndarray, percentile: float) -> np.ndarray:
    """
    Estime la lumière atmosphérique globale (A).

    Args:
        hazy_image (np.ndarray): Image brumeuse d'entrée (RGB, 0-1).
        dark_channel (np.ndarray): Canal sombre de l'image.
        percentile (float): Pourcentage des pixels à considérer (ex: 0.001 pour 0.1%).

    Returns:
        np.ndarray: Lumière atmosphérique (A) sous forme d'un vecteur RGB. Shape: (3,).
    """
    total_pixels = dark_channel.size
    num_brightest = int(total_pixels * percentile)
    
    flat_dark_channel = dark_channel.flatten()
    indices = np.argpartition(flat_dark_channel, -num_brightest)[-num_brightest:]
    
    coords = np.unravel_index(indices, dark_channel.shape)
    
    candidate_pixels = hazy_image[coords]
    
    brightest_idx = np.argmax(np.sum(candidate_pixels, axis=1))
    
    atmospheric_light = candidate_pixels[brightest_idx]
    
    return atmospheric_light


def estimate_initial_transmission(hazy_image: np.ndarray, atmospheric_light: np.ndarray, patch_size: int, omega: float) -> np.ndarray:
    """
    Estime la carte de transmission initiale.
    Basée sur l'équation (12) du papier.

    Args:
        hazy_image (np.ndarray): Image brumeuse d'entrée (RGB, 0-1).
        atmospheric_light (np.ndarray): Lumière atmosphérique (A).
        patch_size (int): Taille du patch pour le calcul du canal sombre.
        omega (float): Facteur de conservation de la brume.

    Returns:
        np.ndarray: Carte de transmission initiale. Shape: (hauteur, largeur).
    """
    hazy_image_norm = hazy_image / atmospheric_light
    
    transmission = 1 - omega * get_dark_channel(hazy_image_norm, patch_size)
    
    return transmission


def refine_transmission_soft_matting(transmission: np.ndarray, hazy_image: np.ndarray, lambda_val: float, epsilon: float) -> np.ndarray:
    """
    Affine la carte de transmission en utilisant la méthode "Soft Matting"

    Args:
        transmission (np.ndarray): Carte de transmission initiale.
        hazy_image (np.ndarray): Image brumeuse couleur (0-1), utilisée comme guide.
        lambda_val (float): Paramètre de régularisation lambda.
        epsilon (float): Régularisateur pour l'inversion de la matrice de covariance.

    Returns:
        np.ndarray: Carte de transmission affinée.
    """
    pass


def refine_transmission_guided_filter(transmission: np.ndarray, hazy_image_gray: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    """
    Affine la carte de transmission en utilisant un Filtre Guidé (basé sur le papier "Guided Image Filtering").

    Args:
        transmission (np.ndarray): Carte de transmission initiale.
        hazy_image_gray (np.ndarray): Image brumeuse en niveaux de gris (0-1), utilisée comme guide.
        radius (int): Rayon du filtre.
        epsilon (float): Paramètre de régularisation.

    Returns:
        np.ndarray: Carte de transmission affinée.
    """
    mean_I = ndimage.uniform_filter(hazy_image_gray, size=radius)
    mean_p = ndimage.uniform_filter(transmission, size=radius)
    corr_I = ndimage.uniform_filter(hazy_image_gray * hazy_image_gray, size=radius)
    corr_Ip = ndimage.uniform_filter(hazy_image_gray * transmission, size=radius)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I
    
    mean_a = ndimage.uniform_filter(a, size=radius)
    mean_b = ndimage.uniform_filter(b, size=radius)
    
    refined_transmission = mean_a * hazy_image_gray + mean_b
    return np.clip(refined_transmission, 0, 1)


def recover_scene_radiance(hazy_image: np.ndarray, atmospheric_light: np.ndarray, transmission: np.ndarray, t0: float) -> np.ndarray:
    """
    Récupère l'image sans brume (radiance de la scène).
    Basée sur l'équation (16) du papier.

    Args:
        hazy_image (np.ndarray): Image brumeuse d'entrée (RGB, 0-1).
        atmospheric_light (np.ndarray): Lumière atmosphérique (A).
        transmission (np.ndarray): Carte de transmission (brute ou affinée).
        t0 (float): Borne inférieure pour la transmission.

    Returns:
        np.ndarray: Image sans brume (RGB, 0-1).
    """
    transmission_3d = np.expand_dims(transmission, axis=2)
    
    transmission_clamped = np.maximum(transmission_3d, t0)
    
    scene_radiance = (hazy_image - atmospheric_light) / transmission_clamped + atmospheric_light
    
    return np.clip(scene_radiance, 0, 1)