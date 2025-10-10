"""
Module principal contenant la logique de l'algorithme de suppression de brume.
"""
from typing import Dict, Any, Callable, Optional
import numpy as np

from . import algorithms as alg
from . import preprocessing as prep

def run_haze_removal_pipeline(
    hazy_image: np.ndarray,
    config: Dict[str, Any],
    log_callback: Callable[[str, Optional[Dict]], None],
    image_to_base64_func: Callable[[np.ndarray], str]
):
    """
    Exécute le pipeline complet de suppression de brume pour une analyse unique
    et envoie les résultats intermédiaires via un callback.

    Args:
        hazy_image (np.ndarray): L'image brumeuse d'entrée (tableau NumPy, float 0-1).
        config (Dict[str, Any]): Dictionnaire de configuration pour l'algorithme.
        log_callback (Callable): Fonction de rappel pour logger la progression et les images.
        image_to_base64_func (Callable): Fonction pour convertir une image en base64.
    """
    
    log_callback("Début du traitement...", None)
    
    alg_config = config['algorithm']
    ref_config = config['refinement']

    # --- Étape 1: Calcul du Dark Channel ---
    log_callback("Calcul du canal sombre...", None)
    dark_channel = alg.get_dark_channel(hazy_image, alg_config['patch_size'])
    dark_channel_b64 = image_to_base64_func(dark_channel)
    log_callback("Canal sombre calculé.", {"name": "dark_channel", "image": dark_channel_b64})

    # --- Étape 2: Estimation de la lumière atmosphérique ---
    log_callback("Estimation de la lumière atmosphérique...", None)
    atmospheric_light = alg.estimate_atmospheric_light(
        hazy_image, dark_channel, alg_config['atmospheric_light_percentile']
    )
    log_callback(f"Lumière atmosphérique (A) = [{', '.join(f'{c:.3f}' for c in atmospheric_light)}]", None)

    # --- Étape 3: Estimation de la transmission initiale ---
    log_callback("Estimation de la transmission initiale...", None)
    initial_transmission = alg.estimate_initial_transmission(
        hazy_image, atmospheric_light, alg_config['patch_size'], alg_config['omega']
    )
    initial_trans_b64 = image_to_base64_func(initial_transmission)
    log_callback("Transmission initiale estimée.", {"name": "initial_transmission", "image": initial_trans_b64})

    # --- Étape 4: Affinement avec le Filtre Guidé ---
    log_callback("Affinement de la transmission avec le Filtre Guidé...", None)
    gf_config = ref_config['guided_filter']
    hazy_gray = prep.convert_to_grayscale(hazy_image)
    
    refined_transmission_gf = alg.refine_transmission_guided_filter(
        initial_transmission, hazy_gray, gf_config['radius'], gf_config['epsilon']
    )
    refined_trans_b64 = image_to_base64_func(refined_transmission_gf)
    log_callback("Transmission affinée.", {"name": "refined_transmission", "image": refined_trans_b64})

    # --- Étape 5: Récupération de l'image finale ---
    log_callback("Récupération de la radiance de la scène...", None)
    scene_radiance_gf = alg.recover_scene_radiance(
        hazy_image, atmospheric_light, refined_transmission_gf, alg_config['t0']
    )
    final_result_b64 = image_to_base64_func(scene_radiance_gf)
    log_callback("Image finale restaurée.", {"name": "final_result", "image": final_result_b64})

    log_callback("Pipeline terminé avec succès.", None)


def process_image_for_experiment(
    hazy_image: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Exécute une version optimisée du pipeline retournant uniquement l'image finale.
    Idéal pour les traitements en lots du mode expérimental.

    Args:
        hazy_image (np.ndarray): L'image brumeuse d'entrée (float 0-1).
        config (Dict[str, Any]): Dictionnaire de configuration pour ce run.

    Returns:
        np.ndarray: L'image finale débruitée (float 0-1).
    """
    alg_config = config['algorithm']
    ref_config = config['refinement']

    dark_channel = alg.get_dark_channel(hazy_image, alg_config['patch_size'])
    
    atmospheric_light = alg.estimate_atmospheric_light(
        hazy_image, dark_channel, alg_config['atmospheric_light_percentile']
    )
    
    initial_transmission = alg.estimate_initial_transmission(
        hazy_image, atmospheric_light, alg_config['patch_size'], alg_config['omega']
    )
    
    gf_config = ref_config['guided_filter']
    hazy_gray = prep.convert_to_grayscale(hazy_image)
    
    refined_transmission = alg.refine_transmission_guided_filter(
        initial_transmission, hazy_gray, gf_config['radius'], gf_config['epsilon']
    )
    
    scene_radiance = alg.recover_scene_radiance(
        hazy_image, atmospheric_light, refined_transmission, alg_config['t0']
    )
    
    return scene_radiance
