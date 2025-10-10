"""
Module principal contenant la logique de l'algorithme de suppression de brume.
"""

import logging
from typing import Dict, Any, Callable
import numpy as np

from . import algorithms as alg
from . import preprocessing as prep
from . import visualization as vis

def run_haze_removal_pipeline(
    hazy_image: np.ndarray,
    config: Dict[str, Any],
    log_callback: Callable[[str, Dict], None],
    image_to_base64_func: Callable[[np.ndarray], str]
):
    """
    Exécute le pipeline complet de suppression de brume sur une image.

    Args:
        hazy_image (np.ndarray): L'image brumeuse d'entrée (tableau NumPy, float 0-1).
        config (Dict[str, Any]): Dictionnaire de configuration pour l'algorithme.
        log_callback (Callable[[str, Dict], None]): Fonction de rappel pour logger la progression.
                                       Le premier argument est le message, le second est un dict optionnel de données.
        image_to_base64_func (Callable[[np.ndarray], str]): Fonction pour convertir une image en base64.
    """
    
    log_callback("Début du traitement...", {})
    
    alg_config = config['algorithm']
    ref_config = config['refinement']

    # --- Étape 1: Calcul du Dark Channel ---
    log_callback("Calcul du canal sombre...", {})
    dark_channel = alg.get_dark_channel(hazy_image, alg_config['patch_size'])
    dark_channel_b64 = image_to_base64_func(dark_channel)
    log_callback("Canal sombre calculé.", {"type": "result", "name": "dark_channel", "image": dark_channel_b64})

    # --- Étape 2: Estimation de la lumière atmosphérique ---
    log_callback("Estimation de la lumière atmosphérique...", {})
    atmospheric_light = alg.estimate_atmospheric_light(
        hazy_image, dark_channel, alg_config['atmospheric_light_percentile']
    )
    log_callback(f"Lumière atmosphérique estimée (A) = [{', '.join(f'{c:.3f}' for c in atmospheric_light)}]", {})

    # --- Étape 3: Estimation de la transmission initiale ---
    log_callback("Estimation de la transmission initiale...", {})
    initial_transmission = alg.estimate_initial_transmission(
        hazy_image, atmospheric_light, alg_config['patch_size'], alg_config['omega']
    )
    initial_trans_b64 = image_to_base64_func(initial_transmission)
    log_callback("Transmission initiale estimée.", {"type": "result", "name": "initial_transmission", "image": initial_trans_b64})

    # --- Étape 4: Affinement et récupération ---
    refinement_method = ref_config['method']
    if refinement_method not in ["guided_filter", "all"]:
        log_callback(f"Avertissement : méthode d'affinement '{refinement_method}' non supportée par l'UI, utilisation de 'guided_filter'.", {}) # Guided filter obligatoire pour la version web
    
    log_callback("Affinement de la transmission avec le Filtre Guidé...", {})
    gf_config = ref_config['guided_filter']
    hazy_gray = prep.convert_to_grayscale(hazy_image)
    
    refined_transmission_gf = alg.refine_transmission_guided_filter(
        initial_transmission, hazy_gray, gf_config['radius'], gf_config['epsilon']
    )
    refined_trans_b64 = image_to_base64_func(refined_transmission_gf)
    log_callback("Transmission affinée avec le Filtre Guidé.", {"type": "result", "name": "refined_transmission", "image": refined_trans_b64})

    log_callback("Récupération de la radiance de la scène...", {})
    scene_radiance_gf = alg.recover_scene_radiance(
        hazy_image, atmospheric_light, refined_transmission_gf, alg_config['t0']
    )
    final_result_b64 = image_to_base64_func(scene_radiance_gf)
    log_callback("Image finale restaurée.", {"type": "result", "name": "final_result", "image": final_result_b64})

    log_callback("Pipeline terminé avec succès.", {})
