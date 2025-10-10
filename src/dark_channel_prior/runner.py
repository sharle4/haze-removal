"""
Module contenant le pipeline de traitement principal pour une seule image.
"""

import os
import logging
from typing import Dict, Any
import numpy as np

from . import algorithms as alg
from . import preprocessing as prep
from . import visualization as vis
from . import io

def process_single_image(
    hazy_image: np.ndarray,
    config: Dict[str, Any],
    output_dir: str
):
    """
    Exécute le pipeline complet de suppression de brume pour une image et une config.

    Args:
        hazy_image (np.ndarray): Image brumeuse d'entrée (float, 0-1).
        config (Dict[str, Any]): Dictionnaire de configuration.
        output_dir (str): Répertoire où sauvegarder tous les artéfacts.
    """
    logging.info(f"Début du traitement pour le répertoire : {output_dir}")

    alg_config = config.get('algorithm', {})
    ref_config = config.get('refinement', {})

    # --- Étape 1: Algorithme de base ---
    logging.info("Calcul du canal sombre...")
    dark_channel = alg.get_dark_channel(hazy_image, alg_config.get('patch_size', 15))

    logging.info("Estimation de la lumière atmosphérique...")
    atmospheric_light = alg.estimate_atmospheric_light(
        hazy_image, dark_channel, alg_config.get('atmospheric_light_percentile', 0.001)
    )
    logging.info(f"Lumière atmosphérique estimée (A) = {atmospheric_light}")

    logging.info("Estimation de la transmission initiale...")
    initial_transmission = alg.estimate_initial_transmission(
        hazy_image, atmospheric_light, alg_config.get('patch_size', 15), alg_config.get('omega', 0.95)
    )

    # --- Étape 2: Affinement et récupération ---
    results = {}
    transmissions = {}
    refinement_method = ref_config.get('method', 'guided_filter')

    if refinement_method in ["guided_filter", "all"]:
        logging.info("--- Affinement avec le Filtre Guidé ---")
        gf_config = ref_config.get('guided_filter', {})
        hazy_gray = prep.convert_to_grayscale(hazy_image)

        refined_transmission_gf = alg.refine_transmission_guided_filter(
            initial_transmission,
            hazy_gray,
            gf_config.get('radius', 60),
            gf_config.get('epsilon', 1e-3)
        )
        scene_radiance_gf = alg.recover_scene_radiance(
            hazy_image, atmospheric_light, refined_transmission_gf, alg_config.get('t0', 0.1)
        )
        results["Filtre Guidé"] = scene_radiance_gf
        transmissions["Filtre Guidé"] = refined_transmission_gf

    if refinement_method in ["soft_matting", "all"]:
        logging.warning("L'affinement par Soft Matting est très lent et gourmand en mémoire.")
        sm_config = ref_config.get('soft_matting', {})
        
        refined_transmission_sm = alg.refine_transmission_soft_matting(
            initial_transmission,
            hazy_image,
            sm_config.get('lambda', 0.001),
            sm_config.get('epsilon', 1e-7),
            sm_config.get('win_size', 3)
        )
        scene_radiance_sm = alg.recover_scene_radiance(
            hazy_image, atmospheric_light, refined_transmission_sm, alg_config.get('t0', 0.1)
        )
        results["Soft Matting"] = scene_radiance_sm
        transmissions["Soft Matting"] = refined_transmission_sm

    # --- Étape 3: Sauvegarde des résultats ---
    logging.info("Sauvegarde des résultats...")
    
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    io.save_image(initial_transmission, os.path.join(figures_dir, "transmission_initial.png"))
    for method_name, result_img in results.items():
        filename = f"result_dehazed_{method_name.lower().replace(' ', '_')}.png"
        io.save_image(result_img, os.path.join(figures_dir, filename))

    for method_name, trans_map in transmissions.items():
        filename = f"transmission_{method_name.lower().replace(' ', '_')}.png"
        vis.save_transmission_map(trans_map, os.path.join(figures_dir, filename))
    
    if results:
        vis.save_comparison_figure(
            hazy_image, results, transmissions,
            os.path.join(figures_dir, "comparison.png")
        )

    logging.info(f"Traitement terminé pour {output_dir}")
