"""
Module principal définissant la classe Dehazer.
"""
import logging
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from . import core, utils

logger = logging.getLogger(__name__)

class Dehazer:
    """
    Classe principale pour exécuter le pipeline de suppression de brume (Dark Channel Prior).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le Dehazer avec une configuration.

        Args:
            config (Dict[str, Any]): Dictionnaire de configuration (ex: chargé depuis yaml).
                                     Doit contenir les clés 'algorithm' et éventuellement 'refinement'.
        """
        self.config = config
        self.alg_config = config.get('algorithm', {})
        self.ref_config = config.get('refinement', {})
        self.sky_config = config.get('sky_detection', {})

        required_keys = ['patch_size', 'omega', 'atmospheric_light_percentile', 't0']
        for key in required_keys:
            if key not in self.alg_config:
                logger.warning(f"Clé de configuration manquante : algorithm.{key}. Utilisation des valeurs par défaut potentielles.")
        
        self.patch_size = self.alg_config.get('patch_size', 15)
        self.omega = self.alg_config.get('omega', 0.95)
        self.atm_percentile = self.alg_config.get('atmospheric_light_percentile', 0.001)
        self.t0 = self.alg_config.get('t0', 0.1)

        # Paramètres Sky Detection
        self.enable_sky_detection = self.sky_config.get('enabled', True)
        self.sky_intensity_threshold = self.sky_config.get('intensity_threshold', 0.85)
        self.sky_gradient_threshold = self.sky_config.get('gradient_threshold', 0.05)
        self.sky_strength = self.sky_config.get('correction_strength', 0.95)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Exécute le pipeline complet.
        
        Args:
            image (np.ndarray): Image d'entrée RGB, float [0, 1]. Shape (H, W, 3).

        Returns:
            np.ndarray: Image désembuée RGB, float [0, 1]. Shape (H, W, 3).
        """
        if image.max() > 1.0 + 1e-5:
             logger.warning("Attention: Image non normalisée [0,1].")

        logger.info("--- Début Inférence DCP ---")
        
        # 1. Dark Channel
        dark_channel = core.get_dark_channel(image, self.patch_size)

        # 2. Lumière Atmosphérique
        atmospheric_light = core.estimate_atmospheric_light(
            image, dark_channel, self.atm_percentile
        )
        logger.info(f"Lumière atmosphérique estimée (A) : {np.array2string(atmospheric_light, precision=3)}")

        # 3. Transmission Initiale (Coarse Map)
        raw_transmission = core.estimate_initial_transmission(
            image, atmospheric_light, self.patch_size, self.omega
        )

        # --- OPTIMISATION : SKY DETECTION ---
        # Cette étape est cruciale pour éviter le bruit dans le ciel.
        # Elle doit se faire AVANT le raffinement pour que le Guided Filter lisse la transition.
        current_transmission = raw_transmission
        
        if self.enable_sky_detection:
            logger.info("Détection et protection du ciel...")
            sky_mask = core.compute_sky_mask(
                image, 
                min_intensity=self.sky_intensity_threshold,
                max_gradient=self.sky_gradient_threshold
            )
            
            # Application de la correction
            current_transmission = core.apply_sky_protection(
                raw_transmission, sky_mask, k_strength=self.sky_strength
            )
            
            # (Optionnel) Debug: Sauvegarde du masque si besoin
            # from PIL import Image
            # Image.fromarray((sky_mask * 255).astype(np.uint8)).save("debug_sky_mask.png")
            # Image.fromarray((current_transmission * 255).astype(np.uint8)).save("debug_trans_corrected.png")

        # 4. Raffinement (Guided Filter / Soft Matting)
        # On raffine la transmission corrigée
        gf_config = self.ref_config.get('guided_filter', {})
        radius = gf_config.get('radius', 60)
        epsilon = gf_config.get('epsilon', 1e-3)
        
        logger.info(f"Raffinement Guided Filter (r={radius}, eps={epsilon}).")
        gray_guide = utils.convert_to_grayscale(image)
        
        refined_transmission = core.refine_transmission_guided_filter(
            current_transmission, gray_guide, radius, epsilon
        )

        # 5. Restauration finale
        logger.info("Reconstruction de la radiance...")
        scene_radiance = core.recover_scene_radiance(
            image, atmospheric_light, refined_transmission, self.t0
        )

        return scene_radiance