"""
Module principal définissant la classe Dehazer, point d'entrée de la librairie.
"""
import logging
from typing import Any, Dict

import numpy as np

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

        required_keys = ['patch_size', 'omega', 'atmospheric_light_percentile', 't0']
        for key in required_keys:
            if key not in self.alg_config:
                logger.warning(f"Clé de configuration manquante : algorithm.{key}. Utilisation des valeurs par défaut potentielles.")
        
        self.patch_size = self.alg_config.get('patch_size', 15)
        self.omega = self.alg_config.get('omega', 0.95)
        self.atm_percentile = self.alg_config.get('atmospheric_light_percentile', 0.001)
        self.t0 = self.alg_config.get('t0', 0.1)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Exécute le désembuage sur une image (inférence).

        Args:
            image (np.ndarray): Image d'entrée RGB, float [0, 1]. Shape (H, W, 3).

        Returns:
            np.ndarray: Image désembuée RGB, float [0, 1]. Shape (H, W, 3).
        """
        if image.max() > 1.0 + 1e-5:
             logger.warning("L'image d'entrée semble ne pas être normalisée entre [0, 1]. Les résultats peuvent être incorrects.")

        logger.info("Début du processus d'inférence.")
        
        # 1. Calcul du Dark Channel
        dark_channel = core.get_dark_channel(image, self.patch_size)

        # 2. Estimation de la lumière atmosphérique
        atmospheric_light = core.estimate_atmospheric_light(
            image, dark_channel, self.atm_percentile
        )
        logger.debug(f"Lumière atmosphérique estimée : {atmospheric_light}")

        # 3. Estimation de la transmission initiale
        initial_transmission = core.estimate_initial_transmission(
            image, atmospheric_light, self.patch_size, self.omega
        )

        # 4. Affinement (Guided Filter par défaut)
        gf_config = self.ref_config.get('guided_filter', {})
        radius = gf_config.get('radius', 60)
        epsilon = gf_config.get('epsilon', 1e-3)
        
        logger.info(f"Affinement de la transmission avec Guided Filter (r={radius}, eps={epsilon}).")
        gray_guide = utils.convert_to_grayscale(image)
        
        refined_transmission = core.refine_transmission_guided_filter(
            initial_transmission, gray_guide, radius, epsilon
        )

        # 5. Restauration de la radiance (image finale)
        logger.info("Restauration de l'image finale.")
        scene_radiance = core.recover_scene_radiance(
            image, atmospheric_light, refined_transmission, self.t0
        )

        return scene_radiance