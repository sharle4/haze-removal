"""
Module principal définissant la classe Dehazer.
Intègre désormais deux méthodes de débrumage : 
1. Dark Channel Prior (He et al.)
2. Color-Lines (Fattal)
"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import cv2
from . import core, utils

logger = logging.getLogger(__name__)

class Dehazer:
    """
    Classe principale pour exécuter les pipelines de suppression de brume.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le Dehazer avec une configuration.
        """
        self.config = config
        self.alg_config = config.get('algorithm', {})
        self.ref_config = config.get('refinement', {})
        self.sky_config = config.get('sky_detection', {})

        # Paramètres communs / He et al.
        self.patch_size = self.alg_config.get('patch_size', 15)
        self.omega = self.alg_config.get('omega', 0.95)
        self.atm_percentile = self.alg_config.get('atmospheric_light_percentile', 0.001)
        self.t0 = self.alg_config.get('t0', 0.1)

        # Paramètres Sky Detection (He et al.)
        self.enable_sky_detection = self.sky_config.get('enabled', True)
        self.sky_intensity_threshold = self.sky_config.get('intensity_threshold', 0.85)
        self.sky_gradient_threshold = self.sky_config.get('gradient_threshold', 0.05)
        self.sky_strength = self.sky_config.get('correction_strength', 0.95)

    def infer_he(self, image: np.ndarray) -> np.ndarray:
        """
        Exécute la méthode Dark Channel Prior (He et al. 2009).
        """
        if image.max() > 1.0 + 1e-5:
             logger.warning("Attention: Image non normalisée [0,1].")

        # 1. Dark Channel
        dark_channel = core.get_dark_channel(image, self.patch_size)

        # 2. Lumière Atmosphérique
        atmospheric_light = core.estimate_atmospheric_light(
            image, dark_channel, self.atm_percentile
        )

        # 3. Transmission Initiale
        raw_transmission = core.estimate_initial_transmission(
            image, atmospheric_light, self.patch_size, self.omega
        )

        # 4. Sky Detection
        current_transmission = raw_transmission
        if self.enable_sky_detection:
            sky_mask = core.compute_sky_mask(
                image, 
                min_intensity=self.sky_intensity_threshold,
                max_gradient=self.sky_gradient_threshold
            )
            current_transmission = core.apply_sky_protection(
                raw_transmission, sky_mask, k_strength=self.sky_strength
            )

        # 5. Raffinement (Guided Filter)
        gf_config = self.ref_config.get('guided_filter', {})
        radius = gf_config.get('radius', 60)
        epsilon = gf_config.get('epsilon', 1e-3)
        
        gray_guide = utils.convert_to_grayscale(image)
        refined_transmission = core.refine_transmission_guided_filter(
            current_transmission, gray_guide, radius, epsilon
        )

        # 6. Restauration
        scene_radiance = core.recover_scene_radiance(
            image, atmospheric_light, refined_transmission, self.t0
        )

        return scene_radiance

    def infer_fattal(self, image: np.ndarray) -> np.ndarray:
        """
        Exécute la méthode Color-Lines (Fattal 2014).
        """
        win_size = self.alg_config.get('fattal_window_size', 7)
        
        result_uint8, _, _ = core.dehaze_fattal_lpc_ransac_pca(
            image, 
            window_size=win_size,
            t0=self.t0,
            r_guided=40,    # Valeurs recommandées pour Fattal
            eps_guided=1e-3
        )
        
        return result_uint8.astype(np.float32) / 255.0

    def infer(self, image: np.ndarray, method: str = "he") -> np.ndarray:
        """
        fonction générique.
        Args:
            method: "he" ou "fattal"
        """
        if method.lower() == "fattal":
            return self.infer_fattal(image)
        else:
            return self.infer_he(image)