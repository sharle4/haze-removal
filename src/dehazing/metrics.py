"""
Module d'évaluation et de métriques pour le débrumage d'images.
Implémente des métriques Full-Reference (PSNR, SSIM, CIEDE2000) et No-Reference (Hautière, Colorfullness, DarkChannel Residual).
"""

import logging
from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray, rgb2lab, deltaE_ciede2000
from scipy.ndimage import sobel

from . import core

logger = logging.getLogger(__name__)


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Calcule la magnitude du gradient (Sobel) sur une image en niveaux de gris."""
    if image.ndim == 3:
        image = rgb2gray(image)
    
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    mag = np.hypot(dx, dy)
    return mag

def calculate_indicators(original: np.ndarray, restored: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    """
    Calcule les indicateurs e, r et sigma.
    
    Args:
        original: Image brumeuse (float 0-1).
        restored: Image restaurée (float 0-1).
        threshold: Seuil de détection des bords (5% par défaut).
        
    Returns:
        Dict contenant:
        - 'e': Nombre de bords visibles dans l'image restaurée.
        - 'r': Taux de nouveaux bords visibles (Visibilité retrouvée).
        - 'sigma': Pourcentage de pixels saturés (noirs/blancs).
    """
    # 1. Conversion gris
    orig_gray = rgb2gray(original) if original.ndim == 3 else original
    rest_gray = rgb2gray(restored) if restored.ndim == 3 else restored
    
    # 2. Détection des bords
    grad_orig = compute_gradient_magnitude(orig_gray)
    grad_rest = compute_gradient_magnitude(rest_gray)
    
    # Masques binaires des bords visibles
    edges_orig = grad_orig > threshold
    edges_rest = grad_rest > threshold
    
    n_orig = np.count_nonzero(edges_orig)
    n_rest = np.count_nonzero(edges_rest)
    
    # 3. Calcul de r (Rate of new visible edges)
    # Hautière définit r comme le ratio des nouveaux bords sur les bords originaux
    if n_orig == 0:
        r = 0.0
    else:
        r = (n_rest - n_orig) / n_orig
        
    # 4. Calcul de Sigma (Saturation)
    # Pourcentage de pixels devenus tout noirs (0) ou tout blancs (1)
    # On utilise une petite marge epsilon pour les flottants
    eps = 1e-3
    saturated = np.count_nonzero(restored < eps) + np.count_nonzero(restored > (1.0 - eps))
    sigma = saturated / restored.size
    
    return {"e": n_rest, "r": r, "sigma": sigma}


def compute_ciede2000(restored: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calcule la différence de couleur perceptuelle (Delta E 2000).
        Plus la valeur est basse, meilleure est la fidélité couleur.
        """
        # Conversion RGB -> Lab (Nécessite des images en [0, 1] ou [0, 255] interprétées correctement)
        # skimage assume float inputs are in [0, 1]
        lab_rest = rgb2lab(restored)
        lab_gt = rgb2lab(ground_truth)
        
        # Calcul du Delta E sur toute l'image
        delta_e = deltaE_ciede2000(lab_gt, lab_rest)
        return float(np.mean(delta_e))


def compute_colorfulness(image: np.ndarray) -> float:
        """
        Calcule l'indice de 'Colorfulness' (Hasler & Suesstrunk, 2003).
        Mesure à quel point l'image est colorée/vivante.
        M = sigma_rgby + 0.3 * mu_rgby
        """
        if image.ndim != 3:
            return 0.0
            
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        
        # Espace de couleur opposant (rg, yb)
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        
        std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
        mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        
        return float(std_root + 0.3 * mean_root) 


def compute_fr_metrics(self, restored: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques Full-Reference (PSNR, SSIM)
    
    Args:
        restored: Image débrumée (H, W, 3), float [0, 1].
        ground_truth: Image claire (H, W, 3), float [0, 1].
        
    Returns:
        Dictionnaire des scores.
    """
    if restored.shape != ground_truth.shape:
        logger.warning(f"Dimensions incompatibles: {restored.shape} vs {ground_truth.shape}. Redimensionnement ignoré, calcul annulé.")
        return {}
    metrics = {}
    
    # 1. PSNR
    metrics['psnr'] = peak_signal_noise_ratio(ground_truth, restored, data_range=1.0)
    
    # 2. SSIM
    metrics['ssim'] = structural_similarity(ground_truth, restored, data_range=1.0, channel_axis=2)
    
    # 3. CIEDE2000
    metrics['ciede2000'] = compute_ciede2000(restored, ground_truth)
    
    return metrics

def compute_nr_metrics(self, original: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
    """Métriques No-Reference (Hautière, Colorfulness, Dark Channel Residual)."""
    metrics = {}
    
    # 1. Hautière (Contraste structurel)
    metrics.update(calculate_indicators(original, restored))
    
    # 2. Colorfulness (Vivacité des couleurs)
    metrics['colorfulness_in'] = compute_colorfulness(original)
    metrics['colorfulness_out'] = compute_colorfulness(restored)
    
    # 3. Dark Channel Residual (Densité de brume restante)
    # On recalcule le DC sur l'image finale. Il devrait être proche de 0.
    dc_restored = core.get_dark_channel(restored, patch_size=15)
    metrics['dark_channel_residual'] = float(np.mean(dc_restored))
    
    return metrics