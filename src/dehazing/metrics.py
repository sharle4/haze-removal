"""
Module d'évaluation et de métriques pour le débrumage d'images.

Ce module implémente des métriques standard de la littérature pour évaluer la qualité
de la restauration d'image.

Métriques Full-Reference (nécessitent une image sans brume) :
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- CIEDE2000 (Delta E - Fidélité des couleurs perceptuelle)

Métriques No-Reference (Blind Assessment) :
- Hautière et al. (2008) : Indicateurs e (bords), r (taux de restauration), sigma (saturation).
- Colorfulness (Hasler & Suesstrunk, 2003).
- Dark Channel Residual : Mesure la densité de brume restante.
"""

import logging
from typing import Dict, Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray, rgb2lab, deltaE_ciede2000
from scipy.ndimage import sobel

from . import core

logger = logging.getLogger(__name__)


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Calcule la magnitude du gradient (Sobel) sur une image.
    Si l'image est RGB, elle est convertie en niveaux de gris.
    """
    if image.ndim == 3:
        image = rgb2gray(image)
    
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    mag = np.hypot(dx, dy)
    return mag

def calculate_hauti_indicators(original: np.ndarray, restored: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    """
    Calcule les indicateurs de visibilité selon Hautière et al. (2008).
    
    Args:
        original: Image brumeuse (float 0-1).
        restored: Image restaurée (float 0-1).
        threshold: Seuil de gradient pour considérer un pixel comme un "bord" (5% par défaut).
        
    Returns:
        Dict:
        - 'e': Nombre de bords visibles dans l'image restaurée.
        - 'r': Taux de gradients restaurés (plus c'est haut, plus on a révélé de détails invisibles).
        - 'sigma': Pourcentage de pixels saturés (perte d'information par clipping).
    """
    # 1. Conversion gris standardisée
    orig_gray = rgb2gray(original) if original.ndim == 3 else original
    rest_gray = rgb2gray(restored) if restored.ndim == 3 else restored
    
    # 2. Calcul des gradients
    grad_orig = compute_gradient_magnitude(orig_gray)
    grad_rest = compute_gradient_magnitude(rest_gray)
    
    # 3. Seuillage pour obtenir les cartes de bords
    edges_orig = grad_orig > threshold
    edges_rest = grad_rest > threshold
    
    n_orig = np.count_nonzero(edges_orig)
    n_rest = np.count_nonzero(edges_rest)
    
    # 4. Calcul de r (Rate of new visible edges)
    # r = (n_r - n_o) / n_o
    if n_orig == 0:
        r = 0.0 # Évite la division par zéro si l'image originale est plate
    else:
        r = float((n_rest - n_orig) / n_orig)
        
    # 5. Calcul de Sigma (Saturation)
    # Pourcentage de pixels devenus tout noirs (0) ou tout blancs (1) dans l'image restaurée
    eps = 1e-4
    saturated_pixels = np.count_nonzero(restored < eps) + np.count_nonzero(restored > (1.0 - eps))
    sigma = float(saturated_pixels / restored.size)
    
    return {"hautiere_e": float(n_rest), "hautiere_r": r, "hautiere_sigma": sigma}


def compute_ciede2000(restored: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calcule la différence de couleur perceptuelle moyenne (Delta E 2000).
    Moins de 1.0 est considéré comme imperceptible par l'œil humain.
    """
    # Conversion RGB -> Lab (skimage gère la conversion depuis [0, 1])
    lab_rest = rgb2lab(restored)
    lab_gt = rgb2lab(ground_truth)
    
    # Calcul du Delta E pixel par pixel
    delta_e = deltaE_ciede2000(lab_gt, lab_rest)
    return float(np.mean(delta_e))


def compute_colorfulness(image: np.ndarray) -> float:
    """
    Calcule l'indice de 'Colorfulness' (Hasler & Suesstrunk, 2003).
    M = sigma_rgby + 0.3 * mu_rgby.
    """
    if image.ndim != 3:
        return 0.0
        
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    
    # Espace de couleur opposant approché
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    return float(std_root + 0.3 * mean_root) 


def compute_fr_metrics(restored: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques avec référence (Full-Reference).
    
    Args:
        restored: Image débrumée (H, W, 3), float [0, 1].
        ground_truth: Image claire de référence (H, W, 3), float [0, 1].
        
    Returns:
        Dictionnaire contenant PSNR, SSIM, CIEDE2000.
    """
    if restored.shape != ground_truth.shape:
        logger.warning(f"Dimensions incompatibles FR: {restored.shape} vs {ground_truth.shape}. Métriques ignorées.")
        return {}
        
    metrics = {}
    
    # 1. PSNR (Plus haut = mieux)
    metrics['psnr'] = float(peak_signal_noise_ratio(ground_truth, restored, data_range=1.0))
    
    # 2. SSIM (Plus proche de 1.0 = mieux)
    metrics['ssim'] = float(structural_similarity(ground_truth, restored, data_range=1.0, channel_axis=2))
    
    # 3. CIEDE2000 (Plus bas = mieux)
    metrics['ciede2000'] = compute_ciede2000(restored, ground_truth)
    
    return metrics


def compute_nr_metrics(original: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques sans référence (No-Reference).
    """
    metrics = {}
    
    # 1. Indicateurs de Hautière (Visibilité des bords vs Saturation)
    metrics.update(calculate_hauti_indicators(original, restored))
    
    # 2. Colorfulness (Vivacité)
    metrics['colorfulness_in'] = compute_colorfulness(original)
    metrics['colorfulness_out'] = compute_colorfulness(restored)
    
    # 3. Dark Channel Residual (Densité de brume)
    dc_restored = core.get_dark_channel(restored, patch_size=15)
    metrics['dark_channel_residual'] = float(np.mean(dc_restored))
    
    return metrics