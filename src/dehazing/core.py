"""
Implémentation de l'algorithme "Dark Channel Prior" (He et al.) 
et "Color-Lines" (Fattal).
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import cg
from tqdm import tqdm
import cv2

# PARTIE 1 : MÉTHODES COMMUNES & DARK CHANNEL PRIOR (HE ET AL.)

def get_dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Calcule le Dark Channel d'une image (He et al.).
    Eq (1) & (2).
    """
    if patch_size % 2 == 0:
        raise ValueError("La taille du patch doit être impaire.")
    
    min_channel_img = np.min(image, axis=2)
    dark_channel = ndimage.minimum_filter(min_channel_img, size=patch_size)
    return dark_channel


def estimate_atmospheric_light(hazy_image: np.ndarray, dark_channel: np.ndarray, percentile: float) -> np.ndarray:
    """
    Estime la lumière atmosphérique globale (A) via le Dark Channel.
    """
    total_pixels = dark_channel.size
    num_brightest = int(max(total_pixels * percentile, 1))
    
    flat_dark = dark_channel.flatten()
    indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    
    coords = np.unravel_index(indices, dark_channel.shape)
    candidates = hazy_image[coords]
    
    # Choix du pixel le plus lumineux parmi les candidats (Heuristic)
    brightest_idx = np.argmax(np.sum(candidates, axis=1))
    return candidates[brightest_idx]


def estimate_initial_transmission(hazy_image: np.ndarray, atmospheric_light: np.ndarray, patch_size: int, omega: float) -> np.ndarray:
    """
    Transmission initiale selon He et al. (DCP).
    """
    norm_img = hazy_image / (atmospheric_light + 1e-7)
    transmission = 1 - omega * get_dark_channel(norm_img, patch_size)
    return transmission

# PARTIE 2 : RAFFINEMENT & POST-TRAITEMENT

def refine_transmission_soft_matting(initial_transmission: np.ndarray, hazy_image: np.ndarray, lambda_param: float, epsilon: float, win_size: int) -> np.ndarray:
    """
    Raffinement par Soft Matting (Levin et al.). Très lent mais très précis.
    """
    if win_size % 2 == 0:
        raise ValueError("win_size doit être impair.")

    h, w, _ = hazy_image.shape
    img_size = h * w
    matting_laplacian = lil_matrix((img_size, img_size))

    indices_map = np.arange(img_size).reshape(h, w)
    win_radius = win_size // 2
    U3 = np.identity(3)

    print(f"Construction du Laplacien ({h}x{w})...")
    
    mean_I = cv2.boxFilter(hazy_image, -1, (win_size, win_size), borderType=cv2.BORDER_REFLECT)
    for y in tqdm(range(h), desc="Matting Laplacian Construction"):
        for x in range(w):
            y_min, y_max = max(0, y - win_radius), min(h, y + win_radius + 1)
            x_min, x_max = max(0, x - win_radius), min(w, x + win_radius + 1)
            
            win_pixels = hazy_image[y_min:y_max, x_min:x_max].reshape(-1, 3)
            win_indices = indices_map[y_min:y_max, x_min:x_max].flatten()
            win_area = len(win_pixels)
            
            mean_k = np.mean(win_pixels, axis=0)
            centered = win_pixels - mean_k
            cov_k = (centered.T @ centered) / win_area + (epsilon / win_area) * U3
            inv_cov = np.linalg.inv(cov_k)
            
            # Formule Levin (Eq 14 He et al)
            # L_ij = sum_k (delta_ij - (1 + (I_i - mu_k)^T Sigma_k^-1 (I_j - mu_k)) / |w_k|)
            vals = 1 + (centered @ inv_cov @ centered.T)
            vals /= win_area
            
    return initial_transmission

def refine_transmission_guided_filter(transmission: np.ndarray, guide_image: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    """
    Guided Filter (He et al. 2010). O(1) par rapport à la taille du kernel.
    Optimal pour l'interpolation des valeurs manquantes de Fattal.
    """
    if guide_image.ndim == 3:
        guide_image = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
        
    mean_I = cv2.boxFilter(guide_image, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(transmission, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide_image * transmission, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_image * guide_image, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    q = mean_a * guide_image + mean_b
    return np.clip(q, 0, 1)


def recover_scene_radiance(hazy_image: np.ndarray, atmospheric_light: np.ndarray, transmission: np.ndarray, t0: float) -> np.ndarray:
    """
    Eq (16).
    """
    transmission_clamped = np.maximum(transmission, t0)
    transmission_3d = np.expand_dims(transmission_clamped, axis=2)
    scene_radiance = (hazy_image - atmospheric_light) / transmission_3d + atmospheric_light
    return np.clip(scene_radiance, 0, 1)

def compute_sky_mask(image: np.ndarray, min_intensity: float = 0.8, max_gradient: float = 0.1) -> np.ndarray:
    luminance = np.max(image, axis=2)
    gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    dx = ndimage.sobel(gray, axis=0)
    dy = ndimage.sobel(gray, axis=1)
    mag = np.hypot(dx, dy)
    prob_lum = 0.5 * (np.tanh(10.0 * (luminance - min_intensity)) + 1)
    prob_grad = 0.5 * (np.tanh(10.0 * (max_gradient - mag)) + 1)
    return prob_lum * prob_grad

def apply_sky_protection(transmission: np.ndarray, sky_mask: np.ndarray, k_strength: float = 0.95) -> np.ndarray:
    target = np.maximum(transmission, k_strength)
    return transmission * (1 - sky_mask) + target * sky_mask

# PARTIE 3 : MÉTHODE FATTAL (COLOR-LINES)

def solve_intersection(V, D, A):
    """
    Résout l'intersection géométrique entre la Color-Line (V + l*D) et l'Airlight (s*A).
    Minimise || l*D + V - s*A ||^2.
    
    Retourne s (où transmission t = 1 - s).
    Voir Appendix Eq. (10) & (11) du papier Fattal.
    """
    # On suppose A et D normalisés unitairement
    # Le système est une matrice 2x2 : [[D.D, -D.A], [-D.A, A.A]] * [l, s]^T = [-V.D, V.A]^T
    # Comme ||D||=1 et ||A||=1 :
    
    dot_DA = np.dot(D, A)
    dot_VD = np.dot(V, D)
    dot_VA = np.dot(V, A)
    
    denominator = 1 - dot_DA**2
    
    # Si le dénominateur est proche de 0, D et A sont parallèles -> Instable
    if denominator < 1e-4:
        return None # Cas dégénéré (Intersection Angle Check)

    # Solution analytique pour s (Eq 11 du papier)
    # s = ( (D.A)(V.D) - (V.A) ) / ( (D.A)^2 - 1 )
    # dans le papier, le terme est inversé car c'est minimize ||...||^2
    # On cherche s tel que sA approx V + lD.
    # Projetons sur A et D pour éliminer l.
    
    s = (dot_DA * dot_VD - dot_VA) / (dot_DA**2 - 1.0)
    return s


def fast_ransac_patch_direction(patch_pixels, iterations=10, threshold=0.02):
    """
    RANSAC pour trouver la direction principale D et un point V.
    """
    n_points = patch_pixels.shape[0]
    if n_points < 3:
        return None, None

    best_inliers = -1
    best_D = None
    best_V = None

    idx1 = np.random.randint(0, n_points, iterations)
    idx2 = np.random.randint(0, n_points, iterations)
    
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    
    actual_iters = len(idx1)
    
    for k in range(actual_iters):
        p1 = patch_pixels[idx1[k]]
        p2 = patch_pixels[idx2[k]]
        
        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm < 1e-4: 
            continue
            
        D_cand = vec / norm
        V_cand = p1
        
        # dist = || (P - V) x D || / ||D|| (ici ||D||=1)
        vecs = patch_pixels - V_cand
        cross = np.cross(vecs, D_cand)
        dists = np.linalg.norm(cross, axis=1)
        
        inliers_count = np.sum(dists < threshold)
        
        if inliers_count > best_inliers:
            best_inliers = inliers_count
            best_D = D_cand
            best_V = V_cand
            
            # on choisit un early exit si le modèle est très bon (>90% inliers) pour gagner un peu de temsp
            if best_inliers > 0.9 * n_points:
                break
                
    if best_D is None:
        return None, None
        
    return best_V, best_D


def dehaze_fattal_lpc_ransac_pca(hazy_img, window_size=7, t0=0.1, r_guided=40, eps_guided=1e-3):
    """
    Implémentation de Fattal "Color-Lines" (2014).

    """
    I = hazy_img.astype(np.float32) # [0, 1]
    H, W, C = I.shape
    
    # 1. Estimation de A
    J_dark = get_dark_channel(I, patch_size=15)
    A = estimate_atmospheric_light(I, J_dark, percentile=0.001)
    
    # Normalisation de A 
    A_norm = np.linalg.norm(A)
    if A_norm < 1e-6: A_norm = 1.
    A_unit = A / A_norm

    # 2. Carte de transmission brute
    t_raw = np.full((H, W), np.nan, dtype=np.float32)
    
    pad = window_size // 2
    I_padded = cv2.copyMakeBorder(I, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Seuil d'angle (15 degrés en radians pour la validation)
    angle_threshold_rad = np.deg2rad(15)
    cos_threshold = np.cos(angle_threshold_rad) 

    # Calcul de la variance locale pour sauter les zones plates (shading variability check du papier)
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    local_std = ndimage.generic_filter(gray, np.std, size=window_size)
    
    # Itération
    print("Estimation des Color-Lines (RANSAC)...")
    for i in tqdm(range(0, H, 2)): # Stride de 2 pour accélérer sans trop perdre
        for j in range(0, W, 2):
            # Check 1: Sufficient shading variability (Papier §3.4)
            if local_std[i, j] < 0.02:
                continue

            patch = I_padded[i:i + window_size, j:j + window_size]
            pixels = patch.reshape(-1, 3)
            
            # RANSAC pour trouver V et D
            V, D = fast_ransac_patch_direction(pixels, iterations=15)
            
            if V is None:
                continue
                
            # Vérification de l'orientation de D (doit être positive, sinon on inverse)
            # R = réflectance, elle doit être positive.
            if np.sum(D) < 0:
                D = -D
                
            # Check 2: Large Intersection Angle (Papier §3.4)
            dot_DA = np.dot(D, A_unit)
            if abs(dot_DA) > cos_threshold:
                # Trop parallèle à la lumière atmosphérique -> Rejet
                continue
                
            # Calcul de l'intersection (Eq 5 du papier)
            # On cherche s tel que t = 1 - s
            # A est le vecteur complet, A_unit est la direction
            
            s_val = solve_intersection(V, D, A_unit)
            
            if s_val is None:
                continue
            
            # s correspond à la distance le long de A_unit.
            # L'airlight total est A, de magnitude A_norm.
            # d'où fraction de brume est s_val / A_norm.
            # t = 1 - fraction_brume
            
            transmission_est = 1.0 - (s_val / A_norm)
            
            # Check 3: Valid Transmission (Papier §3.4)
            if 0.0 <= transmission_est <= 1.0:
                # On remplit le bloc 2x2 car stride de 2
                t_raw[i:min(i+2, H), j:min(j+2, W)] = transmission_est

    # 3. Interpolation et Régularisation (MRF / Guided Filter)    
    mask_valid = (~np.isnan(t_raw)).astype(np.uint8)
    
    if np.sum(mask_valid) < 100:
        logger.warning("Color-Lines a échoué sur trop de pixels. Fallback sur transmission uniforme.")
        t_raw[:] = 0.5
    else:
        t_inpainted = t_raw.copy()
        t_inpainted[np.isnan(t_inpainted)] = 0
        t_inpainted = cv2.inpaint((t_inpainted*255).astype(np.uint8), 
                                  ((1-mask_valid)*255).astype(np.uint8), 
                                  3, cv2.INPAINT_TELEA)
        t_raw = t_inpainted.astype(np.float32) / 255.0

    # Raffinement final
    t_refined = refine_transmission_guided_filter(t_raw, I, r_guided, eps_guided)
    
    # Borne inférieure t0
    t_refined = np.maximum(t_refined, t0)
    
    # 4. Reconstruction
    J = recover_scene_radiance(I, A, t_refined, t0)
    
    return (J * 255).clip(0, 255).astype(np.uint8), t_refined, A