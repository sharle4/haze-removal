"""
Script CLI pour exécuter une comparaison complète (He vs Fattal) sur une image unique.
Génère une image comparative et affiche les métriques dans la console.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
import sys

# Ajout du chemin src au path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

def make_comparison_image(original, res_he, res_fattal):
    """Crée une planche comparative avec labels."""
    h, w, c = original.shape
    
    # Création d'une canvas large (3 images côte à côte)
    canvas = np.zeros((h + 50, w * 3, c), dtype=np.float32)
    
    # Remplissage images
    canvas[50:, :w, :] = original
    canvas[50:, w:2*w, :] = res_he
    canvas[50:, 2*w:, :] = res_fattal
    
    # Conversion uint8 pour OpenCV (textes)
    canvas_uint8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    
    # Ajout des labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas_uint8, "Original (Input)", (10, 35), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas_uint8, "He et al. (DCP)", (w + 10, 35), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas_uint8, "Fattal (Color-Lines)", (2*w + 10, 35), font, 1, (255, 255, 255), 2)
    
    return canvas_uint8.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description="Comparaison He vs Fattal sur une image.")
    parser.add_argument('--config', type=str, required=True, help="Fichier de configuration YAML.")
    parser.add_argument('--image-path', type=str, required=True, help="Image d'entrée.")
    parser.add_argument('--output-dir', type=str, required=True, help="Dossier de sortie.")
    parser.add_argument('--ref', type=str, default=None, help="Image de référence (Ground Truth) optionnelle.")
    
    args = parser.parse_args()
    setup_basic_logging("INFO")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image_path)
    
    # 1. Chargement
    try:
        config = load_config(args.config)
        hazy_image = read_image(str(image_path))
        
        ref_image = None
        if args.ref:
            ref_image = read_image(args.ref)
    except Exception as e:
        logger.error(f"Erreur chargement: {e}")
        return

    dehazer = Dehazer(config)
    
    # 2. Exécution Méthode 1 : He et al.
    logger.info("Exécution méthode He et al. (DCP)...")
    res_he = dehazer.infer_he(hazy_image)
    save_image(res_he, str(output_dir / f"{image_path.stem}_He.png"))
    
    # 3. Exécution Méthode 2 : Fattal
    logger.info("Exécution méthode Fattal (Color-Lines)...")
    try:
        res_fattal = dehazer.infer_fattal(hazy_image)
        save_image(res_fattal, str(output_dir / f"{image_path.stem}_Fattal.png"))
    except Exception as e:
        logger.error(f"Erreur Fattal: {e}")
        res_fattal = np.zeros_like(hazy_image) # Fallback noir

    # 4. Comparaison Visuelle
    logger.info("Génération de la planche comparative...")
    comp_img = make_comparison_image(hazy_image, res_he, res_fattal)
    save_image(comp_img, str(output_dir / f"{image_path.stem}_COMPARISON.png"))
    
    # 5. Calcul et Affichage des Métriques
    logger.info("\n=== RAPPORT DE MÉTRIQUES ===")
    
    def print_metrics(name, img_res, img_in, img_ref=None):
        nr = metrics.compute_nr_metrics(img_in, img_res)
        print(f"\n--- {name} ---")
        print(f"  [NR] Gain Visibilité (r) : {nr['hautiere_r']:.3f}")
        print(f"  [NR] Saturation (Sigma)  : {nr['hautiere_sigma']:.1%}")
        print(f"  [NR] Colorfulness        : {nr['colorfulness_out']:.2f}")
        
        if img_ref is not None:
            fr = metrics.compute_fr_metrics(img_res, img_ref)
            if fr:
                print(f"  [FR] PSNR                : {fr['psnr']:.2f} dB")
                print(f"  [FR] SSIM                : {fr['ssim']:.3f}")
                print(f"  [FR] CIEDE2000 (Color)   : {fr['ciede2000']:.2f}")

    print_metrics("He et al. (DCP)", res_he, hazy_image, ref_image)
    print_metrics("Fattal (Color-Lines)", res_fattal, hazy_image, ref_image)
    print("\n" + "="*30)

if __name__ == '__main__':
    main()