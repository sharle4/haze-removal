"""
Script d'évaluation expérimentale (Batch Processing).

Ce script permet d'analyser la robustesse de l'algorithme en faisant varier
le paramètre critique 'omega' sur une image donnée.
Il calcule systématiquement les métriques (NR et FR si référence fournie)
et exporte les résultats dans un fichier CSV.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict
import copy
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

def run_experiment(
    config_template: dict,
    image_path: Path,
    output_dir: Path,
    ref_path: Optional[Path] = None
):
    """
    Exécute une batterie de tests en faisant varier le paramètre Omega.
    """
    
    # 1. Chargement des images
    logger.info(f"Chargement de l'image : {image_path}")
    hazy_image = read_image(str(image_path))
    
    ref_image = None
    if ref_path and ref_path.exists():
        logger.info(f"Chargement de la référence : {ref_path}")
        ref_image = read_image(str(ref_path))
        if ref_image.shape != hazy_image.shape:
            logger.warning("Attention : Dimensions image/référence différentes ! Les métriques FR ne seront pas calculées.")
            ref_image = None
    
    # 2. Définition de l'espace de recherche (Parameter Space)
    # On fait varier Omega de 0.50 à 1.0 par pas de 0.05
    omega_values = [round(x, 2) for x in np.arange(0.50, 1.01, 0.05)]
    
    results = []
    
    img_result_dir = output_dir / image_path.stem
    img_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Début de l'itération sur {len(omega_values)} valeurs de omega...")

    # 3. Boucle expérimentale
    for omega in tqdm(omega_values, desc="Expérimentation"):
        current_config = copy.deepcopy(config_template)
        
        if 'algorithm' not in current_config:
            current_config['algorithm'] = {}
        current_config['algorithm']['omega'] = omega
        
        try:
            dehazer = Dehazer(current_config)
            restored_image = dehazer.infer(hazy_image)
            
            filename = f"{image_path.stem}_omega_{omega:.2f}.png"
            save_path = img_result_dir / filename
            save_image(restored_image, str(save_path))
            
            # Calcul des métriques
            row = {
                'filename': filename,
                'omega': omega,
                'patch_size': current_config['algorithm'].get('patch_size', 15),
                't0': current_config['algorithm'].get('t0', 0.1)
            }
            
            # Métriques No-Reference (toujours calculées)
            nr_scores = metrics.compute_nr_metrics(hazy_image, restored_image)
            row.update(nr_scores)
            
            # Métriques Full-Reference (si ref dispo)
            if ref_image is not None:
                fr_scores = metrics.compute_fr_metrics(restored_image, ref_image)
                row.update(fr_scores)
                
            results.append(row)
            
        except Exception as e:
            logger.error(f"Echec pour omega={omega}: {e}")
            continue

    # 4. Sauvegarde des données
    if results:
        df = pd.DataFrame(results)
        
        cols = ['filename', 'omega', 'psnr', 'ssim', 'ciede2000', 'hautiere_r', 'hautiere_sigma', 'dark_channel_residual']
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[cols]
        
        csv_path = img_result_dir / "metrics_report.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Rapport de métriques généré : {csv_path}")
        
        best_r = df.loc[df['hautiere_r'].idxmax()]
        logger.info(f"Meilleur restauration (Visibilité) : Omega={best_r['omega']} (r={best_r['hautiere_r']:.3f})")
        
        if 'psnr' in df.columns:
            best_psnr = df.loc[df['psnr'].idxmax()]
            logger.info(f"Meilleur restauration (Fidélité/PSNR) : Omega={best_psnr['omega']} (PSNR={best_psnr['psnr']:.2f}dB)")

def main():
    parser = argparse.ArgumentParser(description="Analyse expérimentale : variation de paramètres.")
    parser.add_argument('--config', type=str, required=True, help="Config de base (YAML).")
    parser.add_argument('--image-path', type=str, required=True, help="Image brumeuse.")
    parser.add_argument('--ref', type=str, default=None, help="Image de référence (Ground Truth) optionnelle.")
    parser.add_argument('--output-dir', type=str, default="experiment", help="Dossier de sortie.")
    
    args = parser.parse_args()
    
    setup_basic_logging("INFO")
    
    image_base_dir = Path('./images')
    output_base_dir = Path('./results')
    
    config = load_config(args.config)
    image_path = image_base_dir / args.image_path
    ref_path = image_base_dir / args.ref if args.ref else None
    output_dir = output_base_dir / args.output_dir
    
    if not image_path.exists():
        logger.error(f"Image introuvable : {image_path}")
        exit(1)

    if ref_path and not ref_path.exists():
        logger.warning(f"Image de référence introuvable : {ref_path}")
        ref_path = None

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(config, image_path, output_dir, ref_path)

if __name__ == '__main__':
    main()