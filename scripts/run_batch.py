"""
Script d'évaluation expérimentale (Batch Processing).

Ce script permet d'analyser la robustesse de l'algorithme en faisant varier
le paramètre critique 'omega' sur une image donnée.
Il calcule systématiquement les métriques (NR et FR si référence fournie),
exporte les résultats dans un fichier CSV et génère des graphiques d'analyse.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

def generate_plots(df: pd.DataFrame, output_dir: Path):
    """
    Génère des graphiques pertinents pour l'analyse académique.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Graphique Compromis No-Reference : Visibilité (r) vs Saturation (Sigma)
    # C'est le graphique le plus important pour régler Omega.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_r = 'tab:blue'
    ax1.set_xlabel('Omega (Force du débrumage)', fontsize=12)
    ax1.set_ylabel('Gain de Visibilité (Hautière r)', color=color_r, fontsize=12)
    sns.lineplot(data=df, x='omega', y='hautiere_r', ax=ax1, color=color_r, marker='o', linewidth=2, label='Visibilité (r)')
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.grid(True, alpha=0.3)

    # Axe secondaire pour Sigma (car l'échelle est différente, souvent très petite)
    ax2 = ax1.twinx()
    color_sigma = 'tab:red'
    ax2.set_ylabel('Taux de Saturation (Sigma)', color=color_sigma, fontsize=12)
    sns.lineplot(data=df, x='omega', y='hautiere_sigma', ax=ax2, color=color_sigma, marker='s', linestyle='--', linewidth=2, label='Saturation (Sigma)')
    ax2.tick_params(axis='y', labelcolor=color_sigma)
    ax2.grid(False)

    plt.title('Impact de Omega : Compromis Visibilité vs Artefacts', fontsize=14)
    fig.tight_layout()
    plt.savefig(output_dir / "analysis_nr_tradeoff.png", dpi=300)
    plt.close()

    # 2. Graphique Full-Reference : Fidélité (PSNR/SSIM) vs Omega
    # Uniquement si une référence était fournie (colonne psnr existe)
    if 'psnr' in df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_psnr = 'tab:green'
        ax1.set_xlabel('Omega', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=12)
        sns.lineplot(data=df, x='omega', y='psnr', ax=ax1, color=color_psnr, marker='o', label='PSNR')
        ax1.tick_params(axis='y', labelcolor=color_psnr)
        
        ax2 = ax1.twinx()
        color_ssim = 'tab:orange'
        ax2.set_ylabel('SSIM', color=color_ssim, fontsize=12)
        sns.lineplot(data=df, x='omega', y='ssim', ax=ax2, color=color_ssim, marker='x', linestyle='--', label='SSIM')
        ax2.tick_params(axis='y', labelcolor=color_ssim)
        
        plt.title('Fidélité structurelle en fonction de la force du débrumage', fontsize=14)
        fig.tight_layout()
        plt.savefig(output_dir / "analysis_fr_fidelity.png", dpi=300)
        plt.close()

    # 3. Graphique Avancé : Front de Pareto (Scatter plot)
    # On veut maximiser r (Y) et minimiser Sigma (X). 
    # Le meilleur point est en haut à gauche.
    plt.figure(figsize=(8, 8))
    scatter = sns.scatterplot(
        data=df, 
        x='hautiere_sigma', 
        y='hautiere_r', 
        hue='omega', 
        palette='viridis', 
        s=100,
        edgecolor='k'
    )
    plt.xlabel('Saturation (Sigma) - À minimiser')
    plt.ylabel('Visibilité (r) - À maximiser')
    plt.title('Espace Performance : Visibilité vs Distorsion')
    
    # Annoter quelques points intéressants
    for line in range(0, df.shape[0], max(1, df.shape[0]//5)): # Annoter 1 point sur 5
        plt.text(
            df.hautiere_sigma.iloc[line], 
            df.hautiere_r.iloc[line]+0.01, 
            f"ω={df.omega.iloc[line]:.2f}", 
            horizontalalignment='left', 
            size='small', 
            color='black'
        )
        
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_pareto.png", dpi=300)
    plt.close()


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
            
            # Métriques Full-Reference (si référence fournie)
            if ref_image is not None:
                fr_scores = metrics.compute_fr_metrics(restored_image, ref_image)
                row.update(fr_scores)
                
            results.append(row)
            
        except Exception as e:
            logger.error(f"Echec pour omega={omega}: {e}")
            continue

    # 4. Sauvegarde et Analyse
    if results:
        df = pd.DataFrame(results)
        
        # Réorganisation et nettoyage des colonnes
        cols = ['filename', 'omega', 'psnr', 'ssim', 'ciede2000', 'hautiere_r', 'hautiere_sigma', 'dark_channel_residual']
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[cols]
        
        csv_path = img_result_dir / "metrics_report.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Rapport CSV généré : {csv_path}")
        
        # Génération des graphiques
        logger.info("Génération des graphiques d'analyse...")
        generate_plots(df, img_result_dir)
        
        # Résumé console
        best_r = df.loc[df['hautiere_r'].idxmax()]
        logger.info(f"-> Max Visibilité : Omega={best_r['omega']} (r={best_r['hautiere_r']:.3f}, sigma={best_r['hautiere_sigma']:.3f})")
        
        if 'psnr' in df.columns:
            best_psnr = df.loc[df['psnr'].idxmax()]
            logger.info(f"-> Max PSNR : Omega={best_psnr['omega']} (PSNR={best_psnr['psnr']:.2f}dB)")

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