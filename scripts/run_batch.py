"""
Script d'évaluation expérimentale (Batch Processing).

Ce script permet d'analyser la robustesse de l'algorithme en faisant varier
le paramètre critique 'omega' sur une image donnée.
Il calcule systématiquement TOUTES les métriques (NR et FR),
exporte les résultats dans un fichier CSV et génère une suite complète de graphiques.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import copy
import math

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

def normalize_series(series):
    """Normalise une série entre 0 et 1 pour le radar chart."""
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

def plot_single_metric(df, x_col, y_col, ax, color, title, ylabel):
    """Fonction utilitaire pour tracer une métrique unique."""
    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, color=color, marker='o', linewidth=2)
    ax.set_ylabel(ylabel, color=color, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

def generate_radar_chart(df, output_dir):
    """Génère un graphique radar pour comparer les profils à faible, moyen et fort Omega."""
    # Sélection de 3 points représentatifs
    omegas = sorted(df['omega'].unique())
    indices = df[df['omega'].isin(omegas[-3:])].index
    selected_rows = df.iloc[indices].copy()
    
    # Métriques à afficher sur le radar (toutes normalisées)
    metrics_to_plot = ['hautiere_r', 'hautiere_sigma', 'colorfulness_out', 'dark_channel_residual']
    if 'psnr' in df.columns:
        metrics_to_plot += ['psnr', 'ssim', 'ciede2000']
    
    # Normalisation pour l'affichage (0-1)
    df_norm = df.copy()
    for m in metrics_to_plot:
        if m in df_norm.columns:
            df_norm[m] = normalize_series(df_norm[m])
            
    selected_norm = df_norm.iloc[indices]
    
    # Création du Radar
    labels = [m.replace('hautiere_', '').replace('_out', '').upper() for m in metrics_to_plot]
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1] # Fermer la boucle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for idx, (_, row) in enumerate(selected_norm.iterrows()):
        values = row[metrics_to_plot].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Omega={row['omega']:.2f}", color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Profil de Performance Normalisé (Radar Chart)", y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_dir / "analysis_global_radar.png", dpi=300)
    plt.close()

def generate_plots(df: pd.DataFrame, output_dir: Path):
    """
    Génère une suite complète de graphiques académiques.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. VISIBILITE (Hautière e & r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_single_metric(df, 'omega', 'hautiere_e', ax1, 'tab:blue', 'Nombre de bords visibles (e)', 'NB Edges')
    plot_single_metric(df, 'omega', 'hautiere_r', ax2, 'tab:cyan', 'Taux de restauration (r)', 'Ratio')
    plt.suptitle("Analyse de la Visibilité (Hautière et al.)")
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_visibility.png", dpi=300)
    plt.close()

    # 2. ARTEFACTS & BRUME (Sigma & Dark Channel)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_sig = 'tab:red'
    ax1.set_xlabel('Omega')
    ax1.set_ylabel('Saturation Sigma (%)', color=color_sig)
    sns.lineplot(data=df, x='omega', y='hautiere_sigma', ax=ax1, color=color_sig, marker='o', label='Saturation (Sigma)')
    ax1.tick_params(axis='y', labelcolor=color_sig)
    
    ax2 = ax1.twinx()
    color_dc = 'tab:gray'
    ax2.set_ylabel('Dark Channel Residual (Brume restante)', color=color_dc)
    sns.lineplot(data=df, x='omega', y='dark_channel_residual', ax=ax2, color=color_dc, marker='s', linestyle='--', label='DC Residual')
    ax2.tick_params(axis='y', labelcolor=color_dc)
    
    plt.title("Compromis : Saturation vs Suppression de Brume")
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_artifacts.png", dpi=300)
    plt.close()

    # 3. ANALYSE COULEUR (Colorfulness)
    # Comparaison In vs Out
    plt.figure(figsize=(10, 6))
    plt.plot(df['omega'], df['colorfulness_out'], label='Restored', color='tab:purple', marker='o')
    plt.axhline(y=df['colorfulness_in'].iloc[0], color='gray', linestyle='--', label='Original Input')
    plt.xlabel('Omega')
    plt.ylabel('Score de Colorfulness')
    plt.title("Impact sur la vivacité des couleurs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_color.png", dpi=300)
    plt.close()

    # 4. FIDELITE (Full Reference) - Seulement si dispo
    if 'psnr' in df.columns:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        
        plot_single_metric(df, 'omega', 'psnr', ax1, 'tab:green', 'PSNR (dB)', 'dB')
        plot_single_metric(df, 'omega', 'ssim', ax2, 'tab:orange', 'SSIM (Structure)', 'Index')
        plot_single_metric(df, 'omega', 'ciede2000', ax3, 'tab:pink', 'Delta E (CIEDE2000 - Erreur Couleur)', 'Delta E')
        
        ax3.set_xlabel('Omega')
        plt.suptitle("Métriques Full-Reference (Fidélité)")
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_fidelity_full.png", dpi=300)
        plt.close()

    # 5. RADAR CHART (Vue synthétique)
    try:
        generate_radar_chart(df, output_dir)
    except Exception as e:
        logger.warning(f"Impossible de générer le Radar Chart: {e}")

def run_experiment(
    config_template: dict,
    image_path: Path,
    output_dir: Path,
    ref_path: Optional[Path] = None
):
    """
    Exécute une batterie de tests en faisant varier le paramètre Omega.
    """
    logger.info(f"Chargement de l'image : {image_path}")
    hazy_image = read_image(str(image_path))
    
    ref_image = None
    if ref_path and ref_path.exists():
        logger.info(f"Chargement de la référence : {ref_path}")
        ref_image = read_image(str(ref_path))
        if ref_image.shape != hazy_image.shape:
            logger.warning("Attention : Dimensions image/référence différentes ! Les métriques FR ne seront pas calculées.")
            ref_image = None
            
    # Espace de recherche
    omega_values = [round(x, 2) for x in np.arange(0.50, 1.01, 0.05)]
    
    results = []
    img_result_dir = output_dir / image_path.stem
    img_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Début de l'itération sur {len(omega_values)} valeurs de omega...")

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
            
            row = {
                'filename': filename,
                'omega': omega,
                'patch_size': current_config['algorithm'].get('patch_size', 15),
                't0': current_config['algorithm'].get('t0', 0.1)
            }
            
            # Calcul NR Metrics
            nr_scores = metrics.compute_nr_metrics(hazy_image, restored_image)
            row.update(nr_scores)
            
            # Calcul FR Metrics
            if ref_image is not None:
                fr_scores = metrics.compute_fr_metrics(restored_image, ref_image)
                row.update(fr_scores)
                
            results.append(row)
            
        except Exception as e:
            logger.error(f"Echec pour omega={omega}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        
        # Tri des colonnes
        first_cols = ['filename', 'omega']
        metrics_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + metrics_cols]
        
        csv_path = img_result_dir / "metrics_report.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Rapport CSV généré : {csv_path}")
        
        logger.info("Génération des graphiques complets...")
        generate_plots(df, img_result_dir)
        logger.info("Terminé.")

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