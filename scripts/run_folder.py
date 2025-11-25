"""
Script de traitement par lot (Batch) comparatif & Analytique.
Compare les méthodes He et Fattal sur un dossier d'images brumeuses.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

def normalize_column(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

def generate_comprehensive_analytics(df: pd.DataFrame, output_dir: Path):
    """
    Génère une suite complète de graphiques pour l'analyse.
    """
    analytics_dir = output_dir / "Analytics_Graphs"
    analytics_dir.mkdir(exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper")
    
    metrics_found = set()
    for col in df.columns:
        if col.startswith('He_'):
            metric_name = col[3:]
            if f'Fattal_{metric_name}' in df.columns:
                metrics_found.add(metric_name)
    
    logger.info(f"Génération des graphiques pour les métriques : {metrics_found}")

    for metric in metrics_found:
        plt.figure(figsize=(7, 7))
        
        x_data = df[f'He_{metric}']
        y_data = df[f'Fattal_{metric}']
        
        sns.scatterplot(x=x_data, y=y_data, alpha=0.7, s=60, edgecolor='k')
        
        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Égalité')
        
        plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.05, label='Fattal meilleur (si >)')
        plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='blue', alpha=0.05, label='He meilleur (si >)')
        
        plt.title(f"Comparaison Paire à Paire : {metric.upper()}")
        plt.xlabel(f"Score He et al. ({metric})")
        plt.ylabel(f"Score Fattal ({metric})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analytics_dir / f"Scatter_{metric}.png", dpi=300)
        plt.close()

    long_data = []
    for _, row in df.iterrows():
        for metric in metrics_found:
            long_data.append({'Method': 'He et al.', 'Metric': metric, 'Value': row[f'He_{metric}']})
            long_data.append({'Method': 'Fattal', 'Metric': metric, 'Value': row[f'Fattal_{metric}']})
    
    df_long = pd.DataFrame(long_data)
    
    for metric in metrics_found:
        plt.figure(figsize=(6, 5))
        subset = df_long[df_long['Metric'] == metric]
        
        sns.boxplot(data=subset, x='Method', y='Value', palette=['tab:blue', 'tab:green'], width=0.5)
        sns.swarmplot(data=subset, x='Method', y='Value', color='k', alpha=0.3, size=4)
        
        plt.title(f"Distribution : {metric.upper()}")
        plt.ylabel("Valeur")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(analytics_dir / f"Boxplot_{metric}.png", dpi=300)
        plt.close()

    if 'He_time' in df.columns and 'Fattal_time' in df.columns:
        plt.figure(figsize=(8, 5))
        df_time = pd.DataFrame({
            'He et al.': df['He_time'],
            'Fattal': df['Fattal_time']
        })
        df_time_melted = df_time.melt(var_name='Method', value_name='Seconds')
        
        sns.barplot(data=df_time_melted, x='Method', y='Seconds', capsize=.1, palette='pastel', errorbar='sd')
        plt.title("Temps d'Exécution Moyen (avec écart-type)")
        plt.ylabel("Temps (s)")
        plt.tight_layout()
        plt.savefig(analytics_dir / "Time_Comparison.png", dpi=300)
        plt.close()

    if 'psnr' in metrics_found and 'ssim' in metrics_found:
        try:
            radar_metrics = ['psnr', 'ssim', 'hautiere_r', 'colorfulness_out']
            radar_metrics = [m for m in radar_metrics if m in metrics_found]
            
            if len(radar_metrics) >= 3:
                means_he = []
                means_fattal = []
                
                for m in radar_metrics:
                    all_vals = pd.concat([df[f'He_{m}'], df[f'Fattal_{m}']])
                    min_v, max_v = all_vals.min(), all_vals.max()
                    
                    mean_he = (df[f'He_{m}'].mean() - min_v) / (max_v - min_v)
                    mean_fattal = (df[f'Fattal_{m}'].mean() - min_v) / (max_v - min_v)
                    
                    means_he.append(mean_he)
                    means_fattal.append(mean_fattal)
                
                labels = [m.upper() for m in radar_metrics]
                num_vars = len(labels)
                angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
                angles += angles[:1]
                
                means_he += means_he[:1]
                means_fattal += means_fattal[:1]
                
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.plot(angles, means_he, color='tab:blue', linewidth=2, label='He et al.')
                ax.fill(angles, means_he, color='tab:blue', alpha=0.1)
                
                ax.plot(angles, means_fattal, color='tab:green', linewidth=2, label='Fattal')
                ax.fill(angles, means_fattal, color='tab:green', alpha=0.1)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                plt.title("Profil de Performance Moyen (Normalisé)", y=1.05)
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.savefig(analytics_dir / "Radar_Summary.png", dpi=300)
                plt.close()
        except Exception as e:
            logger.warning(f"Erreur lors du Radar Chart : {e}")



def flatten_metrics(metrics_dict, prefix):
    """Ajoute un préfixe aux clés des métriques (ex: 'psnr' -> 'He_psnr')."""
    return {f"{prefix}_{k}": v for k, v in metrics_dict.items()}

def make_comparison_strip(original, he, fattal):
    """Crée une bande horizontale simple: Original | He | Fattal"""
    h, w, c = original.shape
    if he.dtype == np.uint8: he = he.astype(np.float32) / 255.0
    if fattal.dtype == np.uint8: fattal = fattal.astype(np.float32) / 255.0
    
    strip = np.hstack((original, he, fattal))
    return strip

def process_dataset(config, input_dir, output_dir, ref_dir=None):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not image_files:
        logger.error("Aucune image trouvée.")
        return

    (output_dir / "He").mkdir(parents=True, exist_ok=True)
    (output_dir / "Fattal").mkdir(parents=True, exist_ok=True)
    (output_dir / "Comparisons").mkdir(parents=True, exist_ok=True)

    results_data = []
    dehazer = Dehazer(config)
    
    logger.info(f"Démarrage du batch sur {len(image_files)} images.")
    if ref_dir:
        logger.info(f"Mode Full-Reference activé avec : {ref_dir}")
    else:
        logger.info("Mode No-Reference (pas de dossier Ground Truth fourni).")

    for img_path in tqdm(image_files, desc="Processing"):
        row = {'filename': img_path.name}
        
        try:
            # 1. Lecture Image
            original = read_image(str(img_path))
            
            # Lecture Référence (si dispo)
            ref_img = None
            if ref_dir:
                ref_p = ref_dir / img_path.name
                if ref_p.exists():
                    ref_img = read_image(str(ref_p))
                    if ref_img.shape != original.shape:
                        logger.warning(f"Dimensions incohérentes pour {img_path.name}. Métriques FR ignorées.")
                        ref_img = None

            # 2. Inférence HE
            t0 = time.time()
            res_he = dehazer.infer_he(original)
            row['He_time'] = time.time() - t0
            save_image(res_he, str(output_dir / "He" / img_path.name))

            # 3. Inférence FATTAL
            t0 = time.time()
            try:
                res_fattal = dehazer.infer_fattal(original)
                row['Fattal_time'] = time.time() - t0
                save_image(res_fattal, str(output_dir / "Fattal" / img_path.name))
                fattal_success = True
            except Exception as e:
                logger.warning(f"Fattal échec sur {img_path.name}: {e}")
                res_fattal = np.zeros_like(original)
                row['Fattal_time'] = 0
                fattal_success = False

            # 4. Comparaison Visuelle
            comp_img = make_comparison_strip(original, res_he, res_fattal)
            save_image(comp_img, str(output_dir / "Comparisons" / f"comp_{img_path.name}"))

            # 5. Calcul Métriques (HE)
            m_he_nr = metrics.compute_nr_metrics(original, res_he)
            row.update(flatten_metrics(m_he_nr, "He"))
            
            if ref_img is not None:
                m_he_fr = metrics.compute_fr_metrics(res_he, ref_img)
                row.update(flatten_metrics(m_he_fr, "He"))

            # 6. Calcul Métriques (FATTAL)
            if fattal_success:
                m_fa_nr = metrics.compute_nr_metrics(original, res_fattal)
                row.update(flatten_metrics(m_fa_nr, "Fattal"))
                
                if ref_img is not None:
                    m_fa_fr = metrics.compute_fr_metrics(res_fattal, ref_img)
                    row.update(flatten_metrics(m_fa_fr, "Fattal"))

        except Exception as e:
            logger.error(f"Erreur globale sur {img_path.name}: {e}")
            continue
            
        results_data.append(row)

    # Export CSV
    if results_data:
        df = pd.DataFrame(results_data)
        
        cols = list(df.columns)
        priority = ['filename', 'He_psnr', 'Fattal_psnr', 'He_ssim', 'Fattal_ssim', 
                    'He_hautiere_r', 'Fattal_hautiere_r']
        ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        
        df = df[ordered_cols]
        csv_path = output_dir / "full_benchmark_report.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Rapport CSV sauvegardé : {csv_path}")
        
        logger.info("Génération des graphiques analytiques...")
        try:
            generate_comprehensive_analytics(df, output_dir)
            logger.info("Terminé. Graphiques disponibles dans 'Analytics_Graphs'.")
        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques : {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Batch Benchmark: He vs Fattal + Analytics")
    parser.add_argument('--config', type=str, required=True, help="Config YAML")
    parser.add_argument('--input-dir', type=str, required=True, help="Dossier images brumeuses")
    parser.add_argument('--output-dir', type=str, default="results/benchmark", help="Sortie")
    parser.add_argument('--ref-dir', type=str, default=None, help="Dossier images de référence (GT) pour métriques Full-Reference")
    
    args = parser.parse_args()
    setup_basic_logging("INFO")
    
    config = load_config(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    
    if not input_dir.exists():
        logger.error(f"Dossier introuvable: {input_dir}")
        exit(1)
        
    process_dataset(config, input_dir, output_dir, ref_dir)

if __name__ == '__main__':
    main()