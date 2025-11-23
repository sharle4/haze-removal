"""
Script de traitement par lot avec Auto-Tuning individuel (Batch Optimization).

Ce script parcourt un dossier d'images, calcule le paramètre Omega optimal pour 
CHAQUE image individuellement, génère le résultat et compile un rapport statistique complet.

Usage:
    python scripts/run_folder.py --config configs/default.yaml --input-dir images/dataset --output-dir results/batch_experiment
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# Ajout du chemin src au path pour les imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

class RobustOptimizer:
    """
    Classe d'optimisation encapsulée. 
    Cherche le meilleur compromis Visibilité/Artefacts pour une image donnée.
    """
    def __init__(self, base_config: dict, image: np.ndarray):
        self.base_config = base_config
        self.image = image
        self.history = {}
        
        # Statistiques de référence pour éviter les dérives
        self.orig_mean_intensity = np.mean(image)

    def _calculate_cost(self, restored: np.ndarray, original: np.ndarray) -> float:
        """
        Fonction de coût composite (Objectif à minimiser).
        Score = - (Gain_Visibilité - Pénalité_Saturation - Pénalité_Assombrissement)
        """
        # 1. Gain de visibilité (Hautière r)
        h_metrics = metrics.calculate_hauti_indicators(original, restored)
        r = h_metrics['hautiere_r']
        
        # Plafonnement académique du gain : au-delà de 3.0, on considère que c'est du bruit
        gain_score = min(r, 3.0)
        
        # 2. Pénalité : Soft Saturation (Pixels brûlés ou bouchés)
        # On pénalise si > 2% de l'image est dans les extrêmes (0.05 - 0.95)
        flat_rest = restored.flatten()
        soft_saturation = (np.count_nonzero(flat_rest < 0.05) + np.count_nonzero(flat_rest > 0.95)) / flat_rest.size
        
        saturation_penalty = 0.0
        if soft_saturation > 0.02:
            saturation_penalty = np.exp(20.0 * (soft_saturation - 0.02)) - 1.0

        # 3. Pénalité : Préservation de la luminosité moyenne
        # On interdit une perte de luminosité globale > 40%
        rest_mean = np.mean(restored)
        brightness_ratio = rest_mean / (self.orig_mean_intensity + 1e-6)
        
        darkness_penalty = 0.0
        if brightness_ratio < 0.60:
            darkness_penalty = np.exp(10.0 * (0.60 - brightness_ratio)) - 1.0

        # Score final (Pondération empirique pour la robustesse)
        score = gain_score - (1.5 * saturation_penalty) - (2.0 * darkness_penalty)
        return -score

    def objective(self, omega: float) -> float:
        """Fonction appelée par le minimiseur."""
        omega = round(float(omega), 4)
        if omega in self.history:
            return self.history[omega]

        # Configuration temporaire
        current_config = self.base_config.copy()
        if 'algorithm' not in current_config:
            current_config['algorithm'] = {}
        current_config['algorithm']['omega'] = omega
        
        # Inférence
        try:
            dehazer = Dehazer(current_config)
            # Note: Le Dehazer utilise la config, donc Sky Detection est actif si activé dans le YAML
            restored = dehazer.infer(self.image)
            loss = self._calculate_cost(restored, self.image)
        except Exception as e:
            logger.warning(f"Erreur d'inférence pour omega={omega}: {e}")
            loss = 0.0 # Pénalité max (score nul)

        self.history[omega] = loss
        return loss

    def find_optimal_omega(self) -> float:
        """Exécute l'optimisation bornée."""
        result = minimize_scalar(
            self.objective, 
            bounds=(0.5, 0.98), 
            method='bounded',
            options={'xatol': 1e-3, 'maxiter': 20}
        )
        return result.x


def process_dataset(config: dict, input_dir: Path, output_dir: Path):
    """
    Traite tout le dossier et génère le rapport CSV.
    """
    # Extensions d'images supportées
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not image_files:
        logger.error(f"Aucune image trouvée dans {input_dir}")
        return

    logger.info(f"Traitement de {len(image_files)} images...")
    
    results_data = []
    
    # Barre de progression
    pbar = tqdm(image_files, desc="Batch Processing")
    
    for img_path in pbar:
        row = {'filename': img_path.name}
        start_time = time.time()
        
        try:
            pbar.set_postfix_str(f"Processing {img_path.name[:10]}...")
            
            # 1. Lecture
            original_image = read_image(str(img_path))
            
            # 2. Optimisation
            optimizer = RobustOptimizer(config, original_image)
            best_omega = optimizer.find_optimal_omega()
            row['optimal_omega'] = round(best_omega, 4)
            
            # 3. Génération Finale
            final_config = config.copy()
            final_config['algorithm']['omega'] = best_omega
            
            dehazer = Dehazer(final_config)
            restored_image = dehazer.infer(original_image)
            
            # 4. Sauvegarde
            save_name = f"{img_path.stem}_dehazed.png"
            save_image(restored_image, str(output_dir / save_name))
            
            # 5. Calcul des Métriques (NR)
            # On calcule tout pour le rapport académique
            nr_metrics = metrics.compute_nr_metrics(original_image, restored_image)
            row.update(nr_metrics)
            
            row['status'] = 'success'

        except Exception as e:
            logger.error(f"Echec sur {img_path.name}: {e}")
            row['status'] = 'failed'
            row['error_msg'] = str(e)
        
        row['processing_time'] = round(time.time() - start_time, 2)
        results_data.append(row)

    # --- Génération du Rapport ---
    if results_data:
        df = pd.DataFrame(results_data)
        
        # Réorganisation des colonnes pour la lisibilité
        cols = ['filename', 'status', 'optimal_omega', 'processing_time', 
                'hautiere_e', 'hautiere_r', 'hautiere_sigma', 
                'colorfulness_out', 'dark_channel_residual']
        # On garde les colonnes existantes, en mettant les importantes au début
        final_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[final_cols]
        
        # Sauvegarde CSV
        csv_path = output_dir / "batch_report.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Rapport détaillé sauvegardé : {csv_path}")
        
        # Affichage des statistiques globales (Console)
        if 'hautiere_r' in df.columns:
            mean_r = df[df['status']=='success']['hautiere_r'].mean()
            mean_omega = df[df['status']=='success']['optimal_omega'].mean()
            logger.info("--- RÉSULTATS GLOBAUX ---")
            logger.info(f"Images traitées avec succès : {len(df[df['status']=='success'])} / {len(df)}")
            logger.info(f"Moyenne Omega Optimal     : {mean_omega:.3f}")
            logger.info(f"Moyenne Gain Visibilité (r): {mean_r:.3f}")
            logger.info("---------------------------")

def main():
    parser = argparse.ArgumentParser(description="Batch Auto-Tuned Dehazing for Datasets.")
    parser.add_argument('--config', type=str, required=True, help="Config de base (YAML).")
    parser.add_argument('--input-dir', type=str, required=True, help="Dossier contenant les images brumeuses.")
    parser.add_argument('--output-dir', type=str, default="results/batch_run", help="Dossier de sortie.")
    
    args = parser.parse_args()
    
    setup_basic_logging("INFO")
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Dossier d'entrée introuvable : {input_dir}")
        exit(1)
        
    config = load_config(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_dataset(config, input_dir, output_dir)

if __name__ == '__main__':
    main()