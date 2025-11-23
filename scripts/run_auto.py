"""
Script d'Optimisation Automatique des Paramètres (Adaptive Dehazing).

Version Académique Avancée :
Utilise une fonction de coût composite régularisée pour éviter le sur-ajustement (over-dehazing).
Prend en compte :
1. Le gain de visibilité (Hautière r)
2. La saturation "douce" (Soft Saturation) pour éviter les noirs bouchés
3. La préservation de la luminosité moyenne (Brightness Fidelity)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# --- VISUALISATION ---
import matplotlib
matplotlib.use('Agg') # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------

# Ajout du chemin src au path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging
from dehazing import metrics

logger = logging.getLogger(__name__)

class AutoTuner:
    def __init__(self, config: dict, image: np.ndarray):
        self.base_config = config
        self.image = image
        self.history = {} 
        self.iteration_order = []
        
        # Pré-calcul des statistiques de l'image originale pour la comparaison
        self.orig_mean_intensity = np.mean(image)
        self.orig_dark_pixels = np.count_nonzero(image < 0.05) / image.size

    def calculate_advanced_score(self, restored: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """
        Calcule un score de qualité plus robuste que la simple métrique de Hautière.
        """
        # 1. Indicateurs de base (Hautière)
        h_metrics = metrics.calculate_hauti_indicators(original, restored)
        r = h_metrics['hautiere_r']
        
        # 2. Soft Saturation (Zone critique)
        # On regarde les pixels < 5% (ombres perdues) et > 95% (blancs brûlés)
        # C'est beaucoup plus sensible que le clipping strict à 0 ou 1
        flat_rest = restored.flatten()
        soft_low = np.count_nonzero(flat_rest < 0.05) / flat_rest.size
        soft_high = np.count_nonzero(flat_rest > 0.95) / flat_rest.size
        soft_saturation = soft_low + soft_high
        
        # 3. Perte de Luminosité (Dimming Artifact)
        # Le DCP a tendance à trop assombrir. On vérifie la perte moyenne.
        rest_mean = np.mean(restored)
        brightness_ratio = rest_mean / (self.orig_mean_intensity + 1e-6)
        
        return {
            'r': r,
            'soft_saturation': soft_saturation,
            'brightness_ratio': brightness_ratio
        }

    def objective_function(self, omega: float) -> float:
        """
        Fonction objectif 'Perceptuelle' à MINIMISER.
        """
        omega = round(float(omega), 4)
        if omega in self.history:
            return self.history[omega]

        # --- Inférence ---
        current_config = self.base_config.copy()
        if 'algorithm' not in current_config:
            current_config['algorithm'] = {}
        current_config['algorithm']['omega'] = omega
        
        dehazer = Dehazer(current_config)
        restored = dehazer.infer(self.image)
        
        # --- Analyse ---
        stats = self.calculate_advanced_score(restored, self.image)
        
        # --- Construction du Score (Heuristique Académique) ---
        
        # 1. Gain (Visibilité)
        # On plafonne le gain utile. Au-delà de r=3.0, c'est souvent du bruit.
        gain_score = min(stats['r'], 3.0) 
        
        # 2. Pénalité : Saturation Douce
        # Si plus de 2% de l'image entre dans les zones extrêmes, on pénalise exponentiellement.
        # Facteur 20: 5% de saturation => exp(1) ~= 2.7 points de pénalité (énorme)
        saturation_penalty = 0.0
        if stats['soft_saturation'] > 0.02: 
             saturation_penalty = np.exp(20.0 * (stats['soft_saturation'] - 0.02)) - 1.0

        # 3. Pénalité : Effondrement de la luminosité
        # Si l'image perd plus de 40% de sa luminosité (ratio < 0.6), c'est inacceptable.
        darkness_penalty = 0.0
        if stats['brightness_ratio'] < 0.60:
            # Pénalité très forte pour interdire les images trop sombres
            darkness_penalty = np.exp(10.0 * (0.60 - stats['brightness_ratio'])) - 1.0

        # Score Final
        score = gain_score - (1.5 * saturation_penalty) - (2.0 * darkness_penalty)
        
        # Logs détaillés pour debugguer le comportement
        logger.debug(
            f"Ω={omega:.3f} | r={stats['r']:.2f} | "
            f"Sat={stats['soft_saturation']:.1%} (Pen={saturation_penalty:.2f}) | "
            f"Bri={stats['brightness_ratio']:.2f} (Pen={darkness_penalty:.2f}) | "
            f"=> Score={score:.4f}"
        )
        
        loss = -score
        self.history[omega] = loss
        self.iteration_order.append(omega)
        
        return loss

    def plot_optimization_landscape(self, output_dir: Path, best_omega: float):
        """
        Génère un graphique montrant les points explorés et le score associé.
        """
        if not self.history:
            return

        data = []
        for i, omega in enumerate(self.iteration_order):
            loss = self.history[omega]
            score = -loss
            data.append({'Iteration': i+1, 'Omega': omega, 'Score': score})
        
        df = pd.DataFrame(data)
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Tracer la courbe triée
        df_sorted = df.sort_values(by='Omega')
        plt.plot(df_sorted['Omega'], df_sorted['Score'], 'b-', alpha=0.5, zorder=1, label='Profil de Score')
        
        # Points évalués
        scatter = plt.scatter(
            df['Omega'], df['Score'], c=df['Iteration'], cmap='viridis', 
            s=80, edgecolor='k', zorder=2, label='Itérations'
        )
        plt.colorbar(scatter, label='Ordre')
        
        # Optimum
        # Récupération sécurisée du score
        rounded_best_omega = round(float(best_omega), 4)
        if rounded_best_omega in self.history:
            best_score = -self.history[rounded_best_omega]
        else:
            best_score = -self.objective_function(best_omega) # Fallback

        plt.scatter(
            [best_omega], [best_score], color='red', s=200, marker='*', 
            edgecolor='white', zorder=3, label=f'Optimum ({best_omega:.3f})'
        )
        
        plt.title('Recherche du Paramètre Optimal (Fonction de Coût Régularisée)', fontsize=14)
        plt.xlabel('Omega', fontsize=12)
        plt.ylabel('Score de Qualité', fontsize=12)
        plt.legend(loc='lower right')
        
        output_path = output_dir / "optimization_landscape.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

    def optimize(self) -> Tuple[float, float]:
        logger.info("Démarrage de l'optimisation avancée...")
        
        # On réduit légèrement la borne haute à 0.98 pour éviter les artefacts de bordure extrêmes
        result = minimize_scalar(
            self.objective_function, 
            bounds=(0.5, 0.99), 
            method='bounded',
            options={'xatol': 1e-3, 'maxiter': 25} 
        )
        return result.x, -result.fun

def main():
    parser = argparse.ArgumentParser(description="Auto-Tune Dehazing : Version Académique Avancée")
    parser.add_argument('--config', type=str, required=True, help="Config YAML")
    parser.add_argument('--image-path', type=str, required=True, help="Image d'entrée")
    parser.add_argument('--output-dir', type=str, default="results/auto_tuned", help="Sortie")
    
    args = parser.parse_args()
    setup_basic_logging("INFO") # Debug activé pour voir les pénalités
    
    image_base_dir = Path('./images')
    output_base_dir = Path('./results')
    
    config = load_config(args.config)
    image_path = image_base_dir / args.image_path
    output_dir = output_base_dir / args.output_dir
    
    if not image_path.exists():
        logger.error(f"Image introuvable : {image_path}")
        exit(1)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    hazy_image = read_image(str(image_path))
    
    tuner = AutoTuner(config, hazy_image)
    best_omega, best_score = tuner.optimize()
    
    try:
        tuner.plot_optimization_landscape(output_dir, best_omega)
    except Exception as e:
        logger.warning(f"Erreur plot: {e}")
    
    logger.info(f"Génération finale avec Omega={best_omega:.4f}")
    final_config = config.copy()
    final_config['algorithm']['omega'] = best_omega
    
    dehazer = Dehazer(final_config)
    final_image = dehazer.infer(hazy_image)
    
    filename = f"{Path(args.image_path).stem}_auto_omega_{best_omega:.2f}.png"
    save_image(final_image, str(output_dir / filename))

if __name__ == '__main__':
    main()