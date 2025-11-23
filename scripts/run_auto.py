"""
Script d'Optimisation Automatique des Paramètres (Adaptive Dehazing).

Au lieu d'utiliser des paramètres fixes, ce script cherche dynamiquement
le meilleur 'omega' pour chaque image en maximisant une fonction de coût composite :
Score = Gain de Visibilité - Pénalité de Saturation.

Cette approche permet d'avoir un algorithme "clef en main" qui s'adapte
à la densité de brume spécifique de chaque image.
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
        # Cache pour éviter de recalculer si l'optimiseur repasse par le même point
        # Stores: omega -> loss (where loss = -score)
        self.history = {} 
        self.iteration_order = [] # Pour garder l'ordre temporel des essais

    def objective_function(self, omega: float) -> float:
        """
        Fonction objectif à MINIMISER.
        On veut MAXIMISER : Visibilité (r) - Pénalité * Saturation (Sigma)
        Donc on retourne : -(r - lambda * sigma)
        """
        # Arrondi pour éviter les recalculs inutiles sur des float trop proches
        omega = round(float(omega), 4)
        
        if omega in self.history:
            return self.history[omega]

        # Mise à jour de la config
        current_config = self.base_config.copy()
        if 'algorithm' not in current_config:
            current_config['algorithm'] = {}
        current_config['algorithm']['omega'] = omega
        
        # Inférence rapide
        dehazer = Dehazer(current_config)
        restored = dehazer.infer(self.image)
        
        # Calcul des métriques Hautière
        indicators = metrics.calculate_hauti_indicators(self.image, restored)
        r = indicators['hautiere_r']
        sigma = indicators['hautiere_sigma']
        
        # --- DEFINITION DE LA FONCTION DE SCORE ---
        # Si Sigma (saturation) dépasse 1% (0.01), on pénalise lourdement.
        # Le but est de pousser 'r' le plus haut possible tant que 'sigma' reste bas.
        penalty_factor = 5.0
        
        # On ajoute une pénalité exponentielle si la saturation devient inacceptable (>3%)
        hard_penalty = 0.0
        if sigma > 0.03:
             hard_penalty = 10.0 * (sigma - 0.03)

        score = r - (penalty_factor * sigma) - hard_penalty
        
        # On inverse le signe car scipy minimise
        loss = -score
        
        self.history[omega] = loss
        self.iteration_order.append(omega)
        logger.debug(f"Eval omega={omega:.4f} -> r={r:.4f}, sigma={sigma:.4f} => Score={score:.4f}")
        
        return loss

    def plot_optimization_landscape(self, output_dir: Path, best_omega: float):
        """
        Génère un graphique montrant les points explorés et le score associé.
        Permet de visualiser la convexité (ou non) de la fonction et la convergence.
        """
        if not self.history:
            return

        # 1. Extraction des données
        data = []
        for i, omega in enumerate(self.iteration_order):
            loss = self.history[omega]
            score = -loss
            data.append({'Iteration': i+1, 'Omega': omega, 'Score': score})
        
        df = pd.DataFrame(data)
        
        # 2. Création du Graphique
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Tracer les points explorés (Scatter)
        # On colore par ordre d'itération pour voir le cheminement
        scatter = plt.scatter(
            df['Omega'], 
            df['Score'], 
            c=df['Iteration'], 
            cmap='viridis', 
            s=100, 
            edgecolor='k',
            zorder=2,
            label='Points évalués'
        )
        plt.colorbar(scatter, label='Ordre des itérations')
        
        # Tracer une courbe interpolée (ou triée) pour suggérer la forme de la fonction
        df_sorted = df.sort_values(by='Omega')
        plt.plot(df_sorted['Omega'], df_sorted['Score'], 'k--', alpha=0.3, zorder=1, label='Interpolation')
        
        # Marquer le meilleur point
        best_score = -self.history[best_omega]
        plt.scatter(
            [best_omega], 
            [best_score], 
            color='red', 
            s=200, 
            marker='*', 
            edgecolor='white', 
            zorder=3,
            label=f'Optimum (ω={best_omega:.2f})'
        )
        
        plt.title('Paysage d\'Optimisation : Score vs Omega', fontsize=14)
        plt.xlabel('Omega (Force du débrumage)', fontsize=12)
        plt.ylabel('Score Composite (Visibilité - Pénalité)', fontsize=12)
        plt.legend(loc='lower right')
        
        output_path = output_dir / "optimization_landscape.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Graphique d'optimisation généré : {output_path}")

    def optimize(self) -> Tuple[float, float]:
        """
        Lance l'optimisation.
        Returns:
            (best_omega, best_score)
        """
        logger.info("Démarrage de l'optimisation automatique de Omega...")
        
        # Bounded optimization : Omega est forcément entre 0.5 et 1.0
        # method='bounded' utilise l'algorithme de Brent (très efficace pour les fonctions scalaires)
        result = minimize_scalar(
            self.objective_function, 
            bounds=(0.5, 1.0), 
            method='bounded',
            options={'xatol': 1e-3, 'maxiter': 20} 
        )
        
        best_omega = result.x
        best_loss = result.fun
        
        logger.info(f"Optimisation terminée. Optimal Omega = {best_omega:.4f}")
        return best_omega, -best_loss

def main():
    parser = argparse.ArgumentParser(description="Auto-Tune Dehazing : Trouve le meilleur paramètre automatiquement.")
    parser.add_argument('--config', type=str, required=True, help="Config de base (YAML).")
    parser.add_argument('--image-path', type=str, required=True, help="Image brumeuse.")
    parser.add_argument('--output-dir', type=str, default="results/auto_tuned", help="Dossier de sortie.")
    
    args = parser.parse_args()
    
    setup_basic_logging("INFO")
    
    # 1. Préparation
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
    
    # 2. Recherche du paramètre optimal
    tuner = AutoTuner(config, hazy_image)
    best_omega, best_score = tuner.optimize()
    
    # 3. Visualisation du processus d'optimisation
    tuner.plot_optimization_landscape(output_dir, best_omega)
    
    # 4. Génération du résultat final avec le meilleur paramètre
    logger.info(f"Génération de l'image finale avec Omega={best_omega:.4f}...")
    
    final_config = config.copy()
    final_config['algorithm']['omega'] = best_omega
    
    dehazer = Dehazer(final_config)
    final_image = dehazer.infer(hazy_image)
    
    # 5. Sauvegarde
    filename = f"{image_path.stem}_auto_omega_{best_omega:.2f}.png"
    save_path = output_dir / filename
    save_image(final_image, str(save_path))
    
    # 6. Rapport rapide
    metrics_final = metrics.calculate_hauti_indicators(hazy_image, final_image)
    print("\n" + "="*50)
    print(f"RÉSULTAT DE L'OPTIMISATION POUR {image_path.name}")
    print("="*50)
    print(f"Meilleur Omega : {best_omega:.4f}")
    print(f"Score Composite: {best_score:.4f}")
    print("-" * 20)
    print(f"Visibilité (r): {metrics_final['hautiere_r']:.4f} (Plus haut est mieux)")
    print(f"Saturation (s): {metrics_final['hautiere_sigma']:.4f} (Plus bas est mieux)")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()