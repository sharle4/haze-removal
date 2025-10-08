"""
Script principal pour exécuter une expérience de suppression de brume sur une seule image.
"""

import argparse
import os
import logging
import yaml
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dark_channel_prior import config as cfg
from src.dark_channel_prior import utils
from src.dark_channel_prior import io
from src.dark_channel_prior import algorithms as alg
from src.dark_channel_prior import preprocessing as prep
from src.dark_channel_prior import visualization as vis

def main(args):
    """
    Fonction principale orchestrant l'algorithme de suppression de brume.
    """
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = cfg.load_config(args.config)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    utils.setup_logging(args.output_dir, config)

    logging.info(f"Début de l'expérience pour l'image : {args.image_path}")
    
    # --- 1. Lecture de l'image ---
    hazy_image = io.read_image(args.image_path)
    if hazy_image is None:
        logging.error("Échec de la lecture de l'image. Arrêt de l'expérience.")
        return
    
    alg_config = config['algorithm']
    gf_config = config['guided_filter']

    # --- 2. Exécution de l'algorithme ---
    with tqdm(total=5, desc="Traitement de l'image") as pbar:
        logging.info("Calcul du canal sombre...")
        dark_channel = alg.get_dark_channel(hazy_image, alg_config['patch_size'])
        pbar.update(1)

        logging.info("Estimation de la lumière atmosphérique...")
        atmospheric_light = alg.estimate_atmospheric_light(
            hazy_image, dark_channel, alg_config['atmospheric_light_percentile']
        )
        logging.info(f"Lumière atmosphérique estimée (A) = {atmospheric_light}")
        pbar.update(1)

        logging.info("Estimation de la transmission initiale...")
        initial_transmission = alg.estimate_initial_transmission(
            hazy_image, atmospheric_light, alg_config['patch_size'], alg_config['omega']
        )
        pbar.update(1)

        logging.info("Affinement de la transmission avec le filtre guidé...")
        hazy_gray = prep.convert_to_grayscale(hazy_image)
        refined_transmission = alg.refine_transmission_guided_filter(
            initial_transmission, hazy_gray, gf_config['radius'], gf_config['epsilon']
        )
        pbar.update(1)
        
        logging.info("Récupération de la radiance de la scène...")
        scene_radiance = alg.recover_scene_radiance(
            hazy_image, atmospheric_light, refined_transmission, alg_config['t0']
        )
        pbar.update(1)

    # --- 3. Sauvegarde des résultats ---
    logging.info("Sauvegarde des résultats...")
    
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    io.save_image(scene_radiance, os.path.join(figures_dir, "result_dehazed.png"))
    vis.save_transmission_map(initial_transmission, os.path.join(figures_dir, "transmission_initial.png"))
    vis.save_transmission_map(refined_transmission, os.path.join(figures_dir, "transmission_refined.png"))
    
    vis.save_comparison_figure(
        hazy_image, scene_radiance, refined_transmission,
        os.path.join(figures_dir, "comparison.png")
    )
    
    logging.info("Expérience terminée avec succès.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exécute l'algorithme Dark Channel Prior sur une image.")
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier de configuration YAML.")
    parser.add_argument('--image-path', type=str, required=True, help="Chemin vers l'image brumeuse d'entrée.")
    parser.add_argument('--output-dir', type=str, required=True, help="Répertoire où sauvegarder les résultats.")
    
    args = parser.parse_args()
    main(args)