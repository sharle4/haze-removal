"""
Script pour exécuter une unique expérience de suppression de brume (wrapper).
"""

import argparse
import os
import yaml

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dark_channel_prior import config as cfg
from src.dark_channel_prior import utils
from src.dark_channel_prior import io
from src.dark_channel_prior import runner

def main():
    """
    Fonction principale orchestrant un unique traitement.
    """
    parser = argparse.ArgumentParser(description="Exécute l'algorithme Dark Channel Prior sur une image.")
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier de configuration YAML.")
    parser.add_argument('--image-path', type=str, required=True, help="Chemin vers l'image brumeuse d'entrée.")
    parser.add_argument('--output-dir', type=str, required=True, help="Répertoire où sauvegarder les résultats.")
    
    args = parser.parse_args()
    
    # --- 1. Configuration et préparation ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = cfg.load_config(args.config)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    utils.setup_logging(args.output_dir, config)

    # --- 2. Lecture de l'image ---
    hazy_image = io.read_image(args.image_path)
    if hazy_image is None:
        print("Échec de la lecture de l'image. Arrêt.")
        return
        
    # --- 3. Lancement du traitement ---
    runner.process_single_image(hazy_image, config, args.output_dir)
    
    print(f"\nTraitement terminé. Résultats sauvegardés dans : {args.output_dir}")

if __name__ == '__main__':
    main()
