"""
Script CLI pour exécuter le désembuage sur une seule image.
"""

import argparse
import logging
from pathlib import Path

from dehazing import Dehazer
from dehazing.utils import load_config, read_image, save_image, setup_basic_logging

logger = logging.getLogger(__name__)

def main():
    """
    Fonction principale orchestrant un unique traitement.
    """
    parser = argparse.ArgumentParser(description="Exécute l'algorithme Dark Channel Prior sur une image.")
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier de configuration YAML.")
    parser.add_argument('--image-path', type=str, required=True, help="Chemin vers l'image brumeuse d'entrée.")
    parser.add_argument('--output-dir', type=str, required=True, help="Répertoire où sauvegarder le résultat.")
    parser.add_argument('--verbose', action='store_true', help="Active les logs détaillés (DEBUG).")
    
    args = parser.parse_args()
    
    # 0. Configuration des logs
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_basic_logging(log_level)
    
    # 1. Préparation des chemins
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = Path(args.image_path)
    output_image_path = output_dir / f"{image_path.stem}_dehazed.png"

    logger.info(f"Démarrage du traitement pour : {image_path}")

    try:
        # 2. Chargement de la configuration et de l'image
        config = load_config(args.config)
        hazy_image = read_image(str(image_path))
        
        # 3. Initialisation du Dehazer
        logger.info("Initialisation de l'algorithme...")
        dehazer = Dehazer(config)
        
        # 4. Exécution de l'inférence
        logger.info("Traitement en cours...")
        dehazed_image = dehazer.infer(hazy_image)
        
        # 5. Sauvegarde du résultat
        save_image(dehazed_image, str(output_image_path))
        logger.info(f"Succès ! Résultat sauvegardé dans : {output_image_path}")

    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {e}", exc_info=True)
        exit(1)

if __name__ == '__main__':
    main()