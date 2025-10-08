"""
Fonctions utilitaires générales pour le projet.
"""

import logging
import os
from typing import Dict, Any

def setup_logging(log_dir: str, config: Dict[str, Any]):
    """
    Configure le système de logging pour écrire dans la console et un fichier.

    Args:
        log_dir (str): Répertoire où sauvegarder le fichier de log.
        config (Dict[str, Any]): Dictionnaire de configuration contenant le niveau de log.
    """
    log_level_str = config.get('logging', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    log_path = os.path.join(log_dir, 'run.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configuré. Niveau: {log_level_str}. Fichier: {log_path}")
