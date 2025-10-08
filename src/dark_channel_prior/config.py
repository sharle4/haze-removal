"""
Fonctions utilitaires pour charger et valider la configuration depuis un fichier YAML.
"""

import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path (str): Chemin vers le fichier de configuration YAML.

    Returns:
        Dict[str, Any]: Dictionnaire contenant la configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration chargée depuis : {config_path}")
        return config
    except FileNotFoundError:
        print(f"Erreur : Le fichier de configuration '{config_path}' est introuvable.")
        raise
    except yaml.YAMLError as e:
        print(f"Erreur lors du parsing du fichier YAML '{config_path}': {e}")
        raise
