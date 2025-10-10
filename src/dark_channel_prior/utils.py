"""
Fonctions utilitaires générales pour le projet.
"""

import logging
import os
import itertools
from typing import Dict, Any, Iterator, Tuple

def setup_logging(log_dir: str, config: Dict[str, Any]):
    """
    Configure le système de logging pour écrire dans la console et un fichier.

    Args:
        log_dir (str): Répertoire où sauvegarder le fichier de log.
        config (Dict[str, Any]): Dictionnaire de configuration contenant le niveau de log.
    """
    log_level_str = config.get('logging', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_path = os.path.join(log_dir, 'run.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configuré. Niveau: {log_level_str}. Fichier: {log_path}")


def _deep_update(d: Dict, u: Dict) -> Dict:
    """Met à jour un dictionnaire de manière récursive."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _set_nested_key(d: Dict, key_path: str, value: Any):
    """Définit une valeur dans un dictionnaire imbriqué en utilisant un chemin de clé."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def generate_experiment_configs(exp_config: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Génère les configurations pour chaque run d'une expérience.

    Args:
        exp_config (Dict[str, Any]): Le contenu du fichier de configuration de l'expérience.

    Returns:
        Iterator[Tuple[str, Dict[str, Any]]]: Un itérateur de tuples, où chaque tuple
                                              contient le nom du run et son dictionnaire
                                              de configuration complet.
    """
    from .config import load_config
    
    base_config_path = exp_config['base_config']
    
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.getcwd(), base_config_path)

    base_config = load_config(base_config_path)

    param_grid = exp_config['parameter_grid']
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combination in itertools.product(*param_values):
        run_config = _deep_update({}, base_config)
        run_name_parts = []

        for param_name, value in zip(param_names, combination):
            _set_nested_key(run_config, param_name, value)
            
            short_name = param_name.split('.')[-1]
            run_name_parts.append(f"{short_name}-{value}")
        
        run_name = "_".join(run_name_parts)
        yield run_name, run_config