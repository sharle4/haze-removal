"""
Script d'orchestration pour lancer des batteries d'expériences en parallèle.
"""

import argparse
import os
import time
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dark_channel_prior import config as cfg
from src.dark_channel_prior import utils
from src.dark_channel_prior import io
from src.dark_channel_prior import runner

def worker_process(args: tuple):
    """
    Fonction exécutée par chaque processus worker.

    Args:
        args (tuple): Un tuple contenant (image_path, config, output_dir).
    """
    image_path, config, output_dir = args
    try:
        hazy_image = io.read_image(image_path)
        if hazy_image is not None:
            runner.process_single_image(hazy_image, config, output_dir)
        return output_dir, "Success", ""
    except Exception as e:
        print(f"Erreur dans le worker pour {output_dir}: {e}")
        return output_dir, "Failed", str(e)

def main():
    """
    Fonction principale pour orchestrer l'expérience.
    """
    parser = argparse.ArgumentParser(
        description="Lance une batterie d'expériences de suppression de brume en parallèle."
    )
    parser.add_argument(
        '--exp-config',
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration de l'expérience YAML."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Répertoire racine où sauvegarder les résultats de l'expérience."
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=cpu_count(),
        help=f"Nombre de processus workers à utiliser. Défaut: {cpu_count()} (nombre de cœurs CPU)."
    )
    args = parser.parse_args()

    start_time = time.time()
    
    # --- 1. Création du répertoire de sortie et chargement de la configuration ---
    exp_name = os.path.splitext(os.path.basename(args.exp_config))[0]
    experiment_root_dir = os.path.join(args.output_dir, f"{exp_name}_{int(start_time)}")
    os.makedirs(experiment_root_dir, exist_ok=True)
    
    print(f"Début de l'expérience : '{exp_name}'")
    print(f"Résultats sauvegardés dans : {experiment_root_dir}")

    exp_config = cfg.load_config(args.exp_config)
    
    # --- 2. Génération des configurations pour chaque test ---
    tasks = []
    configs_to_run = utils.generate_experiment_configs(exp_config)
    print(configs_to_run, type(configs_to_run))
    
    image_path = exp_config['image_path']
    if not os.path.exists(image_path):
        print(f"ERREUR: L'image d'entrée '{image_path}' est introuvable.")
        return

    for run_name, run_config in configs_to_run:
        run_output_dir = os.path.join(experiment_root_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        tasks.append((image_path, run_config, run_output_dir))

    print(f"{len(tasks)} combinaisons de paramètres à tester avec {args.workers} workers.")

    # --- 3. Exécution des tâches en parallèle ---
    results = []
    with Pool(processes=args.workers) as pool:
        with tqdm(total=len(tasks), desc="Progression de l'expérience") as pbar:
            for result in pool.imap_unordered(worker_process, tasks):
                results.append(result)
                pbar.update()

    # --- 4. Sauvegarde du compte-rendu de l'expérience ---
    summary_path = os.path.join(experiment_root_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_name', 'status', 'error_message', 'output_path'])
        
        for output_dir, status, error in sorted(results):
             run_name = os.path.basename(output_dir)
             writer.writerow([run_name, status, error, output_dir])

    end_time = time.time()
    print("\nExpérience terminée.")
    print(f"Durée totale : {end_time - start_time:.2f} secondes.")
    print(f"Résumé de l'expérience sauvegardé dans : {summary_path}")

if __name__ == '__main__':
    main()