"""
Fonctions pour la visualisation et la sauvegarde de figures.
"""

import matplotlib
matplotlib.use('Agg') # Evite d'utiliser tkinter
import matplotlib.pyplot as plt
import numpy as np

def save_transmission_map(transmission_map: np.ndarray, save_path: str, cmap: str = 'gray'):
    """
    Sauvegarde la carte de transmission en tant qu'image en utilisant matplotlib.

    Args:
        transmission_map (np.ndarray): Carte de transmission 2D.
        save_path (str): Chemin de destination du fichier.
        cmap (str): Colormap à utiliser pour la visualisation.
    """
    try:
        plt.imsave(save_path, transmission_map, cmap=cmap, vmin=0, vmax=1)
        print(f"Carte de transmission sauvegardée à : {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la carte de transmission : {e}")

def save_comparison_figure(
    original: np.ndarray,
    results_dict: dict,
    transmissions_dict: dict,
    save_path: str
):
    """
    Sauvegarde une figure comparative montrant l'image originale, les différents
    résultats de suppression de brume et leurs cartes de transmission.

    Args:
        original (np.ndarray): Image originale brumeuse.
        results_dict (dict): Dictionnaire de { 'Nom de la méthode': image_resultat }.
        transmissions_dict (dict): Dictionnaire de { 'Nom de la méthode': carte_transmission }.
        save_path (str): Chemin où sauvegarder la figure.
    """
    num_methods = len(results_dict)
    num_cols = 1 + num_methods
    
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 10))
    
    # --- Ligne 1: Images ---
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Image Originale Brumeuse")
    axes[0, 0].axis('off')

    for i, (method_name, result_img) in enumerate(results_dict.items()):
        ax = axes[0, i + 1]
        ax.imshow(result_img)
        ax.set_title(f"Résultat ({method_name})")
        ax.axis('off')
        
    # --- Ligne 2: Cartes de transmission ---

    axes[1, 0].axis('off') 
    
    for i, (method_name, trans_map) in enumerate(transmissions_dict.items()):
        ax = axes[1, i + 1]
        im = ax.imshow(trans_map, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Transmission ({method_name})")
        ax.axis('off')

    for i in range(num_methods + 1, num_cols):
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure de comparaison sauvegardée à : {save_path}")