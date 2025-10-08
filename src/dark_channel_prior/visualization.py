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
    result: np.ndarray, 
    transmission: np.ndarray, 
    save_path: str
):
    """
    Sauvegarde une figure comparative montrant l'image originale, le résultat et la carte de transmission.

    Args:
        original (np.ndarray): Image originale brumeuse.
        result (np.ndarray): Image sans brume.
        transmission (np.ndarray): Carte de transmission affinée.
        save_path (str): Chemin où sauvegarder la figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Image Originale Brumeuse")
    axes[0].axis('off')
    
    axes[1].imshow(result)
    axes[1].set_title("Image Sans Brume")
    axes[1].axis('off')
    
    axes[2].imshow(transmission, cmap='gray')
    axes[2].set_title("Carte de Transmission (Profondeur)")
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure de comparaison sauvegardée à : {save_path}")
