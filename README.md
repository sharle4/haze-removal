# haze-removal

Ce projet est une implémentation en Python du papier de recherche "Single Image Haze Removal Using Dark Channel Prior" par Kaiming He, Jian Sun, et Xiaoou Tang (2009). L'objectif est de reproduire fidèlement les algorithmes décrits pour supprimer la brume d'une seule image.

Le projet utilise à la fois la méthode de "soft matting" ainsi que l'altrenative "Guided Filter" proposé par Kaiming He, Jian Sun, et Xiaoou Tang dans l'article "Guided Image Filtering" (2010).

Utilisation classique du projet (sous windows) :

'git clone https://github.com/sharle4/haze-removal.git
py -m venv venv #création d'un environnement Python virtuel
source venv/Scripts/activate #sous windows, ou source venv/bin/activate sous Linux
pip install -r requirements.txt
$ python scripts/run_experiment.py #avec args facultatifs'