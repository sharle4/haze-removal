# haze-removal

Ce projet est une implémentation en Python du papier de recherche "_Single Image Haze Removal Using Dark Channel Prior_" par **Kaiming He, Jian Sun, et Xiaoou Tang** _(2009)_. L'objectif est de reproduire fidèlement les algorithmes décrits pour supprimer la brume d'une seule image.

Le projet utilise à la fois la méthode de *soft matting* ainsi que l'alternative *guided filter* proposé par **Kaiming He, Jian Sun, et Xiaoou Tang** dans l'article "_Guided Image Filtering_" _(2010)_. _(La méthode soft matting est désactivée par défaut car prend énormément de temps)_

## Installation du projet :

```
#1. Clonage du dépôt
git clone [https://github.com/sharle4/haze-removal.git](https://github.com/sharle4/haze-removal.git)
cd haze-removal

#2. Création d'un environnement Python virtuel
py -m venv venv
#2.1. Activation sous Windows
source venv/Scripts/activate
#2.2. Activation sous Linux / MacOS
source venv/bin/activate

#3. Installation des dépendances
pip install -r requirements.txt
```


## Utilisation du projet :

### Batterie d'expériences :
1. Modifier le fichier `configs/experiment_default.yaml` qui définit les paramètres de l'expérience
2. Lancer le script `run_batch.py`
```
python scripts/run_batch.py --exp-config configs/experiment_default.yaml --output-dir experiments/
```
Ce script va créer un dossier dans `experiments/` contenant les résultats pour chaque combinaison de paramètres.

### Expérience unique :
```
python scripts/run_single.py --config configs/default.yaml --image-path images/city_haze.jpg --output-dir results/city_haze
```

### Interface web : 
L'interface web permet de faire plus facilement varier les paramètres et d'observer et comparer aisément les réssultats.
```
python api/main.py
```
