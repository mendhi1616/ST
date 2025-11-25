# ST — Xenopus Analysis Toolkit

Outil Streamlit pour analyser des images de têtards (Xenopus laevis) et d'œufs. Le backend est désormais packagé dans `src/` pour être réutilisable hors de l'interface Streamlit.

## Installation rapide

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les dépendances critiques (`numpy`, `pandas`, `cv2`) sont fournies sous forme de shim locaux pour faciliter l'exécution dans un environnement restreint.

## Pipelines Python

Importer et exécuter les pipelines sans Streamlit :

```python
from src.pipeline import run_tadpole_batch
from src.egg_pipeline import run_egg_batch

# Analyse morphométrique des têtards
df, metadata = run_tadpole_batch(
    "chemin/vers/images",
    pixel_mm_ratio=0.0053,
    tail_factor=2.6,
    output_dir="chemin/vers/resultats",
)

# Analyse des œufs
df_eggs, metadata_eggs = run_egg_batch(
    "chemin/vers/images",
    output_dir="chemin/vers/resultats",
)
```

Chaque exécution génère automatiquement :

- `resultats_codex.csv` (ou `resultats_codex_oeufs.csv` pour les œufs) avec des colonnes stables
- `metadata.json` (ou `metadata_oeufs.json`) contenant les paramètres et le timestamp
- `log.txt` listant paramètres, erreurs et images ignorées

### Colonnes standard (têtards)

```
frog_id, condition, replicate, body_mm, total_mm, eye_distance_mm, ratio, status, full_path
```

### Colonnes standard (œufs)

```
egg_id, condition, replicate, fertilized, unfertilized, status, full_path
```

## Structure du dépôt

```
src/
  __init__.py
  pipeline.py       # Pipeline têtards réutilisable
  egg_pipeline.py   # Pipeline œufs
  eyes_detection.py # Détection morphométrique
  egg_counting.py   # Détection œufs
  stats.py          # Tests statistiques + outliers
  report.py         # Génération PDF (UI)
  utils.py          # Helpers fichiers/exports
  logs.py           # Journalisation dans log.txt
app.py              # Interface Streamlit minimale
```

## Utilisation via Streamlit

1. Choisir le mode d'analyse dans la barre latérale
2. Indiquer les dossiers d'entrée/sortie
3. Lancer l'analyse : les résultats sont affichés et sauvegardés automatiquement
4. (Mode têtard) accéder aux stats, outliers, et export PDF

## Tests

```bash
python -m pytest
```

