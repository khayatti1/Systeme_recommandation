# Systeme recommandation

##  Description
**Systeme_recommandation** est une application Python conçue pour implémenter un système de recommandation.  
Elle utilise des bibliothèques de data science et de machine learning afin d’analyser les données utilisateurs et proposer des recommandations personnalisées.  
Une interface simple est développée avec **Streamlit** pour interagir avec le système.

## Fonctionnalités principales
- **Chargement et traitement des données** avec **pandas** et **PySpark**.
- **Recommandations basées sur le contenu et/ou collaboratives** grâce à **scikit-learn**.
- **Interface utilisateur** via **Streamlit** pour tester et visualiser les résultats.
- **Expérimentation rapide** de modèles et ajustement des paramètres.
- **CRUD des données** (ajouter, consulter, modifier et supprimer des informations si intégré à une base).

## Architecture & Technologies
- **Langage** : Python (100 %)
- **Bibliothèques principales** :
  - `pandas` → manipulation et analyse de données
  - `pyspark` → traitement distribué et scalable
  - `scikit-learn` → machine learning et calcul de similarités
  - `streamlit` → interface utilisateur simple et interactive
- **Structure** :
  - `src/` → code source principal
  - `requirements.txt` → dépendances Python
  - `.gitignore` → fichiers à ignorer pour le versionnement

## Installation
1. Clonez ou téléchargez le projet :
    ```bash
    git clone https://github.com/khayatti1/Systeme_recommandation.git
    ```
2. Accédez au répertoire du projet :
    ```bash
    cd Systeme_recommandation
    ```
3. Créez un environnement virtuel (recommandé) et activez-le :
    ```bash
    python -m venv venv
    source venv/bin/activate   # macOS / Linux
    venv\Scripts\activate      # Windows
    ```
4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
5. Lancez l'application Streamlit :
    ```bash
    streamlit run app.py
    ```

## Utilisation
- Accédez à l’interface locale générée par Streamlit (par défaut : [http://localhost:8501](http://localhost:8501)).
- Chargez vos données ou utilisez les exemples fournis.
- Choisissez le type de recommandation (contenu, collaboratif…).
- Visualisez les résultats directement dans l’interface.

