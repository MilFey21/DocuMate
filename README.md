DocuMate
==============================

The software to finally bring order to your file system.
How does it technically work:
1) All files in selected (root_folder in params.yaml) folder will be exposed to algorithm.
2) If there are files with extension <code>.dmate</code> then embeddings of those files will be added to search index
thus increasing precision of algorithm and folders containing those files will be open to being target destination of
new files. Beware of languages being used.
3) Embeddings of all the other files within mentioned above folders will be created.
4) All files, which are in supported formats, from folder 'Downloaded' will be taken and their embeddings will be
created.
5) Using ML-model (KNN / Nearest Centroids algorithms) target folder will be assigned for new files and moved there.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pipreqs`
    │
    ├── params.yaml        <- YAML file with the main hyperparameters that could be easily changed
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── my_utils.py 
    │   ├── data           <- Scripts to download or generate data
    │   │   └── create_db.py
    │
    │   ├── modules        <- Main modules of the app devided into files acc. to logical steps 
    │   │   └── extract_embedding.py
    │   │   └── ml_model.py
    │   │   └── app.py
    │
    └── .pre-commit-config.yaml  <- File with configured pre-commit hooks

Requirements
--------
- Python (~3.11)

Note that this code was debugged so far only for macOS. Debugging for Windows is coming soon :) 

How to run
--------
1. Set required variables and parameters manually in file <code>params.yaml</code>
2. Set up virtual env
3. Install libraries\
<code>python -m pip install -r ./requirements.txt</code>
4. Run app/code
<code>python src/modules/app.py --config=params.yaml</code>