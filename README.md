# AILS-MICCAI-UWF4DR-Challenge

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

As a part of the grand challenges published by MICCAI 2024, we tackle the UWF4DR challenge in the context of the AI in 
Life Science course at JKU Linz. The challenge addresses several classification tasks related to ultra-wide field fundus images. 

* Task 1: Image quality assessment for ultra-widefield fundus (UWF) images
* Task 2: Identification of referable diabetic retinopathy (RDR) 
* Task 3: Identification of diabetic macular edema (DME)

see [CodaLab Competition - MICCAI UWF4DR 2024](https://codalab.lisn.upsaclay.fr/competitions/18605) for more information.

## Bootstrap me!

Either execute the Makefile first, or set up the conda environment on your own (plain python) and execute

```
./tools/install_requirements.py

```

After that, download required data and model checkpoints with:

```
./tools/download_data_and_chkpts.py

```

For executing python files, you need to make sure that the working directory is part of your PYTHONPATH, so that modules can be resolved properly. You can generate a command to do this using this tool (or use any other option which works across platforms nicely, if you find one, please tell me):

```
./tools/generate_python_path_command.py

```

Happy coding!

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for ails_miccai_uwf4dr_challenge
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
|
├── tools              <- tools for the data processing pipeline of the project
│
└── ails_miccai_uwf4dr_challenge                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes ails_miccai_uwf4dr_challenge a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
