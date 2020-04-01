# HVAC model building case study

## About
The files within this repository are used to conduct a case study related to occupancy detection of an office room. Its content, tasks and analysis are summarized within a [Jupyter notebook](/notebooks/00-summary.ipynb).
The [layout](https://drivendata.github.io/cookiecutter-data-science/#directory-structure) of the repository follows best practice from [cookiecutter datascience](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science).
The case study is based on data from [UCI's machine learning repository](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+) and is similar to the analysis conducted by [Candanedo and Feldheim](https://www.sciencedirect.com/science/article/abs/pii/S0378778815304357).


## Prerequisites
* (Anaconda) Python 3.6+
* Unix-like environment (Linux, macOS, WSL on Windows)

## Usage
1. Clone this repository and navigate into its root directory
1. Create a new conda environment via `conda env create -f conda_env.yml` and activate it `conda activate hvac`
1. Run jupyter lab in order to run/adapt notebooks `jupyter lab`
1. Optionally: Launch MLflow in order to track experiments `./project/start_mlflow_server.sh` (make sure to adapt script variables accordingly)

## Limitations
* Model training, validation and deployment tasks are currently only implemented as Jupyter noteboos, not as dedicated Python scripts
 
## Resources
All materials (documentation, papers, books, blog posts, etc.) used to conduct the study are listed in
the references section of the summary notebook (see above).
