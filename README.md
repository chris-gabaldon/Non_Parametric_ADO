# ADO_scripts

This repository hosts all the Python scripts related to the work titled *"Fast nonparametric Bayesian framework for on-the-fly adaptive design optimization using discrete priors."*

Inside, you will find the following main folders:

- **Algorithm**: Contains the primary scripts used to build the model and all the necessary functions. This folder also provides a general example demonstrating how to conduct an MSE analysis for numerical simulations, as well as a step-by-step tutorial for applying the algorithm to your own experiment.

- **Our Online Experiment**: The data_analysis.py script contains all the statistical analyses and allows for the reconstruction of the figures presented in the article. Additionally, the database is accessible. Within the Python script, there is a specific cell designed to filter the appropriate population used in the analysis, as the experiment was conducted online, resulting in numerous incomplete sessions that need to be excluded.  Also in this folder is available the supplentary information file with additional details about the online experiment.
  In the subfolder **online_experiment**, you will also find all routines related to the online experiment, including both the frontend and backend components.

- **Pseudo Experiments**: This folder contains the data and analysis scripts corresponding to the pseudo-experiments discussed in the paper. It includes two subfolders, each associated with one of the pseudo-experiments used.

- **Figures**: This folder contains specific scripts for reproducing each simulation shown in the article. Each script is named FigN.py, where N corresponds to the figure number in the manuscript. Running any of these scripts will automatically generate the corresponding plot.
