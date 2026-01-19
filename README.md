# ADO_scripts

This repository hosts all the Python scripts related to the work titled *"Fast nonparametric Bayesian framework for on-the-fly adaptive design optimization using discrete priors."*

Inside, you will find the following main folders:

## Algorithm

This folder contains the core implementation of the Adaptive Design Optimization (ADO) framework proposed in the article. It includes all scripts required to define priors, run Bayesian updates, simulate experimental data, and apply the full workflow to new experiments.

The main scripts are:

- **ADO_experiment.py**  
  This script implements the core ADO algorithm. Starting from a user-defined discrete prior, it:
  - Constructs the likelihood and response matrices M associated with each candidate design.
  - Performs Bayesian updates of the posterior distribution after each simulated or real measurement.
  - Computes the expected information gain for all available designs at each trial.
  - Selects the optimal design by maximizing the information gain criterion.

  This script contains the main logic of the adaptive procedure and is used both in simulations and in real experimental workflows.

- **priors.py**  
  This script defines a collection of parametrized prior distributions used throughout the paper. It includes commonly used functional forms such as exponential, logarithmic, and other monotonic or nonlinear priors. These priors can be easily customized or extended and are designed to be directly compatible with the ADO algorithm implemented in ADO_experiment.py.

- **ground_truth.py**  
  This script implements the simulation workflow used to evaluate the performance of the algorithm. It generates synthetic experimental data by assuming a known ground-truth model and simulating responses according to that model.  
  It allows the user to:
  - Run pseudo-experiments under controlled conditions.
  - Quantitatively assess convergence, accuracy, and robustness of the ADO procedure.
  - Reproduce the simulation-based analyses reported throughout the article.

- **example_of_use.py**  
  This is a fully self-contained, step-by-step example demonstrating how to use the complete ADO workflow in practice. It shows how to:
  - Define a prior distribution.
  - Initialize the ADO algorithm.
  - Run adaptive trials with Bayesian updates.
  - Extract and visualize the final results.

  This script is intended as the main entry point for new users who want to apply the algorithm to their own experiments with minimal setup.


  ## Our Online Experiment:
   The data_analysis.py script contains all the statistical analyses and allows for the reconstruction of the figures presented in the article. Additionally, the database is accessible. Within the Python script, there is a specific cell designed to filter the appropriate population used in the analysis, as the experiment was conducted online, resulting in numerous incomplete sessions that need to be excluded.  Also in this folder is available the supplentary information file with additional details about the online experiment.
  In the subfolder **online_experiment**, you will also find all routines related to the online experiment, including both the frontend and backend components.

## Pseudo Experiments**: 
This folder contains the data and analysis scripts corresponding to the pseudo-experiments discussed in the paper. It includes two subfolders, each associated with one of the pseudo-experiments used.

- **Figures**: This folder contains specific scripts for reproducing each simulation shown in the article. Each script is named FigN.py, where N corresponds to the figure number in the manuscript. Running any of these scripts will automatically generate the corresponding plot.
