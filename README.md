# Simulating clinical trials for model-informed precision dosing: Using warfarin treatment as a use case

This GitHub repository serves as documentation and reproduction source for the results published in XX. It contains the raw data, the data derived during the analysis, the model specifications (SBML format) and executable scripts (Python scripts as well as Jupyter notebooks).

## Looking at the results

The results are documented by multiple notebooks. To open the notebooks, please follow the links below:

1. [MIPD trial results & dosing strategies of models](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/results.ipynb)

To inspect the scripts used to generate data, implement models and to estimate model parameters, please follow the links below:

### Datasets
Clinical dataset published by Warfarin Consortium (March 2008):
1. [INR measurements under maintenance warfarin treatment](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/clinical_warfarin_inr_steady_state.csv) [[Raw dataset]](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/raw_data/clinical_steady_state_INR_data_original_data.xls) [[Preprocessing script]](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/prepare_clinical_data.ipynb)

Simulated measurements:
1. [Warfarin trial phase I](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_I.csv)
2. [Warfarin trial phase II](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_II.csv)
3. [Warfarin trial phase III](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_III.csv)
4. [MIPD trial cohort](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/mipd_trial_cohort.csv)
5. [MIPD trial results: Regression model](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/mipd_trial_predicted_dosing_regimens_deep_regression.csv)
6. [MIPD trial results: Deep RL model](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/mipd_trial_predicted_dosing_regimens.csv)
7. [MIPD trial results: PKPD model](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/mipd_trial_predicted_dosing_regimens.csv)

### Model implementations

1. [TODO](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/exponential_growth_model.py)

### Data-generating scripts

Clinical trial phase I:
1. [TODO](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/1_generate_data.py)

### MIPD trial
Early cancer growth:
1. [TODO](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/2_run_nlme_inference.py)

## Reproducing the results

To reproduce the results, the GitHub repository can be cloned, and the scripts
can be executed locally. For ease of execution, we prepared a `Makefile` that
runs the scripts in the correct order. Please find a step-by-step instruction
how to install the dependencies and how to reproduce the results, once the
repostory has been cloned.

#### 1. Install dependencies

- 1.1 Install CVODE (myokit uses CVODE to solve ODEs)

For Ubuntu:
```bash
apt-get update && apt-get install libsundials-dev
```
For MacOS:
 ```bash
brew update-reset && brew install sundials
```
For Windows:
    No action required. Myokit installs CVODE automatically.

- 1.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 2. Reproduce results

You can now reproduce the results by running

```bash
make all
```

This may take a while (hours to days), because you are re-running all scripts
sequentially. To reproduce only the plots from the existing data you can run

```bash
make plot_results
```

You can also run each script individually, but be aware that some scripts are
dependent on the data derived in other scripts.
