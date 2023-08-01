# Simulating clinical trials for model-informed precision dosing: Using warfarin treatment as a use case

This GitHub repository serves as documentation and reproduction source for the results published in XX. It contains the raw data, the data derived during the analysis, the model specifications (SBML format) and executable scripts (Python scripts as well as Jupyter notebooks).

## Looking at the results

The results are documented by multiple notebooks. To open the notebooks, please follow the links below:

1. [MIPD trial results & dosing strategies of models](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/results.ipynb)

To inspect the scripts used to generate data, implement models and to estimate model parameters, please follow the links below:

### Datasets
Clinical dataset published by Warfarin Consortium (March 2008):
1. [INR measurements under maintenance warfarin treatment](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/clinical_warfarin_inr_steady_state.csv) [[Raw dataset](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/raw_data/clinical_steady_state_INR_data_original_data.xls)] [[Preprocessing script](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/prepare_clinical_data.ipynb)]

Simulated measurements:
1. [Clinical trial phase I](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_I.csv) [[Data-generating script](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/2_perform_trial_phase_1.py)]
2. [Clinical trial phase II](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_II.csv) [[Data-generating script](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/3_perform_trial_phase_2.py)]
3. [Clinical trial phase III](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/trial_phase_III.csv) [[Data-generating script](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/4_perform_trial_phase_3.py)]
4. [MIPD trial cohort](https://github.com/DavAug/mipd-warfarin/blob/main/results/data/mipd_trial_cohort.csv) [[Data-generating script](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/5_simulate_cohort_for_mipd_trial.py)]
5. [MIPD trial results: Regression model](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/mipd_trial_predicted_dosing_regimens_deep_regression.csv) [[Data-generating script](#mipd-trial-simulation)]
6. [MIPD trial results: Deep RL model](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/mipd_trial_predicted_dosing_regimens.csv) [[Data-generating script](#mipd-trial-simulation)]
7. [MIPD trial results: PKPD model](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/mipd_trial_predicted_dosing_regimens.csv) [[Data-generating script](#mipd-trial-simulation)]

### Model implementations

1. [Warfarin clinical trial model](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/model.py) [[SBML file (*in vivo* model)](https://github.com/DavAug/mipd-warfarin/blob/main/models/wajima_coagulation_model.xml)] [[SBML file (INR test model)](https://github.com/DavAug/mipd-warfarin/blob/main/models/wajima_inr_test_model.xml)] [[Parameters](https://github.com/DavAug/mipd-warfarin/blob/main/models/hartmann_coagulation_model_parameters.csv)]
2. [Regression model](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/model.py) [[Training script](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/3_calibrate_nn_model_to_trial_phase_3_data.py)] [[Model weights](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/model/deep_regression_best.pickle)]
3. [Deep RL model](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/model.py) [[Training script](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/1_calibrate_model.py)] [[Model weights](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/models/dqn_model.pickle)]
4. [PKPD model](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/model.py) [[SBML file](https://github.com/DavAug/mipd-warfarin/blob/main/models/hamberg_warfarin_inr_model_with_sensitivities.xml)] [[Inference script (CTI)](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/1_calibrate_model_to_trial_phase_1_data.py)] [[Inference script (CTII)](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/2_calibrate_model_to_trial_phase_2_data.py)] [[Inference script (CTIII)](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/3_calibrate_model_to_trial_phase_3_data.py)] [[Posterior distribution](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/posteriors/posterior_trial_phase_III.nc)]

### MIPD trial simulation

Scripts used in all MIPD trial simulations:
1. [Monitoring data simulation](https://github.com/DavAug/mipd-warfarin/blob/main/results/1_systems_pharmacology_model/7_simulate_tdm_data_for_mipd_trial.py)

Scripts specific to the different MIPD models:
1. [Regression model](https://github.com/DavAug/mipd-warfarin/blob/main/results/3_regression_model/4_predict_dosing_regimens_for_mipd_trial_cohort_nn_regression.py)
2. [Deep RL model](https://github.com/DavAug/mipd-warfarin/blob/main/results/4_reinforcement_learning/2_predict_dosing_regimen_for_mipd_trial_cohort.py)
3. [PKPD model](https://github.com/DavAug/mipd-warfarin/blob/main/results/2_semi_mechanistic_model/6_predict_dosing_regimens_for_mipd_cohort_bayesian_optimisation.py)

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

You can reproduce the results using the Makefile. First install ``nbconvert`` which helps to execute the notebooks from the terminal

```bash
pip install nbconvert
```

You can now reproduce all data and figures in the article using
```bash
make all
```

This may take a while (hours to days), because you are re-running all scripts
sequentially.

To reproduce only the plots from the existing data you can run

```bash
make plot_results
```

You can also run each script individually, but be aware that some scripts are
dependent on the data derived in other scripts.
