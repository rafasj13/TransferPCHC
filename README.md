# TransferPCHC
Transfer learning for nonparametric Bayesian Networks


## PREPARE target and source tasks
All the reference models are available. To get the modified sources, run the following files:
### Synthetic SPBNs and GBNs
- s_bnlearn.R for bnlearn networks (<model>.rds)
- s_synthetic.py for SPBNs (<model>.pickle)
### UCI repository
To get the datasets for these networks, first run "clean_uci.ipynb". Then, the reference HC and PC models, as well as the posterior modifications for the sources, are obtained through:
- s_uci_ml.py (<model>.pickle).
  
Since the .pickle files are already available, the file can be modified to avoid training each reference again. Thus, simply reading the reference and getting the arc modifications.

## RUN experiments 
In the experiments file, the target and modified source tasks are loaded and fitted (needed PREPARE step). Then, a sample is generated from each task, and Gaussian noise is added to the sources. The file to run is:
- transfer_experiments.py

## RESULTS
The results are stored at:
- exps/
- critical_diff/
  
The plots are obtained from:
- plots.ipynb

### STATISTICAL TEST: Critical difference diagram
The Bergmann-Hommel post-hoc is done in R. To get the analysis, the files to run are:
- 1. plots.ipynb --> There is a section in which the results from the plots are transformed into a specific format for the statistical test
- 2. critical_diff_bergmann.R --> This file performs the Friedman test with Bergmann-Hommel post-hoc and stores the results (p-vales and ranking). Installation of scmamp_0.2.1.tar.gz  library is needed
- 3. critical_diff_diag_prep.ipynb --> This file takes the results from R and draws the critical difference diagram
