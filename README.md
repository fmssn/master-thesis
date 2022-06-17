# Files
### `repetitions_subsettreatments.joblib`
Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Misra-Matching for all three ML-Methods + combinations with all shrinkage methods. Only treatments 1, 2, 4 and 5 were considered.

### `repetitions_alltreatments.joblib`
Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Misra-Matching for all three ML-Methods + combinations with all shrinkage methods. All treatments were considered.

### `overfit.joblib`
Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Misra-Matching for all three ML-Methods + combinations with all shrinkage methods. All treatments were considered. Matching is only done on the test set to get the counterpart of the training error.

### `plots.py`
Code for creating plots used in the Analytics.ipynb

### `mlmethods.py`
Main script with all ML-Method classes and the code for Misra-Matching. Is only used for importing, empty `main()`

### `expdata.csv`
Raw data of the experiment.

### `cv_script.py`
Script for hyper-parameter tuning of the ML-Methods.

### `Analytics.ipynb`
Jupyter notebook for creating descriptional statistics, result tables and figures.

