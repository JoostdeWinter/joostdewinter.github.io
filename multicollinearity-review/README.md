# Multicollinearity in Statistical Modeling: A Review

Code and results for:

De Winter, J. C. F. (2026). *Multicollinearity in statistical modeling: A review*. https://joostdewinter.github.io/multicollinearity-review.pdf

## Contents

`produce_figures.m`: MATLAB script that reproduces Figures 1–4 and the simulation results in Textboxes 1–4.

`results/`: output of the script, as reported in the paper.

| File | Content |
|---|---|
| `Figure1_results.csv` | Population and estimated coefficients, R², and VIF for r12 = 0 and 0.9 (Figure 1, Textbox 1) |
| `Figure2_results.csv` | Mean, SD, and correlation of the two coefficient estimates per predictor correlation (Figure 2, Textbox 2) |
| `Figure3_MonteCarlo_results.csv` | Test RMSE and coefficient MSE of OLS, ridge regression, and Lasso over 1,000 replications (Figure 3, Textbox 3) |
| `Figure3_Lasso_nonzero_counts.csv` | Number of nonzero Lasso coefficients per replication |
| `Figure3_counterchecks.csv` | Counterchecks for Textbox 3 |
| `Figure4_equivalent_estimands.csv` | The same quantities estimated from the uncentered and centered fit (Figure 4, Textbox 4) |
| `replication_results.mat` | All Monte Carlo results in MATLAB format |

## Running the script

Requirements: MATLAB R2020a or later with the Statistics and Machine Learning Toolbox. The Parallel Computing Toolbox is optional; without it, the Monte Carlo analysis runs serially.

Run `produce_figures.m` in MATLAB. The figures and tables are saved in the folder `multicollinearity_outputs`. Random seeds are fixed, so the results match the files in `results/`.

## License

The code and results are released under the MIT License (see `LICENSE`). Please cite the paper when you use them.
