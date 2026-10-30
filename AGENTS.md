# AGENTS.md

## Project purpose

This repository contains the final practical assignment for the Machine Learning
course (INF01017). The task is binary email classification: `Prediction = 1` is
spam and `Prediction = 0` is a normal email. The main implementation is
`src/spam-detection.ipynb`.

Read this file before changing the notebook. Preserve the work already completed,
avoid unrelated rewrites, and do not delete user-created files.

## Repository map

- `src/spam-detection.ipynb`: T1 data preparation, experiments, evaluation, and report
  export code.
- `src/spam-detection-finetuning.ipynb`: standalone T2 download, split, Optuna tuning,
  and finalist evaluation workflow.
- `src/data/emails.csv`: local dataset. It is ignored by Git.
- `src/results/`: generated experiment results, organized by phase and output type.
- `src/results/data_overview/text/dataset_preview.txt`: intentionally ignored by
  Git because it contains a full textual rendering of the dataset.
- `reports/T1_Report.pdf`: report delivered for the first part of the assignment.
- `specs/T1 - Enunciado.pdf`: official requirements for T1.
- `specs/T2 - Enunciado.pdf`: official requirements for T2. This document is the
  source of truth for the remaining work.

## Decisions and work completed so far

### Data and preprocessing

- The target column is `Prediction`.
- The identifier column `Email No.` is excluded from the features.
- The CSV already contains numeric word counts. Use `TfidfTransformer`, not
  `TfidfVectorizer`.
- TF-IDF configuration:

  ```python
  TfidfTransformer(
      norm="l2",
      use_idf=True,
      smooth_idf=True,
      sublinear_tf=True,
  )
  ```

- There is a single stratified 80/20 train/test split with `random_state=42`.
- `X_test` and `y_test` are the held-out test set. They must not be used to fit
  preprocessing, select hyperparameters, adjust thresholds, or choose features.
- T1 uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` only on the
  training portion.
- Inside each fold, TF-IDF is fitted on the internal training data and only
  transformed on the validation data. The final T1 preprocessing is fitted on all
  `X_train` and then applied to `X_train` and `X_test`.
- Class distributions are plotted for the complete dataset, training set, and test
  set. Stratification preserves approximately the same class proportions in all
  partitions and folds.

### Metrics and models

- Metric helpers have the signature `(y_true, y_pred)`.
- The metrics are accuracy, F1-score, precision, and recall. Spam (`1`) is the
  positive class, and metrics that support it use `zero_division=0`.
- The historical T1 notebook ranks by accuracy, but the submitted T1 report
  prioritizes spam precision to avoid false positives. T2 follows the report:
  precision is primary; accuracy, F1-score, and recall remain mandatory.
- Five algorithms were spot-checked using identical folds:
  - Logistic Regression
  - Linear SVM
  - Decision Tree
  - XGBoost
  - Neural Network (`MLPClassifier`)
- T1 intentionally used default/fixed configurations without hyperparameter search
  or threshold optimization.

### Current T1 ranking

The cross-validation results currently stored in the repository rank the models as
follows by mean accuracy:

1. Linear SVM: 0.9867
2. Neural Network: 0.9845
3. Logistic Regression: 0.9792
4. XGBoost: 0.9662
5. Decision Tree: 0.9081

The group explicitly selected Neural Network and Decision Tree in section 7 of
the submitted T1 report, and requested XGBoost as a third candidate. These are
the agreed T2 candidates. Their investigative selection, despite the stronger
linear baselines, is justified in new notebook Markdown using training/CV
evidence only. Do not revert to the top-three-by-accuracy default.

The test set has already been evaluated once for the T1 comparison. Its stored
results are descriptive T1 results only. Do not use those values to guide T2
hyperparameter choices. For T2, freeze the search design first, tune only with the
training data, and perform the definitive test comparison only after tuning is
complete.

### Output conventions

- Notebook prose, labels, model names, tables, and plot titles are in English.
- Generated outputs belong under `src/results/`, divided into meaningful
  subdirectories rather than placed directly at the root:
  - `data_overview/text/`
  - `exploratory_analysis/figures/`
  - `preprocessing/text/`
  - `cross_validation/text/`
  - `cross_validation/figures/`
  - `test_evaluation/text/`
- Add similarly organized T2 directories, for example:
  - `hyperparameter_optimization/text/`
  - `hyperparameter_optimization/figures/`
  - `final_comparison/text/`
  - `final_comparison/figures/`
  - `error_analysis/text/`
  - `error_analysis/figures/`
  - `model_interpretation/text/`
  - `model_interpretation/figures/`
- Save plots as PDF and textual/statistical experiment results as UTF-8 TXT.
- Do not track `dataset_preview.txt`. Other relevant reports may be committed.
- Clear stored notebook outputs before pushing if the notebook approaches GitHub's
  100 MB file limit. Do not remove code when clearing outputs.

## T2 implementation status and agreed execution

- T2 workflow is in `src/spam-detection-finetuning.ipynb`; the T1 notebook contains
  only its original T1 cells.
- The T2 notebook downloads the Kaggle CSV, creates a stratified 80/20 split
  (seed 42), prepares conventional stratified five-fold CV, and defines
  leakage-safe TF-IDF pipelines for Neural Network, Decision Tree, and XGBoost.
- Precision for spam is the primary tuning metric; accuracy, F1, and recall are
  recorded. The notebook contains Optuna search functions and a final held-out
  evaluation cell. The final cell is an explicit run step and may be expensive.
- The user requested preparation without execution. Do not run the notebook or its
  training/evaluation cells unless the user later authorizes execution.
- The final evaluation function checks that the completed search summary matches
  the configured protocol, study count, and trial budget.
- T2 result paths are organized under `src/results/hyperparameter_optimization/`
  and `src/results/final_comparison/`.
- Remaining: run authorized experiments, produce required comparison figures,
  aggregate error analysis, interpret one model, obtain the Model Card template,
  prepare the presentation, and complete group review. T2 is not complete.

## T2 requirements and remaining work

The following checklist comes from `specs/T2 - Enunciado.pdf`. The implementation
status above distinguishes completed preparation from running and pending work.

### 1. Select and justify the promising strategies (completed)

- Formally select two or three algorithms from T1.
- Agreed selection: Neural Network, Decision Tree, and XGBoost (documented above).
- Explain why these models were selected using CV mean, standard deviation, and all
  four metrics. Discuss computational cost and model diversity where relevant.

### 2. Define the T2 validation protocol before tuning (completed)

- Keep the current 20% stratified test holdout untouched during all optimization.
- Choose and document one valid training/validation strategy:
  - holdout test plus stratified cross-validation on the remaining 80%; or
  - nested stratified cross-validation if the extra computational cost is accepted.
- The current five-fold stratified CV on `X_train` is a valid starting point for the
  first option. If nested CV is adopted, justify the outer/inner fold counts and
  runtime trade-off.
- Fix all random seeds where the estimator or search method supports them.
- All candidate models must be evaluated using identical splits.

### 3. Prevent data leakage during hyperparameter optimization (implemented)

- Put TF-IDF and the estimator in the same scikit-learn `Pipeline` used by the
  search procedure.
- Pass the original count matrix for the training portion to the search. Do not
  search over the already transformed `X_train_tfidf` because TF-IDF must be refit
  independently inside every search split.
- Fit IDF, feature selection, balancing, imputation, or any future learned
  preprocessing step only on the corresponding internal training fold.
- Apply learned transformations to validation/test data only with `transform`.
- Never optimize hyperparameters or decision thresholds against `X_test`/`y_test`.

### 4. Optimize each selected algorithm

- Define a justified hyperparameter search space for each of the selected models.
- Use an automated search such as `GridSearchCV`, `RandomizedSearchCV`, Optuna, or
  another justified method. The specification mentions Grid Search, Hyperopt, and
  Optuna as possibilities; no particular library is mandatory.
- Optimize the declared primary metric (spam precision) while recording
  F1-score, precision, and recall for every relevant configuration.
- Reasonable parameters to investigate include:
  - Decision Tree: depth, split/leaf sizes, feature sampling, criterion, and pruning.
  - XGBoost: boosting rounds, depth, learning rate, sampling, and regularization.
  - Neural Network: hidden-layer structure, activation, regularization `alpha`,
    learning rate, and batch size. Early stopping remains disabled in this run.
- Keep the search computationally feasible and report search method, search space,
  number of evaluated configurations, runtime, best parameters, and best CV score.
- Do not silently change the primary metric or tune a classification threshold
  without documenting and justifying that methodological decision.

### 5. Compare optimized models on the held-out test set

- After every search is finalized, refit each selected pipeline on all of
  `X_train`/`y_train` using its best hyperparameters.
- Evaluate each optimized model once on `X_test`/`y_test`.
- Report accuracy, F1-score, precision, and recall consistently.
- Compare strengths, weaknesses, variability, and the gap between CV and test
  performance.
- Identify the best overall model or clearly explain a practical tie and the
  conditions under which each winner is preferable.

### 6. Add required comparison visualizations

- Create visualizations that materially support the final comparison.
- For this classification task, include at least:
  - confusion matrix for every optimized finalist;
  - metric comparison chart;
  - ROC curves and ROC-AUC when decision scores/probabilities are available;
  - precision-recall curves and average precision, which are especially useful for
    the positive spam class.
- Use decision scores for `LinearSVC` where probabilities are unavailable, or
  explicitly justify probability calibration if it is added.
- Save every final plot as PDF in the appropriate T2 results subdirectory.

### 7. Perform aggregated error analysis

- Analyze errors for the best model, or for all tied winners.
- Do not write a sample-by-sample narrative. Aggregate findings instead.
- At minimum, compare false positives and false negatives, identify which class is
  harder, quantify each error type, and examine recurring word-count/TF-IDF
  patterns associated with mistakes.
- Investigate whether apparently confusing emails contain attributes more typical
  of the opposite class.
- Discuss the practical impact of false positives (normal email marked as spam)
  and false negatives (spam missed by the model).

### 8. Interpret one final model

- Select one model based on the definitive test analysis.
- Apply an appropriate interpretation method and identify the features that appear
  most relevant and how they affect decisions.
- For a linear winner, coefficient-based global interpretation mapped back to the
  original word columns is a strong baseline; permutation importance or SHAP may
  complement it when methodologically appropriate.
- For a neural-network winner, use a justified model-agnostic method and account for
  the sparse, high-dimensional input and computational cost.
- Produce report-ready tables/plots and discuss both plausible and unexpected
  relationships. Do not present feature importance as proof of causality.

### 9. Produce the mandatory T2 deliverables

- Create the final PDF using the Model Card structure supplied by the professor in
  Moodle. The format is mandatory. Obtain the template before authoring the final
  report; it is not currently present in this repository.
- The Model Card must include group identification, model development information,
  results, interpretation, limitations, and a link to the notebook/repository.
- Prepare the oral presentation. It is worth 20% of the T2 grade.
- Submit one delivery per group through Moodle by **29 November 2026 at 23:59**.
- Late work loses one point per day according to the specification.

### 10. Ensure reproducibility and responsible use of AI

- Keep code, seeds, dependency installation, output paths, and methodology
  reproducible.
- Every methodological choice and conclusion must be understood and defensible by
  the group.
- Review and validate AI-assisted code and text before incorporating it. The group
  remains responsible for correctness, validity, and reproducibility.

## T2 definition of done

T2 is complete only when all of the following are true:

- Two or three T1 finalists are explicitly selected and justified.
- Hyperparameters are optimized using training/validation data only.
- TF-IDF is fitted inside every relevant training fold with no test leakage.
- Search spaces, best parameters, CV statistics, and runtime are recorded.
- Finalists are refitted on all training data and compared once on the held-out
  test set with all four metrics.
- Confusion matrices, metric comparison, ROC, and precision-recall analyses are
  available or any omission is explicitly justified.
- Aggregated error analysis is documented.
- One final model is interpreted with report-ready evidence.
- T2 outputs are organized as PDF/TXT under `src/results/`.
- The mandatory Model Card PDF and oral presentation are complete.
- The notebook is valid, reproducible, small enough for GitHub, and the repository
  contains no tracked dataset preview.

## Working rules for future agents

- Inspect the current notebook and Git status before editing.
- Treat the official PDF specifications as authoritative if this guide conflicts
  with them.
- Make focused edits and preserve unrelated user work.
- Do not create unnecessary files or delete existing work.
- Do not retrain expensive models or overwrite stored results unless the user asks
  for execution or the task explicitly requires regenerated evidence.
- When experiments are executed, save their statistical and graphical outputs in
  the organized results tree and ensure filenames describe the experiment.
- Before pushing, verify that `dataset_preview.txt` is ignored and that no file
  exceeds GitHub's size limit.
