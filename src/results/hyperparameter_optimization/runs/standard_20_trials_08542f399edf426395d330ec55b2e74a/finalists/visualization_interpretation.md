# T2 final comparison visualizations

Run: `standard_20_trials_08542f399edf426395d330ec55b2e74a`. The 1,035 held-out emails contain 735 normal messages and 300 spam messages. The reconstructed split and all model predictions were checked against the saved test predictions before generating the figures. These are descriptive results for one holdout, not evidence of statistical significance.

## Confusion matrices

Rows are the true classes and columns are predicted classes. TN is correctly classified normal mail; FP is normal mail marked spam; FN is spam missed; TP is correctly detected spam.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Neural Network | 727 | 8 | 5 | 295 |
| Decision Tree | 685 | 50 | 46 | 254 |
| XGBoost | 714 | 21 | 11 | 289 |

On this test set, the Neural Network makes the fewest false positives and false negatives. The Decision Tree makes the most of each. False positives can hide legitimate messages; false negatives leave spam in the inbox.

## Metric comparison

The grouped chart compares accuracy, spam precision, spam recall, and F1 at the classifiers' default prediction thresholds. Accuracy measures all correct predictions; precision measures the proportion of predicted spam that is actually spam; recall measures the proportion of spam detected; F1 balances precision and recall.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Neural Network | 0.9874 | 0.9736 | 0.9833 | 0.9784 |
| Decision Tree | 0.9072 | 0.8355 | 0.8467 | 0.8411 |
| XGBoost | 0.9691 | 0.9323 | 0.9633 | 0.9475 |

The Neural Network leads all four observed metrics, followed by XGBoost, then the Decision Tree. The primary selection metric was training-CV precision, not this test result. These comparisons alone do not establish a significant difference or generalization to other datasets.

## ROC curves

ROC plots the true-positive rate against the false-positive rate over score thresholds; ROC-AUC summarizes ranking performance, with 0.5 as the random-ranking reference. Test ROC-AUC: Neural Network 0.9985, XGBoost 0.9954, Decision Tree 0.8893. Both Neural Network and XGBoost rank these emails very well; the Decision Tree ranks them less well. AUC does not describe the exact error counts at the default threshold, and high AUC does not guarantee calibrated probabilities.

## Precision-recall curves

These curves show precision and recall across score thresholds. Average precision (AP) summarizes the curve using recall increments. The dashed reference is the test spam prevalence (300/1,035 = 0.2899); it is not a performance guarantee. Test AP: Neural Network 0.9963, XGBoost 0.9880, Decision Tree 0.7519. The first two maintain stronger ranking quality on this holdout; the plot displays the precision/recall trade-off as the threshold changes. No threshold was selected using the test set.
