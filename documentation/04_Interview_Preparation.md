# CreditGuard — Interview Preparation

## 1. What is CreditGuard?

CreditGuard is a machine learning portfolio project that estimates loan default probability using borrower and loan characteristics. It includes an interactive Streamlit application.

## 2. What is the objective?

To explore credit risk patterns, train a classification model, and demonstrate its predictions through a web application.

## 3. What type of ML problem is this?

Supervised learning — binary classification.

The target, `loan_status`, has two classes:

* `0` — No default recorded
* `1` — Default recorded

## 4. Which algorithm did you use?

Random Forest Classifier, an ensemble algorithm that combines multiple decision trees.

## 5. Why did you perform EDA?

To understand feature distributions, identify data-quality issues, and explore relationships between borrower characteristics and default status.

## 6. How did you handle missing values?

I used median imputation for numerical features after identifying and handling duplicates and implausible values.

**Limitation:** Imputation occurred before the train-test split in the existing workflow. A better approach is to fit preprocessing on training data only.

## 7. Which evaluation metrics did you use?

Accuracy, precision, recall, ROC-AUC, and the confusion matrix.

These metrics help evaluate overall performance, positive predictions, missed defaults, and ranking ability.

## 8. What is the difference between precision and recall?

* **Precision:** Of the cases predicted as default, how many were actually defaults?
* **Recall:** Of the actual default cases, how many did the model identify?

## 9. Why did you use a decision threshold of 0.40?

The application uses 0.40 to convert estimated default probabilities into class predictions.

However, the threshold was explored using the test set. It should be selected using validation data for a more reliable evaluation.

## 10. How did you deploy the model?

I saved the trained model using Joblib and integrated it into a Streamlit application that collects inputs and displays predictions.

## 11. What challenges did you identify?

The main challenges were missing values, duplicate records, implausible values, target imbalance, preprocessing leakage, and the need for independent model evaluation.

## 12. What would you improve?

I would use a leakage-free preprocessing pipeline, select the threshold using validation data, evaluate on an untouched test set, investigate probability calibration, and add automated tests.

## 13. Can this application make real lending decisions?

No. It is an educational portfolio project. Its risk categories and probability outputs have not been established as validated real-world credit-risk assessments.

## 14. What did you learn?

I gained practical experience in data understanding, EDA, preprocessing, Random Forest classification, model evaluation, threshold selection, model serialization, Streamlit development, and documenting ML limitations.

---

## Interview Reminder

Explain what you actually implemented, why you made each choice, what the results mean, and what you would improve.

**Be honest about limitations:** the reported metrics are not an independent final evaluation because the test set was used during threshold exploration.
