# CreditGuard — Project Learning Notes

## Project: Credit Risk Assessment & Loan Default Prediction

**Project Type:** Supervised Machine Learning — Binary Classification
**Model:** Random Forest Classifier
**Application:** Streamlit
**Target Variable:** `loan_status`

---

# 1. Problem Understanding

## Problem Statement

Loan default creates financial risk for lenders. This project explores borrower and loan characteristics to predict whether a record belongs to the default class using a machine learning classification model.

## Objective

Build a machine learning model that estimates the probability of loan default and integrate it into an interactive Streamlit application.

## Target Variable

`loan_status`

| Value | Meaning             |
| ----- | ------------------- |
| 0     | No default recorded |
| 1     | Default recorded    |

## ML Formulation

* **Learning type:** Supervised learning
* **Problem type:** Binary classification
* **Input:** Borrower and loan characteristics
* **Output:** Estimated probability of the default class and a threshold-based class prediction

## Important Distinction

The model predicts patterns learned from historical data. It does not establish that a borrower will definitely default, and the prediction is not itself a lending decision.

---

# 2. Data Understanding

## Objective

Understand the dataset's structure, variables, data types, and quality before performing EDA or model development.

## Original Dataset

`credit_risk_dataset.csv`

**Original shape:** 32,581 rows × 12 columns

## Initial Data-Quality Findings

| Issue                       | Count |
| --------------------------- | ----: |
| Duplicate rows              |   165 |
| Missing `person_emp_length` |   895 |
| Missing `loan_int_rate`     | 3,116 |

## Target Distribution

| `loan_status`           |  Count | Approximate Percentage |
| ----------------------- | -----: | ---------------------: |
| 0 — No default recorded | 25,473 |                 78.18% |
| 1 — Default recorded    |  7,108 |                 21.82% |

## Why Data Understanding Matters

* It reveals the dataset's size and structure.
* It helps identify missing values and duplicates.
* It clarifies which column represents the prediction target.
* It helps identify possible data-quality problems.
* It provides context before making modeling decisions.

---

# 3. Exploratory Data Analysis (EDA)

## Objective

Explore the distribution of variables and investigate how borrower and loan characteristics are associated with the target variable.

## Important Findings

### Target Distribution

The target classes are imbalanced: approximately 78.18% of records belong to class 0 and 21.82% to class 1.

**Learning:** Accuracy alone may not adequately describe performance on an imbalanced classification problem.

### Loan Amount

Applicants with recorded defaults had a higher median loan amount than applicants without recorded defaults.

**Learning:** Loan amount may contain useful predictive information, but the relationship should be evaluated alongside other features.

### Interest Rate

Applicants with recorded defaults had a higher median interest rate.

**Learning:** Interest rate may be associated with default status in this dataset. This observation does not establish causation.

### Loan-to-Income Ratio

The median `loan_percent_income` was approximately:

* Default records: 0.24
* Non-default records: 0.13

**Learning:** The proportion of income represented by a loan may help characterize borrower risk.

### Home Ownership

The observed default rate was higher for RENT than OWN.

**Learning:** Compare category-specific rates as well as category counts. A category with many records can have a different risk rate from a category with fewer records.

### Loan Intent

DEBTCONSOLIDATION had the highest observed default rate among the loan-intent categories examined.

**Learning:** Categorical variables may contain useful patterns that are not apparent from numerical summaries alone.

### Previous Default History

Applicants with a recorded previous default had a higher observed default rate.

**Learning:** Historical information can be associated with the target, but its usefulness must be assessed without introducing leakage or inappropriate assumptions.

### Loan Grade

Grade G showed a high observed default rate, but its sample size was small.

**Learning:** Always consider sample size when interpreting category-specific rates. A high rate based on few observations may be less stable.

## EDA Interpretation Rule

An observed association is not proof of causation. EDA helps generate hypotheses and understand data; it does not independently prove that a feature causes loan default.

---

# 4. Data Cleaning and Preprocessing

## Objective

Improve data quality and prepare a consistent dataset for analysis and modeling.

## Cleaning Operations Performed

1. Created a working copy of the original DataFrame.
2. Removed exact duplicate records.
3. Replaced ages above 100 with missing values.
4. Replaced employment lengths above 50 years with missing values.
5. Imputed missing numerical values using medians.
6. Validated the cleaned dataset.
7. Saved the cleaned dataset.

## Cleaned Dataset

`credit_risk_cleaned.csv`

**Final shape:** 32,416 rows × 12 columns

## Final Validation Results

| Validation                  | Result |
| --------------------------- | -----: |
| Missing values              |      0 |
| Duplicate rows              |      0 |
| Ages above 100              |      0 |
| Employment lengths above 50 |      0 |

## Why Remove Duplicate Records?

Exact duplicates can cause repeated observations to receive additional weight during analysis and model training.

However, duplicate removal should be based on the dataset's meaning and the project's assumptions. Identical-looking rows are not always accidental duplicates in every real-world dataset.

## Why Handle Invalid Values?

Implausible values can distort descriptive statistics and model behavior.

In this project, ages above 100 and employment lengths above 50 years were treated as invalid according to the selected cleaning rules.

These thresholds are project assumptions, not universal rules for every dataset.

## Why Use Median Imputation?

The median is the middle value after sorting observations. It is generally less sensitive to extreme numerical values than the mean.

Median imputation replaces a missing numerical value with the median calculated from available observations.

## Important Limitation: Data Leakage

In the existing workflow, imputation was performed before the train-test split.

This can introduce data leakage because information from the eventual test set can influence preprocessing statistics.

### Better Practice

1. Split the data first.
2. Fit the imputer using training data only.
3. Apply the fitted imputer to validation and test data.
4. Keep all learned preprocessing steps within a reproducible pipeline where appropriate.

The same principle applies to other transformations that learn information from data.

---

# 5. Feature Engineering and Selection

## Objective

Prepare the variables required by the model and ensure the application generates them consistently.

## Features Used by the Current Application

1. `person_age`
2. `person_income`
3. `person_home_ownership`
4. `person_emp_length`
5. `loan_intent`
6. `loan_amnt`
7. `loan_int_rate`
8. `loan_percent_income`
9. `cb_person_default_on_file`
10. `cb_person_cred_hist_length`

## Derived Feature: `loan_percent_income`

The application calculates:

`loan_percent_income = loan_amnt / person_income`

This represents the loan amount relative to the borrower's income.

The application checks that income is greater than zero before calculating the ratio.

## Why Feature Consistency Matters

The features used during inference must match those expected by the fitted model.

Important aspects include:

* Feature names
* Feature order, where relevant to the model interface
* Data types
* Categorical values
* Encoding and preprocessing
* Derived-feature calculations

A mismatch between training and application inputs can cause errors or unreliable predictions.

## Feature Requiring Verification

The cleaned dataset includes `loan_grade`, but the current application's input list does not include it.

The reason for its exclusion must be verified against the original training code and fitted model. It should not be assumed without evidence.

## Important Validation

Confirm that the application's `loan_percent_income` formula matches the definition used during training. Even a seemingly simple derived feature must be calculated consistently.

---

# 6. Train-Test Split and Data Leakage

## Why Split Data?

A model must be evaluated on data that was not used to fit it. Otherwise, the evaluation may not represent how well the model generalizes to unseen records.

### Training Set

Used to fit the model and learn patterns from the data.

### Validation Set

Used for model selection, hyperparameter tuning, and decision-threshold selection.

### Test Set

Reserved for final evaluation after development choices have been made.

## Why Use Stratification?

In classification problems, stratification helps preserve approximately the same target-class proportions across the resulting data splits.

This can be especially useful when one class is less common than the other.

## Data Leakage

Data leakage occurs when information that should not be available during training or model selection influences the learning process or evaluation.

In this project, two important concerns are:

1. Median imputation was performed before splitting the data.
2. The test set was used during decision-threshold exploration.

These practices can make the reported evaluation less independent than intended.

## Recommended Workflow

1. Define the target and features.
2. Create training, validation, and test sets.
3. Fit preprocessing on training data only.
4. Train candidate models using training data.
5. Make development choices using validation data.
6. Select the decision threshold using validation data.
7. Evaluate once on the untouched test set.

---

# 7. Random Forest Classifier

## What Is Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make a prediction.

For classification, the trees contribute to the final class prediction. A fitted Random Forest can also provide class-probability estimates when supported by its configuration and implementation.

## Why It Is Relevant to This Project

The project uses a Random Forest Classifier to learn patterns associated with the default and non-default classes.

The model is stored as:

`models/credit_risk_random_forest.joblib`

## Important Concepts

### Ensemble Learning

Combines multiple models to produce a prediction.

### Decision Tree

A model that makes predictions through a sequence of feature-based splits.

### Class Prediction

The predicted class label, such as 0 or 1.

### Probability Estimate

A numerical estimate associated with a class, returned by `predict_proba` for a compatible classifier.

### Generalization

The ability of a model to perform on previously unseen data.

## Important Note

The model's exact hyperparameters, training configuration, and preprocessing setup should be documented from the original training code rather than inferred from the filename.

---

# 8. Model Evaluation

## Reported Evaluation Results

| Metric                  | Reported Result |
| ----------------------- | --------------: |
| Accuracy                |          91.58% |
| ROC-AUC                 |          92.09% |
| Default-class precision |             89% |
| Default-class recall    |             70% |

These are the project's reported results. They should not be presented as independent final test performance because the test set was used during threshold exploration.

## Confusion Matrix

| Actual / Predicted | No Default | Default |
| ------------------ | ---------: | ------: |
| No Default         |      4,945 |     121 |
| Default            |        425 |     993 |

## Confusion Matrix Terms

### True Negative (TN)

The model predicts no default, and the actual label is no default.

Reported count: 4,945

### False Positive (FP)

The model predicts default, but the actual label is no default.

Reported count: 121

### False Negative (FN)

The model predicts no default, but the actual label is default.

Reported count: 425

### True Positive (TP)

The model predicts default, and the actual label is default.

Reported count: 993

## Accuracy

Measures the proportion of all predictions that are correct.

Accuracy can be misleading when class distributions are imbalanced or when different types of errors have different consequences.

## Precision

For the default class, precision measures the proportion of predicted defaults that were actually defaults in the evaluation data.

Higher precision means fewer false positives relative to the number of positive predictions.

## Recall

For the default class, recall measures the proportion of actual defaults identified by the model.

Higher recall means fewer actual defaults are missed.

## ROC-AUC

ROC-AUC summarizes how well the model ranks positive examples above negative examples across classification thresholds.

It measures ranking discrimination, not probability calibration.

## Important Evaluation Lesson

No single metric provides a complete picture of model performance.

Evaluate accuracy, precision, recall, ROC-AUC, the confusion matrix, and the project's error costs together.

---

# 9. Decision Threshold

## What Is a Decision Threshold?

A decision threshold converts a probability estimate into a class prediction.

The current application uses a threshold of 0.40.

### Current Rule

If the estimated default probability is greater than or equal to 0.40, the predicted class is 1.

Otherwise, the predicted class is 0.

The threshold is stored in:

`models/decision_threshold.txt`

## Why Not Always Use 0.50?

The best threshold depends on the problem's objectives and the relative importance of different errors.

Changing the threshold can change the number of positive predictions, precision, recall, false positives, and false negatives.

## Threshold Trade-Off

A lower threshold generally identifies more cases as positive, which may increase recall while also increasing false positives.

A higher threshold generally identifies fewer cases as positive, which may reduce false positives while missing more actual positive cases.

The exact effects depend on the model's predictions and dataset.

## Important Limitation

The current threshold was explored using the test set. A stronger evaluation would choose the threshold using validation data and reserve the test set for final evaluation.

The value 0.40 should not be described as a universally correct credit-risk threshold.

---

# 10. Model Serialization with Joblib

## What Is Model Serialization?

Serialization stores an object's state so that it can be loaded and reused later.

This allows the application to use the trained model without retraining it whenever a user opens the interface.

## Files Used

* `models/credit_risk_random_forest.joblib`
* `models/decision_threshold.txt`

## Why Save the Threshold Separately?

The model estimates class probabilities, while the threshold determines how those probabilities are converted into class predictions.

Keeping the threshold separately allows the configured decision rule to be changed without necessarily retraining the model.

However, any threshold change should be justified and evaluated appropriately.

## Compatibility Consideration

Serialized machine learning objects can depend on the Python and library versions used during training.

Record package versions and verify compatibility when moving the model to another environment.

Only load serialized model files from trusted sources.

---

# 11. Streamlit Application

## Objective

Provide an interactive interface for entering borrower information and displaying the model's output.

## Main Files

* `app.py` — application logic and interface
* `styles/style.css` — custom styling
* `models/credit_risk_random_forest.joblib` — saved model
* `models/decision_threshold.txt` — configured threshold

## Main Application Workflow

1. Load the application.
2. Load custom CSS when available.
3. Load the saved model.
4. Read and validate the decision threshold.
5. Collect borrower and loan inputs.
6. Validate required inputs.
7. Calculate `loan_percent_income`.
8. Prepare the input in the expected feature format.
9. Obtain the model's class-probability estimates.
10. Identify the probability corresponding to class 1.
11. Compare the probability with the decision threshold.
12. Display the probability, provisional risk category, and informational guidance.

## Why Check `model.classes_`?

The order of the columns returned by `predict_proba` corresponds to the order of the classes in `model.classes_`.

The application identifies the column corresponding to class 1 rather than assuming that a particular column position always represents default.

This makes the class-probability selection more explicit.

## Why Validate the Threshold?

The application checks that the threshold is between 0 and 1.

This is appropriate because a probability-based classification threshold must fall within that range.

## Why Validate Income?

The application calculates:

`loan_percent_income = loan_amnt / person_income`

A zero or negative income value would make this calculation invalid or unsuitable for the intended interpretation. The application therefore checks that income is greater than zero.

## Risk Categories

The current application displays these provisional probability bands:

| Estimated Probability | Display Category |
| --------------------- | ---------------- |
| Below 10%             | Low              |
| 10% to below 25%      | Moderate         |
| 25% to below 40%      | Elevated         |
| 40% or above          | High             |

These bands are illustrative display categories. They have not been established as validated financial risk grades.

## Review Guidance

The application also provides informational review guidance based on the estimated probability and configured threshold.

This guidance is not an automated lending decision and should not be treated as a recommendation to approve or reject a loan.

---

# 12. Important Limitations

The current project has several limitations that should be clearly documented.

1. The existing preprocessing workflow may introduce data leakage because imputation occurred before the train-test split.
2. The test set was used during threshold exploration, so the reported results are not an independent final evaluation.
3. Probability calibration has not been established from the information documented here.
4. The displayed risk bands are illustrative and have not been validated.
5. The reason for excluding `loan_grade` from the current application input list needs to be verified against the training pipeline.
6. The application's derived `loan_percent_income` calculation must match the training definition.
7. The exact model hyperparameters and package versions should be documented from the original environment.
8. Dataset associations do not establish causation.
9. Performance on the historical dataset does not guarantee performance on future applicants.
10. The application is an educational portfolio project, not a production-ready credit-scoring system.

---

# 13. Future Improvements

Recommended improvements include:

* Create separate training, validation, and test sets.
* Move preprocessing into a reproducible scikit-learn pipeline.
* Fit imputation and encoding only on training data.
* Select the decision threshold using validation data.
* Evaluate the final model on an untouched test set.
* Investigate probability calibration.
* Document model hyperparameters and training configuration.
* Record exact package versions.
* Verify the complete training and inference feature schema.
* Add input-validation tests.
* Add model-loading and prediction tests.
* Document limitations, intended use, and risks in a model card.
* Test the application in a clean environment.
* Improve reproducibility and repository organization.

---

# 14. Interview Revision Questions

## Q1. What is the objective of CreditGuard?

To build an educational machine learning application that estimates the probability of loan default from selected borrower and loan characteristics.

## Q2. What type of machine learning problem is this?

Supervised binary classification because the target variable has two classes: 0 and 1.

## Q3. What is the target variable?

`loan_status`.

## Q4. Which algorithm does the current project use?

Random Forest Classifier.

## Q5. Why is EDA important?

EDA helps understand variable distributions, identify data-quality issues, and investigate associations between features and the target.

## Q6. Why was median imputation used?

Median imputation replaces missing numerical values with the median and is generally less sensitive to extreme values than mean imputation.

## Q7. What is data leakage?

Data leakage occurs when information that should not be available during model fitting or evaluation influences the process and can produce overly optimistic results.

## Q8. What is the difference between precision and recall?

Precision measures how many predicted positive cases are actually positive. Recall measures how many actual positive cases the model identifies.

## Q9. What is ROC-AUC?

It measures how well the model ranks positive cases above negative cases across classification thresholds.

## Q10. Why use a decision threshold?

The threshold converts a probability estimate into a class prediction.

## Q11. Why is 0.40 used?

It is the current project's configured threshold. It should not be described as universally optimal, particularly because threshold exploration used the test set.

## Q12. Why save the model?

Saving the trained model allows the application to load and reuse it without retraining every time.

## Q13. Why use Streamlit?

Streamlit provides a way to build an interactive Python web application for demonstrating the model.

## Q14. Is the application ready for real lending decisions?

No. It is an educational portfolio project with evaluation, calibration, and validation limitations.

---

# 15. Final Learning Summary

Through CreditGuard, I practiced an end-to-end machine learning workflow involving:

* Problem definition
* Dataset inspection
* Exploratory data analysis
* Data cleaning
* Missing-value treatment
* Feature preparation
* Binary classification
* Random Forest modeling
* Model evaluation
* Confusion matrix interpretation
* Decision-threshold exploration
* Model serialization
* Streamlit application development
* Input validation
* Documentation and reproducibility

The most important lessons are to understand the data before modeling, prevent data leakage, choose metrics appropriate to the problem, keep training and application features consistent, and distinguish model-estimated probabilities from validated real-world risk.

CreditGuard demonstrates my learning and implementation skills. Its current results should be interpreted in light of the documented limitations, and further validation is required before making stronger claims about model performance.

---

**End of Project Learning Notes**
