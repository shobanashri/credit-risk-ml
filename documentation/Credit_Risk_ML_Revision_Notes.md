# Credit Risk Assessment & Loan Default Prediction

> **Purpose:** Concise, revision-friendly notes covering business understanding, data understanding, EDA, feature selection, model evaluation, and Streamlit.
>
> **Project:** CreditGuard
> **Target:** `loan_status` — 0 = No default recorded, 1 = Default recorded.

---

# Phase 1 — Business Understanding

## 1. Business Problem

* Loan defaults can cause financial losses for lending institutions.
* Lenders need to assess potential default risk before making lending decisions.
* A machine learning system can support risk assessment but should not replace responsible human decision-making.

## 2. Business Objective

Develop a machine learning system that uses borrower and loan information to identify applicants who may be at higher risk of loan default.

### Intended Benefits

* Reduce potential default-related losses.
* Improve credit-risk assessment.
* Flag applications for further review.
* Support credit officers with data-driven insights.

## 3. Main Stakeholders

| Stakeholder             | Main Interest                                                    |
| ----------------------- | ---------------------------------------------------------------- |
| Credit / Risk Team      | Risk assessment and portfolio risk                               |
| Credit / Loan Officers  | Applicant assessment and decision support                        |
| Business Management     | Loss control and sustainable growth                              |
| ML / Data Team          | Data quality, modeling, evaluation, and deployment               |
| Compliance / Governance | Explainability, fairness, responsible data use, and auditability |

## 4. Important Business Questions

* What exactly counts as a default?
* When must the prediction be made?
* Which features are available at that time?
* What action follows a high-risk prediction?
* What are the costs of false positives and false negatives?
* How representative is the historical dataset?
* What business metrics, fairness requirements, and explainability standards matter?

## 5. ML Problem Formulation

* **Input:** Borrower and loan-related features.
* **Target:** `loan_status`.
* **Problem type:** Supervised binary classification.
* **Class 0:** No default recorded.
* **Class 1:** Default recorded.

## 6. Data Leakage — Key Rule

* Use only information available at the intended prediction time.
* Features recorded after a lending decision or after repayment behavior occurs may cause data leakage.
* Leakage can produce overly optimistic evaluation results and poor real-world generalization.
* Verify when `loan_grade` and `loan_int_rate` become available before deciding whether they are appropriate features.

## 7. ML Thinking Chain

**Business Problem → Business Objective → Business Decision → Target → ML Formulation → Evaluation Metric → Model Selection → Deployment**

Choose the model only after understanding the objective, target, and costs of prediction errors.

---

# Phase 2 — Data Understanding

## 1. Dataset Overview

* **Original rows:** 32,581
* **Original columns:** 12
* **Numerical columns by initial pandas dtype:** 8
* **Categorical/text columns:** 4

Some integer columns, such as `loan_status`, represent categories or outcomes rather than continuous measurements.

## 2. Column Dictionary

| Feature                      | Meaning / Consideration                               |
| ---------------------------- | ----------------------------------------------------- |
| `person_age`                 | Applicant age                                         |
| `person_income`              | Applicant income                                      |
| `person_home_ownership`      | Home ownership category                               |
| `person_emp_length`          | Employment length                                     |
| `loan_intent`                | Loan purpose                                          |
| `loan_grade`                 | Loan grade; verify availability at prediction time    |
| `loan_amnt`                  | Loan amount                                           |
| `loan_int_rate`              | Interest rate; verify availability at prediction time |
| `loan_status`                | Target variable                                       |
| `loan_percent_income`        | Loan amount relative to income                        |
| `cb_person_default_on_file`  | Previously recorded default history                   |
| `cb_person_cred_hist_length` | Credit history length                                 |

## 3. Missing Values

| Feature             | Missing Count | Missing Percentage |
| ------------------- | ------------: | -----------------: |
| `person_emp_length` |           895 |              2.75% |
| `loan_int_rate`     |         3,116 |              9.56% |

* `loan_int_rate` has the most missing values.
* No missing values were found in the other columns during the initial inspection.
* **Initial decision:** Investigate the missingness before choosing a treatment strategy.

## 4. Duplicate Records

* Exact duplicate rows detected: **165**.
* Investigate duplicates before removing them.
* **Initial decision:** No rows had been removed at this stage.

## 5. Target Distribution

| `loan_status` | Meaning             |  Count | Percentage |
| ------------- | ------------------- | -----: | ---------: |
| 0             | No default recorded | 25,473 |     78.18% |
| 1             | Default recorded    |  7,108 |     21.82% |

### Key Interpretation

* The target is imbalanced because class 0 is more common than class 1.
* Accuracy alone may be misleading.
* Evaluate precision, recall, F1-score, ROC-AUC, and the confusion matrix according to the project's objectives.

## 6. Initial Data-Quality Checks

### Applicant Age

* Records with `person_age > 100`: **5**.
* Observed ages include 123 and 144.
* These values require investigation before treatment.

### Employment Length

* Records with `person_emp_length > 50`: **2**.
* Records where `person_emp_length > person_age`: **2**.
* Both flagged employment lengths were 123 years, while the recorded ages were 21 and 22.
* These two values are likely data-quality errors.

### Initial Status

At the initial inspection stage:

* No records had been removed or modified.
* No missing values had been imputed.
* No model had been trained.

**Note:** Later cleaning and modeling stages are documented separately below.

---

# Phase 3 — Exploratory Data Analysis (EDA)

## What Is EDA?

Exploratory Data Analysis examines data through descriptive statistics and visualizations to understand distributions, identify data-quality issues, and investigate relationships between features and the target.

For a binary target coded 0/1, the following calculation gives the observed default rate for each category:

```python
df.groupby("FEATURE")["loan_status"].mean() * 100
```

The group mean equals the proportion of records with `loan_status = 1`. Multiplying by 100 converts it to a percentage.

**Interpretation rule:** An observed association does not prove causation or guarantee an individual applicant's outcome.

## 1. Categorical Feature Counts

### `person_home_ownership`

| Category | Applicants |
| -------- | ---------: |
| RENT     |     16,446 |
| MORTGAGE |     13,444 |
| OWN      |      2,584 |
| OTHER    |        107 |

### `loan_intent`

| Category          | Applicants |
| ----------------- | ---------: |
| EDUCATION         |      6,453 |
| MEDICAL           |      6,071 |
| VENTURE           |      5,719 |
| PERSONAL          |      5,521 |
| DEBTCONSOLIDATION |      5,212 |
| HOMEIMPROVEMENT   |      3,605 |

### `loan_grade`

| Grade | Applicants |
| ----- | ---------: |
| A     |     10,777 |
| B     |     10,451 |
| C     |      6,458 |
| D     |      3,626 |
| E     |        964 |
| F     |        241 |
| G     |         64 |

### `cb_person_default_on_file`

| Category | Applicants |
| -------- | ---------: |
| N        |     26,836 |
| Y        |      5,745 |

Counts describe category frequency. Default rates help compare observed outcomes across categories.

---

## 2. Home Ownership vs Loan Default

### Question

Is default rate associated with home ownership status?

### Results

| Home Ownership | Default Rate |
| -------------- | -----------: |
| RENT           |       31.57% |
| OTHER          |       30.84% |
| MORTGAGE       |       12.57% |
| OWN            |        7.47% |

* Highest rate: **RENT — 31.57%**
* Lowest rate: **OWN — 7.47%**
* Difference: **24.10 percentage points**
* `OTHER` has only 107 applicants, so its estimate requires caution.

### Interpretation

Observed default rates vary across home ownership categories. RENT has a higher rate than OWN and MORTGAGE.

### ML Implication

`person_home_ownership` may provide predictive information and requires suitable categorical encoding.

### Business Implication

Investigate the pattern alongside other relevant information. Home ownership alone should not determine lending decisions, and fairness must be considered.

---

## 3. Loan Intent vs Loan Default

### Question

Is default rate associated with the purpose of the loan?

### Results

| Loan Intent       | Default Rate |
| ----------------- | -----------: |
| DEBTCONSOLIDATION |       28.59% |
| MEDICAL           |       26.70% |
| HOMEIMPROVEMENT   |       26.10% |
| PERSONAL          |       19.89% |
| EDUCATION         |       17.22% |
| VENTURE           |       14.81% |

* Highest rate: **DEBTCONSOLIDATION — 28.59%**
* Lowest rate: **VENTURE — 14.81%**
* Difference: **13.78 percentage points**

### Interpretation

Observed default rates vary by loan purpose. Debt consolidation has the highest rate and venture loans the lowest in this dataset.

### ML Implication

`loan_intent` may be useful for prediction and will require appropriate encoding.

### Business Implication

Loan purpose can provide context but should not be used as the sole basis for lending decisions.

---

## 4. Loan Grade vs Loan Default

### Question

How does default rate vary across loan grades?

### Results

| Loan Grade | Applicants | Default Rate |
| ---------- | ---------: | -----------: |
| A          |     10,777 |        9.96% |
| B          |     10,451 |       16.28% |
| C          |      6,458 |       20.73% |
| D          |      3,626 |       59.05% |
| E          |        964 |       64.42% |
| F          |        241 |       70.54% |
| G          |         64 |       98.44% |

* Highest rate: **G — 98.44%**, based on 64 applicants.
* Lowest rate: **A — 9.96%**.
* The observed rate generally increases from grade A to grade G.
* Largest adjacent increase: **C to D — approximately 38.32 percentage points**.

### Interpretation

Loan grade has a strong observed association with default status in this dataset. Grade G has a small sample, so its rate may be less stable.

### ML Implication — Leakage Check

Before using `loan_grade`, verify:

1. When is the grade assigned?
2. Is it available at the intended prediction time?
3. Does it depend on information unavailable when the prediction is supposed to be made?

A highly predictive feature is not automatically an appropriate feature.

### Business Implication

Confirm the meaning and timing of loan grade before using it for credit-risk assessment.

---

## 5. Previous Default History vs Loan Default

### Question

Is previous default history associated with the current target?

### Results

| Previous Default Recorded | Applicants | Default Rate |
| ------------------------- | ---------: | -----------: |
| N                         |     26,836 |       18.39% |
| Y                         |      5,745 |       37.81% |

* Difference: approximately **19.42 percentage points**.
* The observed default rate for Y is approximately **2.06 times** the rate for N.

### Interpretation

Applicants with a previously recorded default have a higher observed current-default rate.

### ML Implication

`cb_person_default_on_file` may be useful, provided it is legitimately available at prediction time.

### Business Implication

Previous default history may inform a broader assessment but should not be the sole basis for a lending decision.

---

## 6. `loan_percent_income` vs Loan Default

### Question

Is the loan amount relative to income associated with default?

### Observations

| Statistic          | No Default | Default |
| ------------------ | ---------: | ------: |
| Mean               |     14.88% |  24.69% |
| Median             |       0.13 |    0.24 |
| Standard Deviation |     0.0873 |  0.1321 |

* Median difference: **0.11**, or 11 percentage points.
* The default group has a higher median and greater variability.
* Boxplots show overlapping distributions and high-value outliers.

### Interpretation

Applicants in the default group generally have a higher loan amount relative to income, but the feature does not perfectly distinguish the two groups.

### ML Implication

`loan_percent_income` may be useful when combined with other features.

### Business Implication

Loan affordability can help inform repayment-risk assessment alongside other relevant information.

### Caution

Outliers are not automatically errors. Verify that the feature is calculated consistently during training and application inference.

---

## 7. `person_age` vs Loan Default

### Observations

* Mean age: approximately 27.81 years for non-defaults and 27.47 years for defaults.
* Median age: 26 years in both groups.
* Five records have ages above 100: two at 123 and three at 144.
* All five suspicious records belong to the non-default group.
* The initial age distribution is concentrated among younger borrowers.

### Interpretation

The two groups have similar age distributions based on their means and medians. The five unusually high ages require data-quality investigation.

### ML Implication

Handle implausible ages appropriately and evaluate the feature's predictive contribution during modeling.

### Business Implication

Interpret age-related patterns in the context of data quality and the population represented by the dataset.

---

## 8. `person_emp_length` vs Loan Default

### Observations

* Mean employment length: 4.97 years for non-defaults and 4.14 years for defaults.
* Median: 4 years for non-defaults and 3 years for defaults.
* Missing values in the original dataset: 895, approximately 2.75%.
* Two records contain employment lengths of 123 years.

### Interpretation

Non-default borrowers have a slightly higher mean and median employment length. The two 123-year values are implausible, while other high values should be investigated rather than automatically removed.

### ML Implication

Handle invalid values and missingness during preprocessing, fitting imputation rules on training data only.

### Business Implication

Employment length may provide context about borrower stability but should not be used as a standalone measure of creditworthiness.

---

## 9. Loan Amount vs Loan Status

### Observations

| Statistic          |   No Default |      Default |
| ------------------ | -----------: | -----------: |
| Mean Loan Amount   |        9,237 |       10,851 |
| Median Loan Amount |        8,000 |        9,600 |
| IQR                | 5,000–12,000 | 5,000–15,000 |

* The default group has a higher median and mean.
* Its interquartile range is wider.
* No missing values were found in `loan_amnt`.

### Interpretation

Defaulting borrowers had higher average and median loan amounts, but the distributions overlap.

### ML Implication

`loan_amnt` may provide useful information alongside income, interest rate, and loan-to-income ratio.

### Business Implication

Assess loan size alongside repayment capacity rather than using loan amount alone.

---

## 10. `loan_int_rate` vs Loan Status

### Observations

| Statistic            | No Default | Default |
| -------------------- | ---------: | ------: |
| Mean Interest Rate   |     10.44% |  13.06% |
| Median Interest Rate |     10.59% |  13.49% |
| IQR                  |       5.01 |    4.84 |

* The default group's interest-rate distribution is shifted higher.
* Original missing values: 3,116, approximately 9.56%.

### Interpretation

Defaulting borrowers generally had higher interest rates. This does not establish that higher rates directly cause default; interest rates may also reflect prior assessments of borrower risk.

### ML Implication

`loan_int_rate` may be useful, but missing-value handling and feature availability must be investigated.

### Business Implication

Consider interest rates alongside the borrower's broader financial situation.

---

## 11. `cb_person_cred_hist_length` vs Loan Status

### Observations

* Mean: 5.84 years for non-defaults and 5.69 years for defaults.
* Median: 4 years in both groups.
* Both groups have an IQR of approximately 5 years.
* The distributions overlap substantially.
* No missing values were found.

### Interpretation

Credit history length appears similar across the two groups based on these descriptive statistics.

### ML Implication

The feature may contribute information alongside other features. Evaluate its predictive value during model development instead of removing it based only on descriptive statistics.

### Business Implication

Credit history length provides context but does not fully describe repayment behavior.

---

## 12. Missing Employment-Length Information

### Observation

An EDA comparison reported:

* Missing employment-length group default rate: **31.72%**
* Available employment-length group default rate: **21.59%**
* Difference: **10.13 percentage points**

### Interpretation

Missing employment-length information was associated with a higher observed default rate in that analysis.

### Decision

Retain records where appropriate and handle missing values during model preprocessing. A missingness indicator can be evaluated, but should not be assumed to improve performance.

**Note:** These rates came from an earlier EDA result. Recalculate them if the dataset or cleaning stage changes.

---

## 13. Numerical Feature Descriptive Analysis

### Objective

Examine central tendency, variability, and range to identify potential skewness, unusual values, and preprocessing needs.

### Reported Findings

| Feature                      |      Mean | Median | Key Observation                           |
| ---------------------------- | --------: | -----: | ----------------------------------------- |
| `person_age`                 |     27.73 |     26 | Initial data contained implausible ages   |
| `person_income`              | 66,091.64 | 55,000 | Strong right skew and extreme high values |
| `person_emp_length`          |      4.78 |      2 | Right-skewed; missing values present      |
| `loan_amnt`                  |  9,593.85 |  8,000 | Mean exceeds median                       |
| `loan_int_rate`              |     11.02 |  10.99 | Missing values present                    |
| `loan_percent_income`        |      0.17 |   0.15 | Some high loan-to-income ratios           |
| `cb_person_cred_hist_length` |      5.81 |      4 | Mean exceeds median                       |

*These statistics were recorded at a particular analysis stage. Values may differ after cleaning or if calculated on a different dataset.*

### ML Implications

1. Investigate distributions and potential extreme values.
2. Preserve legitimate extreme values unless there is evidence of errors.
3. Handle missing values using preprocessing fitted on training data.
4. Evaluate transformations and scaling based on the chosen algorithm.

---

## 14. Income Outlier Analysis — IQR Method

### Objective

Identify potential outliers in `person_income` using the Interquartile Range (IQR) method.

### Results

| Statistic          |         Value |
| ------------------ | ------------: |
| Q1                 |        38,542 |
| Q3                 |        79,218 |
| IQR                |        40,676 |
| Lower Bound        |       -22,472 |
| Upper Bound        |       140,232 |
| Potential Outliers | 1,478 (4.56%) |

* No observations fall below the lower bound.
* 1,478 observations exceed the upper bound.
* The maximum recorded income was 6,000,000.

### Interpretation

The IQR method identifies statistical outliers, not necessarily invalid data. The result is consistent with the strong right skew observed in income.

### Decision

Retain flagged records unless further investigation identifies errors. Evaluate appropriate preprocessing during model development.

---

## 15. Income Distribution Visualization

### Observations

* Most income values are concentrated below approximately 200,000.
* The distribution has a long right tail extending to 6,000,000.
* Extreme values compress the central distribution in the raw histogram.

### Interpretation

The histogram supports the observed positive skewness.

### ML Implication

A transformation may help some algorithms, but its usefulness should be evaluated. Tree-based models generally require less distributional transformation than some other algorithms.

### Decision

Retain original values during EDA and investigate unusual observations before deciding on treatment.

---

## 16. Log-Transformed Income Distribution

### Objective

Use a logarithmic transformation to visualize the income distribution more clearly.

### Observations

* The transformed distribution appears more symmetrical.
* Many observations are concentrated around log10(income) values of 4.7–4.8.
* High values are compressed, making the central distribution easier to inspect.
* The raw income distribution had reported skewness of approximately 32.96.

### ML Implication

A log transformation may be useful for some algorithms, but it must be implemented consistently and evaluated during model development.

### Decision

Do not permanently transform or remove income values based only on a histogram. Compare preprocessing choices using appropriate validation.

---

## 17. Income Outliers vs Loan Default

### Observations

* Potential income outliers: 1,478 records (4.56%).
* Mean income in the flagged group: approximately 220,386.
* Median income: 175,000.
* Maximum income: 6,000,000.
* Non-default records: 1,334 (90.26%).
* Default records: 144 (9.74%).

### Default Rate Comparison

| Group                     | Borrowers | Defaults | Default Rate |
| ------------------------- | --------: | -------: | -----------: |
| Non-outliers              |    30,938 |    6,945 |       22.45% |
| Potential income outliers |     1,478 |      144 |        9.74% |

Difference in observed default rates: **12.71 percentage points**.

### Interpretation

The potential income-outlier group has a lower observed default rate than the non-outlier group. This comparison does not account for other borrower characteristics and does not establish causation.

### Decision

Retain the observations for now. Do not remove them solely because they exceed the IQR boundary.

---

## 18. Final Dataset Structure Validation — Cleaning Stage

### Reported Structure

* Shape: **32,416 rows × 15 columns**
* 12 original dataset columns.
* 3 missingness indicators created during EDA.
* Numerical columns use `int64` and `float64`.
* Categorical columns use `object`.
* Missingness indicators use `bool`.
* `loan_status` uses `int64`.

### Reported Remaining Missing Values at This Stage

| Feature             | Missing Count | Percentage |
| ------------------- | ------------: | ---------: |
| `loan_int_rate`     |         3,095 |      9.55% |
| `person_emp_length` |           889 |      2.74% |
| `person_age`        |             5 |      0.02% |

### Decision

Retain remaining missing values for preprocessing, fitting imputation rules on training data only.

**Important:** This is a recorded intermediate validation result. It should not be confused with the later cleaned dataset, which was reported as 32,416 × 12 with no missing values. Verify the exact DataFrame and notebook stage when reproducing these figures.

---

# Phase 4 — Feature Selection

## 1. Review Existing Features

The dataset contains **11 input features** and one target variable, `loan_status`.

| Feature Type             | Features                                                                                                                              |  Count |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | -----: |
| Numerical                | `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length` |      7 |
| Categorical              | `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`                                                     |      4 |
| **Total Input Features** |                                                                                                                                       | **11** |

### Key Observations

* Seven numerical features represent quantities such as age, income, and loan amount.
* Four categorical features represent categories such as home ownership and loan intent.
* `loan_grade` has seven unique categories.
* `loan_percent_income` is numerical but derived.
* `cb_person_default_on_file` has two categories.

### Feature Selection Reminder

Feature availability, data leakage, predictive usefulness, and consistency with the application must all be considered before finalizing the model inputs.

---

# Evaluation Metrics

## 1. Reported Baseline Results

| Metric    |  Score | Interpretation                                                        |
| --------- | -----: | --------------------------------------------------------------------- |
| Accuracy  | 86.94% | Percentage of all predictions that were correct                       |
| Precision | 77.37% | Of predicted defaulters, percentage that were actual defaults         |
| Recall    | 56.91% | Percentage of actual defaults identified                              |
| F1-score  | 65.58% | Harmonic mean of precision and recall                                 |
| ROC-AUC   | 86.79% | Ability to rank positive cases above negative cases across thresholds |

### Confusion Matrix

```text
[[4830  236]
 [ 611  807]]
```

| Actual Class   | Predicted: No Default (0) | Predicted: Default (1) |
| -------------- | ------------------------: | ---------------------: |
| No Default (0) |     4,830 — True Negative |   236 — False Positive |
| Default (1)    |      611 — False Negative |    807 — True Positive |

### Key Interpretation

* **True Negative:** Correctly predicted no default.
* **False Positive:** Predicted default when the actual class was no default.
* **False Negative:** Missed an actual default.
* **True Positive:** Correctly predicted default.

The baseline recall of 56.91% means that the model identified approximately 56.91% of the actual default cases at the evaluated threshold.

## 2. Decision Threshold Comparison

The following are the recorded qualitative observations from threshold exploration:

| Threshold | Reported Trade-off                                           |
| --------- | ------------------------------------------------------------ |
| 0.50      | Higher precision, but misses more defaulters                 |
| 0.40      | Highest F1-score among the tested thresholds                 |
| 0.30      | Identifies more defaulters but produces more false positives |
| 0.20      | Highest recall, but precision drops considerably             |

For the baseline, 0.40 was preferred when prioritizing F1-score. A lower threshold may be considered if reducing missed defaults is more important, depending on the costs of different errors.

**Evaluation caution:** Threshold selection should use validation data. The test set should remain untouched until final evaluation.

### Later Reported Model Results

A later project record reported:

* Accuracy: **91.58%**
* ROC-AUC: **92.09%**
* Default-class precision: **89%**
* Default-class recall: **70%**

These results belong to a later evaluation record and should not be combined with the baseline metrics above. Because the test set was used during threshold exploration, neither set should be presented as an independent final evaluation without re-evaluation on an untouched test set.

---

# Streamlit

## 1. Install Dependencies

```bash
python -m pip install streamlit joblib pandas numpy scikit-learn
```

## 2. Application Purpose

The Streamlit application allows users to enter borrower and loan information and view the saved model's estimated probability of default.

### Main Components

* `app.py` — application interface and prediction logic.
* `models/credit_risk_random_forest.joblib` — saved model.
* `models/decision_threshold.txt` — decision threshold.
* `styles/style.css` — custom styling.

### Important Checks

* Ensure application features match the trained model's expected inputs.
* Validate user inputs before prediction.
* Calculate `loan_percent_income` consistently with training.
* Confirm that the model and threshold files load correctly.
* Interpret the probability as a model estimate, not necessarily a calibrated real-world probability.

### Risk Categories

The application displays illustrative categories:

| Estimated Probability | Category |
| --------------------- | -------- |
| Below 10%             | Low      |
| 10% to below 25%      | Moderate |
| 25% to below 40%      | Elevated |
| 40% or above          | High     |

These bands are not validated financial risk grades and should not determine lending decisions automatically.

---

# Current Project Status

## Completed

* Business problem and objective.
* Initial dataset inspection and target distribution.
* Initial missing-value and duplicate checks.
* Categorical EDA.
* Numerical EDA and income-outlier investigation.
* Data cleaning and validation.
* Feature preparation and Random Forest modeling.
* Model evaluation and threshold exploration.
* Streamlit application development.

## Important Limitations to Address

* Earlier imputation before splitting may introduce data leakage.
* The test set was used during threshold exploration.
* Probability calibration has not been established.
* Feature timing and training/application schema consistency need verification.
* Reported metrics come from different stages and should remain clearly separated.


---

# Key Concepts to Remember

| Concept                     | Meaning                                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| Target Variable             | The outcome the model predicts                                                 |
| Feature                     | An input variable used to predict the target                                   |
| Binary Classification       | Predicting one of two classes                                                  |
| Target Imbalance            | One target class occurs more frequently than another                           |
| Default Rate                | Percentage of records with `loan_status = 1` in a group                        |
| Percentage-Point Difference | Subtraction of two percentages; e.g., 31.57% − 7.47% = 24.10 percentage points |
| Association                 | A relationship observed in data; not proof of causation                        |
| Data Leakage                | Using information that should not be available during training or evaluation   |
| Outlier                     | A value that differs substantially from the rest; not automatically an error   |
| Imputation                  | Replacing missing values using a selected method                               |
| Precision                   | Of predicted positive cases, how many were actually positive                   |
| Recall                      | Of actual positive cases, how many were identified                             |
| F1-score                    | Harmonic mean of precision and recall                                          |
| ROC-AUC                     | Measures ranking discrimination across classification thresholds               |
| Decision Threshold          | Probability cutoff used to assign a predicted class                            |

---

# Final Learning Summary

Credit risk modeling requires more than training a classifier. It involves understanding the business problem, checking data quality, investigating feature relationships, preventing data leakage, choosing appropriate evaluation metrics, and communicating limitations.

**Key principle:** Build a model that is not only predictive, but also properly evaluated, reproducible, and used responsibly.
