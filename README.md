# CreditGuard --- Credit Risk Assessment & Loan Default Prediction

CreditGuard is a machine-learning portfolio project that estimates the
likelihood of a borrower experiencing loan default based on selected
borrower and loan characteristics. It combines exploratory data
analysis, a trained scikit-learn model, and a Streamlit interface for
exploring predictions.

> **Important:** CreditGuard is an educational project, not a production
> lending system. Its predictions should not be used as the sole basis
> for approving or rejecting real loan applications. The displayed risk
> bands are provisional and have not been validated for real-world
> lending.

## Project Objective

Explore credit-risk patterns in a loan dataset and build an application
that returns a model-estimated default probability for a user-provided
borrower profile.

**Target variable:** `loan_status`

-   `0` --- No default recorded in the dataset
-   `1` --- Default recorded in the dataset

The application displays an estimated probability and a provisional risk
category. It does not make an actual lending decision.

## Features Used by the Model

The model expects these input features:

1.  `person_age`
2.  `person_income`
3.  `person_home_ownership`
4.  `person_emp_length`
5.  `loan_intent`
6.  `loan_amnt`
7.  `loan_int_rate`
8.  `loan_percent_income`
9.  `cb_person_default_on_file`
10. `cb_person_cred_hist_length`

Feature names, categories, units, and preprocessing must remain
consistent with the training pipeline. In particular,
`loan_percent_income` should be calculated and interpreted using the
same definition used during training.

## Model and Reported Evaluation

The current project uses a saved scikit-learn Random Forest model loaded
with `joblib`.

Previously reported evaluation results:

  Metric                      Reported result
  ------------------------- -----------------
  Accuracy                             91.58%
  ROC-AUC                              92.09%
  Default-class precision                 89%
  Default-class recall                    70%

Reported confusion matrix:

``` text
                 Predicted
                 No default   Default
Actual No default    4945        121
Actual Default        425        993
```

The application currently uses a decision threshold of **0.40**, stored
in `models/decision_threshold.txt`.

**Evaluation caveat:** The test set was used during earlier threshold
exploration, so the reported results should not be treated as a fully
independent final evaluation. Reassess performance using a properly
separated validation/test strategy before drawing strong conclusions.

## Provisional Risk Bands

The application currently maps the estimated probability to these
provisional labels:

  Estimated default probability   Displayed category
  ------------------------------- --------------------
  Below 10%                       Low
  10% to below 25%                Moderate
  25% to below 40%                Elevated
  40% or above                    High

These bands are illustrative UI categories. Probability calibration and
the suitability of these cutoffs have not been established; the labels
are not validated credit ratings.

## Project Structure

Your local project may contain additional files. The main expected
structure is:

``` text
Credit-Risk-ML/
├── app.py
├── data/
│   └── raw/
│       └── credit_risk_dataset.csv
├── models/
│   ├── credit_risk_random_forest.joblib
│   └── decision_threshold.txt
├── notebooks/
│   └── 01_data_understanding.ipynb
├── styles/
│   └── style.css
└── README.md
```

## Tech Stack

-   **Python** --- application and analysis code
-   **Pandas** --- tabular data handling
-   **scikit-learn** --- machine-learning pipeline and model
-   **Joblib** --- loading the saved model
-   **Streamlit** --- interactive web interface
-   **Matplotlib / Seaborn** --- exploratory data visualizations

## Setup and Installation

### 1. Open the project

If the project is hosted on GitHub, clone it and enter the project
directory:

``` bash
git clone <your-repository-url>
cd Credit-Risk-ML
```

If it is already on your computer, open the project root folder in VS
Code and open a terminal there.

### 2. Create a virtual environment (recommended)

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install joblib pandas streamlit scikit-learn numpy matplotlib seaborn jupyter ipykernel
```

For reproducible setup, consider maintaining a `requirements.txt` file
with the package versions used by your working environment.

### 4. Check required files

Confirm these files exist:

``` text
app.py
models/credit_risk_random_forest.joblib
models/decision_threshold.txt
```

The raw dataset is required for notebook-based analysis. The app must be
able to load the saved model from the path configured in the code.

### 5. Run the Streamlit app

From the project root:

``` bash
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that address in
your browser.

## Exploratory Data Analysis

The dataset snapshot previously examined in this project had:

-   **Rows:** 32,581
-   **Columns:** 12
-   **Target:** `loan_status`
-   **Duplicate records identified:** 165
-   **Missing `person_emp_length` values:** 895
-   **Missing `loan_int_rate` values:** 3,116
-   **No-default records (`loan_status = 0`):** 25,473 (approximately
    78.18%)
-   **Default records (`loan_status = 1`):** 7,108 (approximately
    21.82%)

These values describe the dataset snapshot examined during the project.
Recalculate them if the source file changes. Missing values, duplicates,
outliers, and target imbalance should be handled deliberately during
data preparation and model training.

## How to Use the App

1.  Start the app with `python -m streamlit run app.py`.
2.  Enter the borrower and loan details requested by the form.
3.  Submit the form to obtain the model-estimated default probability.
4.  Review the displayed provisional risk category.
5.  Interpret the result as a model output, not a guarantee of future
    borrower behaviour or a lending recommendation.

Use realistic inputs that are consistent with the training data. Unusual
values or incorrectly calculated features can produce misleading
predictions.

## Responsible Use and Limitations

-   **Educational use:** This project is for learning, demonstration,
    and portfolio purposes.
-   **No automated lending decision:** Do not use its output as the sole
    basis for loan approval, rejection, pricing, or other consequential
    decisions.
-   **Unvalidated risk bands:** The displayed categories are provisional
    and are not validated credit grades.
-   **Calibration not established:** A predicted probability is not
    necessarily the borrower's true real-world default probability.
-   **Potential bias:** The dataset may contain historical biases or
    fail to represent other populations and lending contexts.
-   **Evaluation limitations:** Earlier threshold selection used the
    test set, so final performance should be reassessed with a clean
    evaluation design.
-   **Privacy:** Do not enter real people's sensitive financial or
    personal information into a public demo.
-   **Model compatibility:** The saved model and preprocessing must be
    compatible with the Python and scikit-learn versions used to run the
    app.

Before any real-world use, the project would require independent
evaluation, probability calibration, bias and fairness analysis,
data-protection safeguards, monitoring, and qualified legal and domain
review.

## Future Improvements

-   Add a `requirements.txt` with pinned package versions.
-   Document the full preprocessing and model-training pipeline.
-   Establish separate training, validation, and test datasets.
-   Tune the decision threshold using validation data only.
-   Evaluate probability calibration and class-specific
    precision/recall.
-   Add automated input validation and clearer handling of out-of-range
    values.
-   Add unit tests for feature preparation and prediction behaviour.
-   Document model metadata and training steps for reproducibility.
-   Add a model card describing intended use, limitations, and
    evaluation details.

## Author

**Shobana Shri**\
B.Tech --- Artificial Intelligence and Data Science

------------------------------------------------------------------------

*CreditGuard is an educational machine-learning portfolio project.*
