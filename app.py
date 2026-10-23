from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditGuard | Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / "credit_risk_random_forest.joblib"
)

THRESHOLD_PATH = (
    PROJECT_PATH
    / "models"
    / "decision_threshold.txt"
)

CSS_PATH = (
    PROJECT_PATH
    / "styles"
    / "style.css"
)


# ============================================================
# 3. LOAD CUSTOM STYLESHEET
# ============================================================

def load_css():
    """Load the stylesheet from the project's styles folder."""

    if not CSS_PATH.exists():
        st.warning(
            f"Custom stylesheet not found: {CSS_PATH}. "
            "The application will use Streamlit's default styling."
        )
        return

    css = CSS_PATH.read_text(encoding="utf-8")

    st.markdown(
        f"<style>\n{css}\n</style>",
        unsafe_allow_html=True,
    )


load_css()


# ============================================================
# 4. MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


# ============================================================
# 5. LOAD THE SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the fitted pipeline and configured decision threshold."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    loaded_threshold = float(
        THRESHOLD_PATH.read_text(
            encoding="utf-8"
        ).strip()
    )

    if not 0 <= loaded_threshold <= 1:
        raise ValueError(
            "The decision threshold must be between 0 and 1."
        )

    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "The saved model does not support predict_proba()."
        )

    return loaded_model, loaded_threshold


try:
    model, threshold = load_model()

except Exception as error:
    st.error(
        "CreditGuard could not load the saved model. "
        "Check the model files and installed dependencies."
    )

    with st.expander("Technical error details"):
        st.exception(error)

    st.stop()


# ============================================================
# 6. RISK CATEGORIES
# ============================================================

def categorize_risk(probability):
    """
    Assign a provisional display category.

    These bands are not validated or calibrated financial risk grades.
    """

    if probability < 0.10:
        return {
            "label": "Low Estimated Risk",
            "style": "risk-low",
            "description": (
                "The model estimates a relatively low probability "
                "of the recorded default outcome."
            ),
        }

    if probability < 0.25:
        return {
            "label": "Moderate Estimated Risk",
            "style": "risk-moderate",
            "description": (
                "The model estimates some likelihood of the recorded "
                "default outcome. Review the estimate alongside "
                "other relevant evidence."
            ),
        }

    if probability < 0.40:
        return {
            "label": "Elevated Estimated Risk",
            "style": "risk-elevated",
            "description": (
                "The estimate falls within an elevated provisional "
                "band. Further review may be warranted under the "
                "applicable assessment policy."
            ),
        }

    return {
        "label": "High Estimated Risk",
        "style": "risk-high",
        "description": (
            "The estimate falls within the highest provisional "
            "display band. Further assessment is recommended "
            "before reaching any lending decision."
        ),
    }


def get_review_guidance(probability, decision_threshold):
    """Explain how the model estimate relates to the configured threshold."""

    if probability >= decision_threshold:
        return {
            "title": "Further review recommended",
            "description": (
                "The model estimate meets or exceeds the configured "
                "decision threshold. Review the application under "
                "the applicable credit assessment policy."
            ),
            "style": "action-review",
        }

    return {
        "title": "Continue standard assessment",
        "description": (
            "The model estimate is below the configured threshold. "
            "Continue the standard assessment process and consider "
            "all other relevant evidence."
        ),
        "style": "action-standard",
    }


# ============================================================
# 7. DISPLAY HELPERS
# ============================================================

def render_result_card(label, value, description, card_class):
    """Render a styled HTML result card."""

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-card-label">{label}</div>
            <div class="result-card-value">{value}</div>
            <div class="result-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(number, title, subtitle=None):
    """Render a consistent section heading."""

    subtitle_html = (
        f'<p class="section-subtitle">{subtitle}</p>'
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-number">{number}</span>
            <div class="section-heading-content">
                <h2>{title}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">CG</div>
            <div>
                <div class="brand-name">CreditGuard</div>
                <div class="brand-caption">CREDIT INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### About this application")

    st.write(
        "CreditGuard estimates the probability of a recorded "
        "loan-default outcome using applicant and loan information."
    )

    st.write(
        "It is a machine learning portfolio demonstration, "
        "not an automated lending decision system."
    )

    st.divider()

    st.markdown("#### Assessment workflow")

    st.markdown(
        """
        **01** &nbsp; Enter applicant information

        **02** &nbsp; Provide loan details

        **03** &nbsp; Generate a model estimate

        **04** &nbsp; Review the result and limitations
        """
    )

    st.divider()

    st.markdown("#### Model information")

    st.markdown(
        """
        <div class="sidebar-info">
            <span class="sidebar-info-label">Algorithm</span>
            <span class="sidebar-info-value">Random Forest</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-info">
            <span class="sidebar-info-label">Decision threshold</span>
            <span class="sidebar-info-value">{threshold:.0%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "Portfolio project · For educational and demonstration use"
    )


# ============================================================
# 9. MAIN PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            MACHINE LEARNING · CREDIT ANALYTICS
        </div>
        <h1>Credit Risk Assessment</h1>
        <p class="hero-description">
            Explore a model-generated estimate of loan-default
            risk using applicant, financial and credit-history
            information.
        </p>
        <div class="hero-tags">
            <span>Random Forest</span>
            <span>10 input features</span>
            <span>Probability-based assessment</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Important information and limitations"):

    st.warning(
        """
        CreditGuard is a machine learning portfolio project.
        Its displayed risk bands are provisional, probability
        calibration has not been established, and suitability
        for production lending decisions has not been demonstrated.

        Do not use this model as the sole basis for approving
        or rejecting a loan application.
        """
    )


# ============================================================
# 10. APPLICANT INPUT FORM
# ============================================================

render_section_heading(
    "01",
    "Applicant Information",
    "Enter the information using the definitions and units "
    "expected by the model's training dataset.",
)

with st.form("credit_assessment_form"):

    st.markdown("#### Personal and financial details")

    personal_col1, personal_col2 = st.columns(2)

    with personal_col1:

        person_age = st.number_input(
            "Applicant age (years)",
            min_value=18,
            max_value=100,
            value=22,
            step=1,
            help="Enter the applicant's age in completed years.",
        )

        person_income = st.number_input(
            "Annual income",
            min_value=1,
            value=360000,
            step=10000,
            help=(
                "Use the same currency units and income definition "
                "as the training dataset."
            ),
        )

        person_emp_length = st.number_input(
            "Employment length (years)",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5,
            help=(
                "Enter employment length using the definition "
                "represented in the training data."
            ),
        )

    with personal_col2:

        person_home_ownership = st.selectbox(
            "Current housing situation",
            options=[
                "RENT",
                "MORTGAGE",
                "OWN",
                "OTHER",
            ],
            format_func=lambda value: {
                "RENT": "Renting",
                "MORTGAGE": "Home with a mortgage",
                "OWN": "Owns a home",
                "OTHER": "Other / not classified",
            }[value],
        )

        cb_person_default_on_file = st.selectbox(
            "Previous default recorded?",
            options=["N", "Y"],
            format_func=lambda value: {
                "N": "No previous default recorded",
                "Y": "Previous default recorded",
            }[value],
            help=(
                "Use the value recorded in the source field. "
                "No recorded default does not prove that no "
                "repayment difficulties have ever occurred."
            ),
        )

    st.divider()

    st.markdown("#### Loan and credit details")

    loan_col1, loan_col2 = st.columns(2)

    with loan_col1:

        loan_intent = st.selectbox(
            "Purpose of the loan",
            options=[
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "DEBTCONSOLIDATION",
                "HOMEIMPROVEMENT",
            ],
            format_func=lambda value: {
                "PERSONAL": "Personal expenses",
                "EDUCATION": "Education",
                "MEDICAL": "Medical expenses",
                "VENTURE": "Business venture",
                "DEBTCONSOLIDATION": "Debt consolidation",
                "HOMEIMPROVEMENT": "Home improvement",
            }[value],
        )

        loan_amnt = st.number_input(
            "Requested loan amount",
            min_value=1,
            value=100000,
            step=10000,
            help=(
                "Use the same currency units as the training dataset."
            ),
        )

        loan_int_rate = st.number_input(
            "Loan interest rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=9.25,
            step=0.25,
            format="%.2f",
            help="Enter 9.25 for an interest rate of 9.25%.",
        )

    with loan_col2:

        cb_person_cred_hist_length = st.number_input(
            "Credit history length (years)",
            min_value=0,
            max_value=100,
            value=3,
            step=1,
            help=(
                "Enter the recorded credit-history duration "
                "using the training dataset's definition."
            ),
        )

        calculated_ratio = loan_amnt / person_income

        st.markdown(
            """
            <div class="ratio-info">
                <div class="ratio-label">Loan-to-income ratio</div>
                <div class="ratio-caption">
                    Requested loan amount ÷ annual income
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Calculated ratio",
            f"{calculated_ratio:.2f}",
        )

        st.caption(
            f"{calculated_ratio:.1%} of annual income"
        )

    st.caption(
        "Verify that this ratio definition matches the "
        "loan_percent_income feature used during model training."
    )

    submitted = st.form_submit_button(
        "Generate Risk Assessment",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# 11. PREDICTION
# ============================================================

if submitted:

    if person_income <= 0:
        st.error("Annual income must be greater than zero.")

    elif calculated_ratio > 10:
        st.error(
            "The requested loan is more than ten times the "
            "entered annual income. Verify the amount, income "
            "and units before proceeding. This is an input "
            "sanity check, not a lending rule."
        )

    else:

        input_data = pd.DataFrame(
            [
                {
                    "person_age": person_age,
                    "person_income": person_income,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "loan_intent": loan_intent,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": loan_int_rate,
                    "loan_percent_income": calculated_ratio,
                    "cb_person_default_on_file": (
                        cb_person_default_on_file
                    ),
                    "cb_person_cred_hist_length": (
                        cb_person_cred_hist_length
                    ),
                }
            ],
            columns=EXPECTED_FEATURES,
        )

        try:

            probabilities = model.predict_proba(input_data)[0]

            # Use the model's explicit class labels rather than
            # assuming that probability column 1 always means class 1.
            model_classes = list(model.classes_)

            if 1 not in model_classes:
                raise ValueError(
                    "The saved model does not contain target class 1."
                )

            default_probability = float(
                probabilities[model_classes.index(1)]
            )

            predicted_class = int(
                default_probability >= threshold
            )

            risk = categorize_risk(default_probability)

            guidance = get_review_guidance(
                default_probability,
                threshold,
            )

            # ------------------------------------------------
            # RESULT HEADER
            # ------------------------------------------------

            st.divider()

            render_section_heading(
                "02",
                "Loan Risk Assessment",
                "Model-generated estimate based on the information entered.",
            )

            # ------------------------------------------------
            # RESULT CARDS
            # ------------------------------------------------

            card_col1, card_col2, card_col3 = st.columns(
                [1, 1, 1],
                gap="medium",
            )

            with card_col1:
                render_result_card(
                    "ESTIMATED DEFAULT PROBABILITY",
                    f"{default_probability:.1%}",
                    "Model-estimated probability of the recorded "
                    "default outcome.",
                    "probability-card",
                )

            with card_col2:
                st.markdown(
                    f"""
                    <div class="result-card risk-card">
                        <div class="result-card-label">
                            MODEL RISK CATEGORY
                        </div>
                        <div class="risk-badge {risk['style']}">
                            {risk['label']}
                        </div>
                        <div class="result-card-description">
                            Provisional display category; not a
                            validated financial risk grade.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col3:
                st.markdown(
                    f"""
                    <div class="result-card action-card">
                        <div class="result-card-label">
                            SUGGESTED ASSESSMENT ACTION
                        </div>
                        <div class="action-title">
                            {guidance['title']}
                        </div>
                        <div class="result-card-description">
                            Informational guidance only. Apply the
                            relevant assessment policy.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # PROBABILITY VISUALIZATION
            # ------------------------------------------------

            st.markdown(
                '<div class="subsection-gap"></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Estimated probability")

            probability_col1, probability_col2 = st.columns(
                [4, 1]
            )

            with probability_col1:
                st.progress(
                    min(max(default_probability, 0.0), 1.0)
                )

            with probability_col2:
                st.markdown(
                    f"""
                    <div class="probability-value">
                        {default_probability:.1%}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "The estimate represents the model's predicted "
                "probability of loan_status = 1. It is not a "
                "guarantee of future borrower behaviour."
            )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.markdown("#### Interpretation")

            st.markdown(
                f"""
                <div class="interpretation-card">
                    <div class="interpretation-heading">
                        {risk['label']}
                    </div>
                    <p>{risk['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="guidance-card">
                    <div class="guidance-label">
                        SUGGESTED NEXT STEP
                    </div>
                    <div class="guidance-title">
                        {guidance['title']}
                    </div>
                    <p>{guidance['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.warning(
                "This result is not a loan approval or rejection. "
                "Review the application under the applicable lending "
                "policy and consider all other relevant evidence."
            )

            # ------------------------------------------------
            # RISK BANDS
            # ------------------------------------------------

            with st.expander(
                "Understand the provisional risk categories"
            ):

                st.write(
                    "The following bands are used only to organise "
                    "the demonstration interface. They have not "
                    "been validated as financial risk grades."
                )

                risk_band_table = pd.DataFrame(
                    {
                        "Estimated probability": [
                            "Below 10%",
                            "10% to below 25%",
                            "25% to below 40%",
                            "40% and above",
                        ],
                        "Display category": [
                            "Low Estimated Risk",
                            "Moderate Estimated Risk",
                            "Elevated Estimated Risk",
                            "High Estimated Risk",
                        ],
                    }
                )

                st.dataframe(
                    risk_band_table,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    f"""
                    **Configured binary decision threshold: {threshold:.0%}**

                    - At or above the threshold: predicted class 1.
                    - Below the threshold: predicted class 0.

                    The binary threshold and four display categories
                    serve different purposes. Neither should be
                    interpreted as an automatic lending decision.
                    """
                )

            # ------------------------------------------------
            # TECHNICAL DETAILS
            # ------------------------------------------------

            with st.expander("Technical Model Details"):

                st.markdown("#### Current prediction")

                technical_col1, technical_col2 = st.columns(2)

                with technical_col1:
                    st.metric(
                        "Predicted target class",
                        str(predicted_class),
                    )

                with technical_col2:
                    st.metric(
                        "Configured threshold",
                        f"{threshold:.0%}",
                    )

                st.markdown(
                    """
                    **Algorithm:** Random Forest Classifier

                    **Input features:** 10

                    **Target variable:** `loan_status`

                    - `0`: no default recorded for the loan
                    - `1`: default recorded for the loan
                    """
                )

                st.markdown("#### Existing model evaluation")

                evaluation_col1, evaluation_col2 = st.columns(2)

                with evaluation_col1:
                    st.metric("Accuracy", "91.58%")
                    st.metric("Default-class precision", "89%")

                with evaluation_col2:
                    st.metric("ROC-AUC", "92.09%")
                    st.metric("Default-class recall", "70%")

                st.caption(
                    "These values are the project's previously reported "
                    "evaluation results, not guarantees of future "
                    "performance. The test set was used during earlier "
                    "model and threshold exploration, so these results "
                    "should not be described as a fully independent "
                    "final estimate."
                )

                st.markdown("#### Limitations")

                st.markdown(
                    """
                    - Probability calibration has not been established.
                    - Display risk bands are provisional.
                    - The model identifies historical patterns; it
                      does not establish causation.
                    - Global feature importance does not explain an
                      individual prediction.
                    - Suitability for production lending decisions
                      has not been demonstrated.
                    """
                )

            # ------------------------------------------------
            # APPLICANT INPUT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "Review the applicant information used"
            ):

                readable_input = input_data.rename(
                    columns={
                        "person_age": "Applicant age (years)",
                        "person_income": "Annual income",
                        "person_home_ownership": "Housing situation",
                        "person_emp_length": "Employment length (years)",
                        "loan_intent": "Loan purpose",
                        "loan_amnt": "Requested loan amount",
                        "loan_int_rate": "Interest rate (%)",
                        "loan_percent_income": "Loan-to-income ratio",
                        "cb_person_default_on_file": (
                            "Previous default recorded"
                        ),
                        "cb_person_cred_hist_length": (
                            "Credit history length (years)"
                        ),
                    }
                )

                st.dataframe(
                    readable_input,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Verify all values, units and feature definitions "
                    "before interpreting the result."
                )

        except Exception as error:

            st.error(
                "CreditGuard could not generate the assessment. "
                "Verify the input values and saved model."
            )

            with st.expander("Technical error details"):
                st.exception(error)


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <div class="footer-brand">CreditGuard</div>
        <div>Credit Risk Assessment &amp; Loan Default Prediction</div>
        <div class="footer-note">
            Machine learning portfolio demonstration · Not a
            production lending decision system
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
