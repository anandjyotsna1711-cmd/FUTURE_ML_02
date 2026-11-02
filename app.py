
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TicketIQ | Support Intelligence",
    page_icon="🎫",
    layout="wide"
)

BASE = Path(__file__).parent

# ---------------- STYLING ----------------

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #edf2ff;
}
[data-testid="stSidebar"] {
    background: #111a2e;
    border-right: 1px solid #263451;
}
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #312e81, #164e63);
    border: 1px solid #394b83;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}
.hero p {
    color: #cbd5e1;
    font-size: 15px;
}
.metric-card {
    background: #151f35;
    border: 1px solid #293958;
    border-radius: 14px;
    padding: 18px;
}
.result-card {
    background: #14243a;
    border: 1px solid #31527a;
    border-radius: 15px;
    padding: 22px;
    margin-top: 12px;
}
.small-muted {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    category = joblib.load(
        BASE / "models" / "category_classifier_svm.joblib"
    )
    priority = joblib.load(
        BASE / "models" / "priority_classifier.joblib"
    )
    return category, priority

try:
    category_model, priority_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    model_error = str(e)

# ---------------- LOAD ANALYTICS DATA ----------------

@st.cache_data
def load_data():
    category_file = BASE / "data" / "priority_labeling_v2.csv"
    priority_file = BASE / "data" / "priority_training.csv"

    category_data = pd.read_csv(category_file)
    priority_data = pd.read_csv(priority_file)

    return category_data, priority_data

try:
    category_data, priority_data = load_data()
    data_ready = True
except Exception:
    data_ready = False

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🎫 TicketIQ")
    st.caption("SUPPORT INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Ticket Classifier",
            "Priority Predictor",
            "Model Insights"
        ]
    )

    st.divider()
    st.markdown("**MODEL STATUS**")

    if models_ready:
        st.success("Category model loaded")
        st.success("Priority model loaded")
    else:
        st.error("Model loading failed")

    st.markdown("---")
    st.caption("Built with Python • NLP • Machine Learning")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>TicketIQ</h1>
    <p>AI-powered support ticket classification and prioritization</p>
    <p>Turning unstructured support requests into actionable insights.</p>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error("Could not load the trained models.")
    st.code(model_error)
    st.stop()

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.subheader("Platform Overview")
    st.caption("An overview of the support ticket intelligence system.")

    if data_ready:
        total_tickets = len(category_data)

        category_counts = (
            category_data["category_truth"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        category_counts.columns = ["Category", "Tickets"]

        priority_counts = (
            priority_data["priority"]
            .fillna("Unclassified")
            .value_counts()
            .reset_index()
        )
        priority_counts.columns = ["Priority", "Tickets"]

        low_count = int(
            priority_data["priority"].eq("Low").sum()
        )
        medium_count = int(
            priority_data["priority"].eq("Medium").sum()
        )
        high_count = int(
            priority_data["priority"].eq("High").sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Tickets", f"{total_tickets:,}")
        c2.metric("Categories", len(category_counts))
        c3.metric("Low Priority", low_count)
        c4.metric("Medium Priority", medium_count)

        st.markdown("### Ticket Distribution")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Category Breakdown")
            fig = px.bar(
                category_counts,
                x="Category",
                y="Tickets",
                color="Category",
                template="plotly_dark",
                text="Tickets"
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Number of Tickets"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Priority Distribution")
            fig = px.pie(
                priority_counts,
                names="Priority",
                values="Tickets",
                hole=0.58,
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "High-priority examples are limited. "
            "The trained priority model predicts Low and Medium only."
        )

    else:
        st.warning(
            "Local analytics datasets are missing. "
            "Place the project CSV files inside the data folder."
        )

    st.markdown("### Intelligence Workflow")

    a, b, c = st.columns(3)

    a.markdown("**01 · Text Processing**")
    a.write("Ticket descriptions are converted into TF-IDF features.")

    b.markdown("**02 · Classification**")
    b.write("Linear SVM models identify ticket categories and priority.")

    c.markdown("**03 · Insights**")
    c.write("Predictions help organize and understand incoming requests.")

# ---------------- CATEGORY CLASSIFIER ----------------

elif page == "Ticket Classifier":

    st.subheader("Ticket Category Classifier")
    st.caption("Enter a support ticket to identify its service category.")

    ticket = st.text_area(
        "Ticket Description",
        placeholder="Example: I cannot access my shared mailbox. Please help restore access.",
        height=160
    )

    if st.button("Classify Ticket", type="primary"):
        if not ticket.strip():
            st.warning("Please enter a ticket description.")
        else:
            prediction = category_model.predict([ticket])[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="small-muted">PREDICTED CATEGORY</p>
                    <h1>{prediction}</h1>
                    <p>Classification completed using the trained Linear SVM model.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "The selected Linear SVM model does not provide "
                "calibrated probability confidence scores."
            )

# ---------------- PRIORITY PREDICTOR ----------------

elif page == "Priority Predictor":

    st.subheader("Ticket Priority Predictor")
    st.caption("Predict whether a ticket should receive Low or Medium priority.")

    ticket = st.text_area(
        "Describe the support issue",
        placeholder="Example: The application is unavailable and I cannot complete my work.",
        height=160
    )

    if st.button("Predict Priority", type="primary"):
        if not ticket.strip():
            st.warning("Please enter a ticket description.")
        else:
            prediction = priority_model.predict([ticket])[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="small-muted">PREDICTED PRIORITY</p>
                    <h1>{prediction}</h1>
                    <p>Prediction generated by the trained priority classifier.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This model supports Low and Medium predictions only. "
                "It has not been trained to identify High priority tickets."
            )

# ---------------- MODEL INSIGHTS ----------------

elif page == "Model Insights":

    st.subheader("Model Performance")
    st.caption("Evaluation results from the project training experiments.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Category Classification")
        st.metric("5-Fold CV Accuracy", "68.24%")
        st.metric("5-Fold Macro F1", "65.62%")
        st.metric("Holdout Accuracy", "66.67%")

        st.markdown("**Selected algorithm:** Linear SVM")
        st.markdown("**Features:** Word + Character TF-IDF")

    with right:
        st.markdown("### Priority Classification")
        st.metric("5-Fold CV Accuracy", "80.61%")
        st.metric("Holdout Accuracy", "83.64%")
        st.metric("Macro F1-Score", "0.83")

        st.markdown("**Selected algorithm:** Linear SVM")
        st.markdown("**Classes:** Low and Medium")

    st.divider()
    st.markdown("### Priority Confusion Matrix")

    matrix_file = BASE / "results" / "priority_confusion_matrix.png"

    if matrix_file.exists():
        st.image(
            str(matrix_file),
            caption="Priority model holdout confusion matrix",
            use_container_width=True
        )
    else:
        st.info("Confusion matrix image not found.")

    st.caption(
        "Metrics are from the project's recorded holdout and "
        "cross-validation experiments."
    )

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "TicketIQ | Support Ticket Classification & Prioritization | "
    "Machine Learning Project"
)
