import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

import plotly.express as px
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Loan Dashboard", layout="wide")

st.title("💼 Loan Prediction Dashboard")
st.markdown("### Intelligent Decision Support System")

# ==============================
# SIDEBAR (INFO ONLY)
# ==============================
st.sidebar.title("⚙️ Control Panel")

show_info = st.sidebar.button("📘 Project Description")

if show_info:
    st.sidebar.success("""
This AI system predicts whether a loan should be approved based on applicant data.

It is used to:
- Reduce manual decision errors
- Improve banking efficiency
- Assess applicant risk using machine learning

Main features:
- Data exploration
- Model training
- Performance evaluation
- Real-time prediction system
""")

uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset", type=["csv"])

# ==============================
# MAIN
# ==============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ==============================
    # FILTERS (PROFESSIONAL CONTROL PANEL)
    # ==============================
    st.subheader("🔍 Smart Data Filters")

    colf1, colf2 = st.columns(2)

    with colf1:
        if "Loan_Status" in df.columns:
            status_filter = st.selectbox("Filter by Loan Status", ["All"] + list(df["Loan_Status"].unique()))

    with colf2:
        sample_pct = st.slider("Select dataset percentage to download", 10, 100, 100)

    filtered_df = df.copy()

    if "Loan_Status" in df.columns and status_filter != "All":
        filtered_df = filtered_df[filtered_df["Loan_Status"] == status_filter]

    download_df = filtered_df.sample(frac=sample_pct/100, random_state=42)

    st.download_button(
        "📥 Download Filtered Dataset",
        download_df.to_csv(index=False),
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )

    # ==============================
    # MEANINGFUL VISUALS
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Loan_Status", color="Loan_Status",
                        title="Loan Approval Distribution",
                        template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Credit_History" in df.columns:
            fig2 = px.histogram(df, x="Credit_History", color="Loan_Status",
                                barmode="group",
                                title="Credit History vs Loan Approval",
                                template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

    if "ApplicantIncome" in df.columns:
        fig3 = px.box(df, x="Loan_Status", y="ApplicantIncome",
                    color="Loan_Status",
                    title="Income Distribution by Loan Status",
                    template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # ==============================
    # FEATURE IMPORTANCE (MORE MEANINGFUL THAN CLUSTERING)
    # ==============================
    st.subheader("📌 Feature Importance (Why Model Decides)")

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"].map({"Y":1, "N":0})

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)

    fig4 = px.bar(feat_df, x="Importance", y="Feature",
                  orientation="h",
                  title="Top 10 Important Features",
                  template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    # ==============================
    # MODEL EVALUATION
    # ==============================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("F1 Score", f"{f1:.2f}")

    st.info("💡 Note: Accuracy depends on dataset quality. It cannot be forced to a fixed value like 90%.")

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, template="plotly_dark"), use_container_width=True)

    # ==============================
    # ROC CURVE
    # ==============================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc:.2f}"))
    fig5.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash")))
    fig5.update_layout(template="plotly_dark", title="ROC Curve")
    st.plotly_chart(fig5, use_container_width=True)

    # ==============================
    # PREDICTION SIMULATOR
    # ==============================
    st.subheader("🧠 Loan Prediction Simulator")

    input_data = {}

    for col in df.drop("Loan_Status", axis=1).columns:
        if df[col].dtype in ["int64", "float64"]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=(min_val+max_val)/2)
        else:
            input_data[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    if st.button("🚀 Predict Loan Approval"):
        pred = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        result_df = input_df.copy()
        result_df["Prediction"] = pred
        result_df["Confidence"] = prob

        if pred[0] == 1:
            st.success(f"✅ Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Rejected (Confidence: {prob:.2f})")

        st.download_button(
            "📥 Download This Prediction",
            result_df.to_csv(index=False),
            file_name="loan_prediction.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Upload a dataset from the sidebar to start")