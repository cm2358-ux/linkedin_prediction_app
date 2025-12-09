# ===============================================================
# LINKEDIN USER PREDICTION DASHBOARD — IMPROVED FULL VERSION
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve

# --------------------------------------------------------------
# Streamlit Page Config + Theme Styling
# --------------------------------------------------------------
st.set_page_config(page_title="LinkedIn Prediction Dashboard", layout="wide")

st.markdown("""
<style>
.stApp { background: #212331; color: white; }
.big-title { font-size: 38px; font-weight: 700; margin-bottom: 15px; }
.section-title { font-size: 28px; font-weight: 600; margin-top: 20px; }
.pred-box { padding: 20px; border-radius: 10px; font-size: 24px; text-align: center; }
.profile-box { padding: 15px; background: rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

def info(text):  
    return f"<span title='{text}' style='cursor: help;'> ⓘ</span>"


# --------------------------------------------------------------
# Load Model + Dataset
# --------------------------------------------------------------
try:
    lr = joblib.load("model.pkl")
except:
    st.error("⚠️ model.pkl not found. Add it to your working directory.")
    st.stop()

df = pd.read_csv("social_media_usage.csv")

# --------------------------------------------------------------
# Data Cleaning + Feature Engineering
# --------------------------------------------------------------
df["sm_li"] = (df["web1h"] == 1).astype(int)
df["female"] = df["gender"].map({1: 0, 2: 1})

df.rename(columns={"educ2": "education", "par": "parent", "marital": "married"}, inplace=True)

df["parent"] = (df["parent"] == 1).astype(int)
df["married"] = (df["married"] == 1).astype(int)

df["income"] = df["income"].where(df["income"].between(1, 9))
df["education"] = df["education"].where(df["education"].between(1, 8))
df["age"] = df["age"].where(df["age"].between(1, 97))

df.dropna(subset=["income","education","parent","married","female","age","sm_li"], inplace=True)

X = df[["income","education","parent","married","female","age"]]
y = df["sm_li"]

income_labels = {
    1:"Less than $10K",2:"$10K–$20K",3:"$20K–$30K",
    4:"$30K–$40K",5:"$40K–$50K",6:"$50K–$75K",
    7:"$75K–$100K",8:"$100K–$150K",9:"$150K+"
}
education_labels = {
    1:"Less than High School",2:"High School",3:"Some College",
    4:"College Graduate",5:"Master’s",6:"Professional Degree",
    7:"Doctorate",8:"Technical/Other"
}


# --------------------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------------------
with st.sidebar:
    st.markdown("## Demographic Inputs")

    income = st.selectbox("Income", income_labels.keys(), format_func=lambda x: income_labels[x])
    education = st.selectbox("Education", education_labels.keys(), format_func=lambda x: education_labels[x])
    parent = st.selectbox("Parent", ["No","Yes"])
    married = st.selectbox("Married", ["No","Yes"])
    gender = st.selectbox("Gender", ["Male","Female"])
    age = st.slider("Age", 18, 97, 42)

parent_val = 1 if parent == "Yes" else 0
married_val = 1 if married == "Yes" else 0
female_val = 1 if gender == "Female" else 0

person = pd.DataFrame({
    "income":[income],
    "education":[education],
    "parent":[parent_val],
    "married":[married_val],
    "female":[female_val],
    "age":[age]
})


# --------------------------------------------------------------
# PAGE TABS
# --------------------------------------------------------------
tab_pred, tab_dynamic, tab_marketing, tab_shap, tab_perf = st.tabs(
    ["Prediction","Interactive Analytics","Marketing Insights","SHAP Explanation","Model Performance"]
)


# ==============================================================
# TAB 1 — PREDICTION RESULTS
# ==============================================================
with tab_pred:

    st.markdown("<h2 class='big-title'>Prediction Results</h2>", unsafe_allow_html=True)

    st.markdown("### Profile Summary")

    st.markdown(f"""
    <div class='profile-box'>
        <b>Age:</b> {age}<br>
        <b>Gender:</b> {gender}<br>
        <b>Education:</b> {education_labels[education]}<br>
        <b>Income:</b> {income_labels[income]}<br>
        <b>Married:</b> {married}<br>
        <b>Parent:</b> {parent}
    </div>
    """, unsafe_allow_html=True)

    proba = lr.predict_proba(person)[0][1]
    prediction = "LinkedIn User" if proba >= 0.5 else "Not a LinkedIn User"

    color = "#1b5e20" if prediction == "LinkedIn User" else "#8b0000"

    st.markdown(f"""
    <div class='pred-box' style="background:{color}">
        {prediction}<br>
        <span style='font-size:18px'>Probability: {proba:.3f} ({proba*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Probability Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        gauge={
            "axis": {"range":[0,100]},
            "steps":[
                {"range":[0,25], "color":"#8b0000"},
                {"range":[25,50], "color":"#ff6347"},
                {"range":[50,75], "color":"#ffd700"},
                {"range":[75,100], "color":"#2e8b57"}
            ],
            "bar":{"color":color}
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)


# ==============================================================
# TAB 2 — INTERACTIVE ANALYTICS
# ==============================================================
with tab_dynamic:

    st.markdown("<h2 class='big-title'>Interactive Analytics</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        f_income = st.checkbox("Filter by Income", True)
        f_parent = st.checkbox("Filter by Parent", True)
    with col2:
        f_edu = st.checkbox("Filter by Education", True)
        f_mar = st.checkbox("Filter by Marital Status", True)
    with col3:
        f_gender = st.checkbox("Filter by Gender", True)

    df_filtered = df.copy()
    if f_income: df_filtered = df_filtered[df_filtered["income"] == income]
    if f_edu: df_filtered = df_filtered[df_filtered["education"] == education]
    if f_parent: df_filtered = df_filtered[df_filtered["parent"] == parent_val]
    if f_mar: df_filtered = df_filtered[df_filtered["married"] == married_val]
    if f_gender: df_filtered = df_filtered[df_filtered["female"] == female_val]

    st.write(f"Rows returned: **{len(df_filtered)}**")

    if len(df_filtered):
        df_filtered["li_label"] = df_filtered["sm_li"].map({0:"Non-User",1:"User"})

        st.markdown("### LinkedIn Usage Breakdown")
        st.plotly_chart(
            px.pie(df_filtered, names="li_label", title="User vs Non-User"),
            use_container_width=True
        )

        st.markdown("### Age Distribution")
        st.plotly_chart(
            px.histogram(df_filtered, x="age", nbins=20, title="Age Distribution"),
            use_container_width=True
        )

    st.markdown("### Age Probability Curve")
    ages = np.arange(18,98)

    sweep = pd.DataFrame({
        "income":income,
        "education":education,
        "parent":parent_val,
        "married":married_val,
        "female":female_val,
        "age":ages
    })

    st.plotly_chart(
        px.line(x=ages, y=lr.predict_proba(sweep)[:,1], title="Predicted Probability Across Age"),
        use_container_width=True
    )


# ==============================================================
# TAB 3 — MARKETING INSIGHTS
# ==============================================================
with tab_marketing:

    st.markdown("<h2 class='big-title'>Marketing Audience Insights</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### Executive Summary  
    The model highlights which demographic segments show the strongest likelihood 
    of LinkedIn adoption. These insights guide audience targeting, messaging strategy, 
    and high-ROI segmentation.
    """)

    df_seg = df.copy()

    # EDUCATION
    st.markdown("## 1. Education Level")
    edu_rates = df_seg.groupby("education")["sm_li"].mean()
    st.plotly_chart(
        px.bar(x=[education_labels[i] for i in edu_rates.index], y=edu_rates, title="Usage by Education"),
        use_container_width=True
    )

    # INCOME
    st.markdown("## 2. Income Level")
    inc_rates = df_seg.groupby("income")["sm_li"].mean()
    st.plotly_chart(
        px.bar(x=[income_labels[i] for i in inc_rates.index], y=inc_rates, title="Usage by Income"),
        use_container_width=True
    )

    # AGE GROUPS
    st.markdown("## 3. Age Groups")
    df_seg["age_group"] = pd.cut(df_seg["age"], bins=[18,25,35,45,55,65,120],
                                 labels=["18–25","26–35","36–45","46–55","56–65","65+"])
    age_rates = df_seg.groupby("age_group")["sm_li"].mean()
    st.plotly_chart(
        px.line(x=age_rates.index.astype(str), y=age_rates, markers=True, title="Usage by Age Group"),
        use_container_width=True
    )

    # GENDER
    st.markdown("## 4. Gender")
    gender_rates = df_seg.groupby("female")["sm_li"].mean()
    st.plotly_chart(
        px.bar(x=["Male","Female"], y=[gender_rates.get(0), gender_rates.get(1)],
               title="Usage by Gender"),
        use_container_width=True
    )

    # MARITAL
    st.markdown("## 5. Marital Status")
    mar_rates = df_seg.groupby("married")["sm_li"].mean()
    st.plotly_chart(
        px.bar(x=["Not Married","Married"],
               y=[mar_rates.get(0), mar_rates.get(1)],
               title="Usage by Marital Status"),
        use_container_width=True
    )

    # PARENTS
    st.markdown("## 6. Parenthood")
    par_rates = df_seg.groupby("parent")["sm_li"].mean()
    st.plotly_chart(
        px.bar(x=["Not Parent","Parent"],
               y=[par_rates.get(0), par_rates.get(1)],
               title="Usage by Parenthood"),
        use_container_width=True
    )

    # STRATEGIC RECS
    st.markdown("## Strategic Recommendations")
    st.markdown("""
    **1. Prioritize high-probability adopters** — ages 26–45, higher-income, higher-educated  
    **2. Tailor messaging** — gender, marital, and parenthood patterns matter  
    **3. Use model probability outputs** to build more efficient audience segments  
    **4. Develop awareness strategies** for older and lower-income groups  
    """)


# ==============================================================
# TAB 4 — SHAP EXPLANATION
# ==============================================================
with tab_shap:

    st.markdown("<h2 class='big-title'>SHAP Model Explanation</h2>", unsafe_allow_html=True)

    explainer = shap.Explainer(lr, X)
    shap_vals = explainer(X)

    # Feature Importance
    st.markdown("### SHAP Feature Importance")
    fig = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals.values, X, plot_type="bar", show=False)
    st.pyplot(fig, clear_figure=True)

    # Summary Plot
    st.markdown("### SHAP Summary Plot")
    fig = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals.values, X, show=False)
    st.pyplot(fig, clear_figure=True)

    # Waterfall
    st.markdown("### SHAP Waterfall Plot (Selected Profile)")
    shap_person = explainer(person)

    fig = plt.figure(figsize=(6,4))
    shap.waterfall_plot(shap_person[0], show=False)
    st.pyplot(fig, clear_figure=True)


# =====================================
# Tab 5 - Market Performance Diagnostics
# ======================================
with tab_perf:

    st.markdown("<h2 class='big-title'>Model Performance Diagnostics</h2>", unsafe_allow_html=True)

    prob = lr.predict_proba(X)[:,1]
    preds = lr.predict(X)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig = plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig, clear_figure=True)

    # ROC
    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(y, prob)
    auc_score = roc_auc_score(y, prob)

    fig = plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    # Odd Ratio
    st.markdown("### Odds Ratios")
    fig = plt.figure(figsize=(6,4))
    sns.barplot(x=np.exp(lr.coef_[0]), y=X.columns)
    st.pyplot(fig, clear_figure=True)

    # PROBABILITY DISTRIBUTION
    st.markdown("### Probability Distribution")
    fig = plt.figure(figsize=(6,4))
    sns.histplot(prob, bins=20, kde=True)
    st.pyplot(fig, clear_figure=True)

    # CALIBRATION
    st.markdown("### Calibration Curve")
    true_prob, pred_prob = calibration_curve(y, prob, n_bins=10)
    fig = plt.figure(figsize=(6,4))
    plt.plot(pred_prob, true_prob, marker="o")
    plt.plot([0,1],[0,1],"--",color="gray")
    st.pyplot(fig, clear_figure=True)

    # PARTIAL DEPENDENCE
    st.markdown("### Partial Dependence — Age")
    age_range = np.arange(18,98)
    pdp_vals = [lr.predict_proba(X.assign(age=a))[:,1].mean() for a in age_range]

    fig = plt.figure(figsize=(6,4))
    plt.plot(age_range, pdp_vals)
    st.pyplot(fig, clear_figure=True)


# --------
# Footer
# ---------
st.markdown("---")
st.caption("Developed by Conal Masters — Georgetown MSBA — Machine Learning Dashboard")
