# ------------------------------------------------------------
# LinkedIn User Prediction Dashboard
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Streamlit Configuration and Global Style
# ------------------------------------------------------------
st.set_page_config(page_title="LinkedIn User Prediction App", layout="wide")

st.markdown("""
<style>
.stApp { background: #212331; color: white; }
.big-title { font-size: 40px; font-weight: 700; }
.pred-box { padding: 20px; border-radius: 10px; font-size: 24px; text-align: center; }
.profile-box { padding: 15px; background: rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

def info(text):
    return f"<span title='{text}' style='cursor: help;'> Info</span>"


# ------------------------------------------------------------
# Load Model and Data Cleaning
# ------------------------------------------------------------
try:
    lr = joblib.load("model.pkl")
except:
    st.error("Model file 'model.pkl' not found.")
    st.stop()

df = pd.read_csv("social_media_usage.csv")

df["sm_li"] = (df["web1h"] == 1).astype(int)
df["female"] = df["gender"].map({1: 0, 2: 1})

df = df.rename(columns={"educ2": "education", "par": "parent", "marital": "married"})

df["parent"] = (df["parent"] == 1).astype(int)
df["married"] = (df["married"] == 1).astype(int)

df["income"] = df["income"].where(df["income"].between(1, 9))
df["education"] = df["education"].where(df["education"].between(1, 8))
df["age"] = df["age"].where(df["age"].between(1, 97))

df = df.dropna(subset=["income", "education", "parent", "married", "female", "age", "sm_li"])

X = df[["income", "education", "parent", "married", "female", "age"]]
y = df["sm_li"]


# ------------------------------------------------------------
# Label Dictionaries
# ------------------------------------------------------------
income_labels = {
    1:"Less than $10K", 2:"$10K–$20K", 3:"$20K–$30K",
    4:"$30K–$40K", 5:"$40K–$50K", 6:"$50K–$75K",
    7:"$75K–$100K", 8:"$100K–$150K", 9:"$150K+"
}

education_labels = {
    1:"Less than High School", 2:"High School", 3:"Some College",
    4:"College Graduate", 5:"Master’s", 6:"Professional Degree",
    7:"Doctorate", 8:"Technical/Other"
}


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## Demographic Inputs")

    income = st.selectbox("Income", list(income_labels.keys()), format_func=lambda x: income_labels[x])
    education = st.selectbox("Education", list(education_labels.keys()), format_func=lambda x: education_labels[x])
    parent = st.selectbox("Parent", ["No", "Yes"])
    married = st.selectbox("Married", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])
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


# ------------------------------------------------------------
# Tab Structure
# ------------------------------------------------------------
tab_pred, tab_marketing, tab_shap, tab_dynamic, tab_perf = st.tabs(
    ["Prediction", "Marketing Insights", "SHAP Explanation",
     "Interactive Analytics", "Model Performance"]
)


# ------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------
with tab_pred:

    st.markdown("<h2 class='big-title'>Prediction Results</h2>", unsafe_allow_html=True)

    st.markdown(
        f"### Profile Summary {info('Inputs used to generate the prediction.')}",
        unsafe_allow_html=True
    )

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
    <div class='pred-box' style='background:{color}'>
        {prediction}<br>
        <span style='font-size:18px'>
            Probability: {proba:.3f} ({proba*100:.1f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "This probability reflects the model’s estimated likelihood of LinkedIn usage given the selected demographics."
    )

    st.markdown("---")

    st.markdown(
        f"### Probability Gauge {info('Visual gauge of predicted LinkedIn usage probability.')}",
        unsafe_allow_html=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        gauge={
            "axis": {"range":[0,100]},
            "steps":[
                {"range":[0,25],"color":"#8b0000"},
                {"range":[25,50],"color":"#ff6347"},
                {"range":[50,75],"color":"#ffd700"},
                {"range":[75,100],"color":"#2e8b57"},
            ],
            "bar":{"color": color}
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — MARKETING INSIGHTS
# ------------------------------------------------------------
with tab_marketing:

    st.markdown("<h2 class='big-title'>Marketing Audience Insights</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### Executive Summary
    The prediction model highlights which demographic groups are most likely to use LinkedIn.
    These insights support audience targeting, message development, and paid media allocation.
    Education, income, and age show the strongest relationships with platform adoption.
    """)

    df_seg = df.copy()

    # EDUCATION
    st.markdown("## Education Level")
    edu_rates = df_seg.groupby("education")["sm_li"].mean()
    st.plotly_chart(px.bar(x=[education_labels[i] for i in edu_rates.index],
                           y=edu_rates.values,
                           title="LinkedIn Usage by Education"), use_container_width=True)

    # INCOME
    st.markdown("## Income Level")
    inc_rates = df_seg.groupby("income")["sm_li"].mean()
    st.plotly_chart(px.bar(x=[income_labels[i] for i in inc_rates.index],
                           y=inc_rates.values,
                           title="LinkedIn Usage by Income"), use_container_width=True)

    # AGE
    st.markdown("## Age Groups")
    df_seg["age_group"] = pd.cut(df_seg["age"],
                                 bins=[18,25,35,45,55,65,120],
                                 labels=["18–25","26–35","36–45","46–55","56–65","65+"])
    age_rates = df_seg.groupby("age_group")["sm_li"].mean()
    st.plotly_chart(px.line(x=age_rates.index.astype(str),
                            y=age_rates.values,
                            title="LinkedIn Usage by Age Group"), use_container_width=True)

    # GENDER
    st.markdown("## Gender Differences")
    gender_rates = df_seg.groupby("female")["sm_li"].mean()
    st.plotly_chart(px.bar(x=["Male","Female"],
                           y=[gender_rates.get(0,0), gender_rates.get(1,0)],
                           title="LinkedIn Usage by Gender"), use_container_width=True)

    # MARITAL STATUS
    st.markdown("## Marital Status")
    mar_rates = df_seg.groupby("married")["sm_li"].mean()
    st.plotly_chart(px.bar(x=["Not Married","Married"],
                           y=[mar_rates.get(0,0), mar_rates.get(1,0)],
                           title="LinkedIn Usage by Marital Status"), use_container_width=True)

    # PARENTHOOD
    st.markdown("## Parenthood")
    par_rates = df_seg.groupby("parent")["sm_li"].mean()
    st.plotly_chart(px.bar(x=["Not Parent","Parent"],
                           y=[par_rates.get(0,0), par_rates.get(1,0)],
                           title="LinkedIn Usage by Parenthood"), use_container_width=True)


# ------------------------------------------------------------
# TAB 3 — SHAP EXPLANATION
# ------------------------------------------------------------
with tab_shap:

    st.markdown("<h2 class='big-title'>SHAP Model Explanation</h2>", unsafe_allow_html=True)

    explainer = shap.Explainer(lr, X)
    shap_vals = explainer(X)

    st.markdown("### Feature Importance")
    fig1 = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals.values, X, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.markdown("### Summary Plot")
    fig2 = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals.values, X, show=False)
    st.pyplot(fig2)

    st.markdown("### Waterfall Plot for Selected Profile")
    shap_person = explainer(person)
    fig3 = plt.figure(figsize=(6,4))
    shap.waterfall_plot(shap_person[0], show=False)
    st.pyplot(fig3)


# ------------------------------------------------------------
# TAB 4 — INTERACTIVE ANALYTICS
# ------------------------------------------------------------
with tab_dynamic:

    st.markdown("## Interactive Analytics")

    col1, col2, col3 = st.columns(3)
    with col1:
        f_income = st.checkbox("Filter by Income", True)
        f_parent = st.checkbox("Filter by Parent", True)

    with col2:
        f_edu = st.checkbox("Filter by Education", True)
        f_mar = st.checkbox("Filter by Marital Status", True)

    with col3:
        f_gender = st.checkbox("Filter by Gender", True)

    df_f = df.copy()
    if f_income: df_f = df_f[df_f["income"] == income]
    if f_edu: df_f = df_f[df_f["education"] == education]
    if f_parent: df_f = df_f[df_f["parent"] == parent_val]
    if f_mar: df_f = df_f[df_f["married"] == married_val]
    if f_gender: df_f = df_f[df_f["female"] == female_val]

    st.write(f"Rows returned: {len(df_f)}")

    df_f["li_label"] = df_f["sm_li"].map({0:"Non-User",1:"User"})

    st.plotly_chart(px.pie(df_f, names="li_label",
                           color="li_label",
                           color_discrete_map={"Non-User":"red","User":"green"}),
                    use_container_width=True)

    st.plotly_chart(px.histogram(df_f, x="age",
                                 nbins=20,
                                 title="Age Distribution"),
                    use_container_width=True)

    ages = np.arange(18,98)
    sweep = pd.DataFrame({
        "income":income,
        "education":education,
        "parent":parent_val,
        "married":married_val,
        "female":female_val,
        "age":ages
    })

    st.plotly_chart(px.line(x=ages,
                            y=lr.predict_proba(sweep)[:,1],
                            title="Predicted Probability Across Age"),
                    use_container_width=True)


# ------------------------------------------------------------
# TAB 5 — MODEL PERFORMANCE (APPENDIX)
# ------------------------------------------------------------
with tab_perf:

    st.markdown("<h2 class='big-title'>Model Performance</h2>", unsafe_allow_html=True)

    preds = lr.predict(X)
    prob = lr.predict_proba(X)[:,1]

    cm = confusion_matrix(y, preds)
    fig = plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0","Predicted 1"],
                yticklabels=["Actual 0","Actual 1"])
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y, prob)
    auc_score = roc_auc_score(y, prob)
    fig = plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

    odds = np.exp(lr.coef_[0])
    fig = plt.figure(figsize=(7,5))
    sns.barplot(x=odds, y=X.columns)
    plt.title("Odds Ratios")
    st.pyplot(fig)

    fig = plt.figure(figsize=(7,5))
    sns.histplot(prob, bins=20, kde=True)
    plt.title("Predicted Probability Distribution")
    st.pyplot(fig)

    true_prob, pred_prob = calibration_curve(y, prob, n_bins=10)
    fig = plt.figure(figsize=(7,5))
    plt.plot(pred_prob, true_prob, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.title("Calibration Curve")
    st.pyplot(fig)

    age_range = np.arange(18,98)
    X_temp = X.copy()
    pdp_vals = []
    for a in age_range:
        X_temp["age"] = a
        pdp_vals.append(lr.predict_proba(X_temp)[:,1].mean())
    fig = plt.figure(figsize=(7,5))
    plt.plot(age_range, pdp_vals)
    plt.title("Partial Dependence: Age")
    st.pyplot(fig)


# ------------------------------------------------------------
st.markdown("---")
st.caption("Developed by Conal Masters — Georgetown MSBA — Machine Learning Dashboard")




