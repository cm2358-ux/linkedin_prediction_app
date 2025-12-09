# =======================================
# LinkedIn User Prediction Dashboard
# =======================================

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

# -------------------------------------------------------
# STREAMLIT CONFIG + GLOBAL STYLE
# -------------------------------------------------------
st.set_page_config(page_title="LinkedIn User Prediction App", layout="wide")

st.markdown("""
<style>
.stApp { background: #212331; color: white; }
.big-title { font-size: 40px; font-weight: 700; }
.pred-box { padding: 20px; border-radius: 10px; font-size: 24px; text-align: center; }
.profile-box { padding: 15px; background: rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Tooltip helper
def info(text):
    return f"<span title='{text}' style='cursor: help;'> ℹ️</span>"


# -------------------------------------------------------
# LOAD MODEL + DATA
# -------------------------------------------------------
lr = joblib.load("model.pkl")

df = pd.read_csv("social_media_usage.csv")
df["sm_li"] = (df["web1h"] == 1).astype(int)
df["female"] = df["gender"].apply(lambda x: 1 if x == 2 else 0 if x == 1 else np.nan)
df = df.rename(columns={"educ2": "education", "par": "parent", "marital": "married"})

X = df[["income", "education", "parent", "married", "female", "age"]]
y = df["sm_li"]

mask = X.notna().all(axis=1)
df = df[mask]
X = X[mask]
y = y[mask]


# -------------------------------------------------------
# LABEL DICTIONARIES
# -------------------------------------------------------
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

# Safe label mapper
def safe_map(series, mapping):
    return [mapping.get(v, f"Other ({v})") for v in series]


# -------------------------------------------------------
# SIDEBAR — USER INPUTS
# -------------------------------------------------------
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


# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab_pred, tab_dynamic, tab_shap, tab_marketing, tab_perf = st.tabs(
    ["Prediction","Interactive Analytics","SHAP Explanation","Marketing Insights","Model Performance"]
)

# ======================================================
# TAB 1 — PREDICTION
# ======================================================
with tab_pred:

    st.markdown("<h2 class='big-title'>Prediction Results</h2>", unsafe_allow_html=True)

    st.markdown(f"### Profile Summary {info('Inputs used to generate the prediction.')}", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='profile-box'>
        <b>Income:</b> {income_labels[income]}<br>
        <b>Education:</b> {education_labels[education]}<br>
        <b>Parent:</b> {parent}<br>
        <b>Married:</b> {married}<br>
        <b>Gender:</b> {gender}<br>
        <b>Age:</b> {age}
    </div>
    """, unsafe_allow_html=True)

    proba = lr.predict_proba(person)[0][1]
    prediction = "LinkedIn User" if proba >= 0.5 else "Not a LinkedIn User"
    color = "#1b5e20" if prediction == "LinkedIn User" else "#8b0000"

    st.markdown(f"""
    <div class='pred-box' style='background:{color}'>
        {prediction}<br>
        <span style='font-size:18px'>Probability: {proba:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Probability Gauge {info('Visual gauge of predicted LinkedIn usage probability.')}", unsafe_allow_html=True)

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
            "bar":{"color":"white"}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)


# ======================================================
# TAB 2 — INTERACTIVE ANALYTICS
# ======================================================
with tab_dynamic:

    st.markdown(f"## Interactive Analytics {info('Explore dataset patterns using filters.')}", unsafe_allow_html=True)

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

    st.markdown(f"### Filtered Dataset Overview {info('Shows number of rows that match selected criteria.')}", unsafe_allow_html=True)
    st.write(f"Rows returned: **{len(df_filtered)}**")

    if len(df_filtered) > 0:

        st.markdown(f"### LinkedIn Usage Breakdown {info('Percentage of LinkedIn vs non-users.')}", unsafe_allow_html=True)
        st.plotly_chart(px.pie(df_filtered, names="sm_li", color="sm_li",
            color_discrete_map={0:"red",1:"green"}), use_container_width=True)

        st.markdown(f"### Age Distribution {info('Distribution of ages in the filtered subset.')}", unsafe_allow_html=True)
        st.plotly_chart(px.histogram(df_filtered, x="age", nbins=20,
                                     color_discrete_sequence=["cyan"]),
                        use_container_width=True)

    st.markdown(f"### Age Probability Curve {info('Predicted LinkedIn probability as age varies.')}", unsafe_allow_html=True)

    ages = np.arange(18,98)
    sweep = pd.DataFrame({
        "income":income, "education":education,
        "parent":parent_val, "married":married_val,
        "female":female_val, "age":ages
    })

    st.plotly_chart(px.line(x=ages, y=lr.predict_proba(sweep)[:,1],
        labels={"x":"Age","y":"Predicted Probability"},
        title="Predicted Probability Across Age"), use_container_width=True)


# ======================================================
# TAB 3 — SHAP VALUES
# ======================================================
with tab_shap:

    st.markdown(f"<h2 class='big-title'>SHAP Model Explanation</h2>", unsafe_allow_html=True)

    explainer = shap.LinearExplainer(lr, X)
    shap_vals = explainer.shap_values(X)

    st.markdown(f"### SHAP Feature Importance {info('Shows which features impact predictions the most.')}", unsafe_allow_html=True)
    fig1 = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.markdown(f"### SHAP Summary Plot {info('Shows direction + magnitude of influence for each feature.')}", unsafe_allow_html=True)
    fig2 = plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals, X, show=False)
    st.pyplot(fig2)

    st.markdown(f"### SHAP Waterfall Plot {info('Explains how THIS prediction was formed.')}", unsafe_allow_html=True)
    fig3 = plt.figure(figsize=(6,4))
    shap.waterfall_plot(
        shap.Explanation(values=explainer.shap_values(person)[0],
                         base_values=explainer.expected_value,
                         data=person.iloc[0],
                         feature_names=person.columns),
        show=False
    )
    st.pyplot(fig3)


# ======================================================
# TAB 4 — MARKETING INSIGHTS (CONSULTING-GRADE VERSION)
# ======================================================
with tab_marketing:

    st.markdown("<h2 class='big-title'>Marketing Audience Insights</h2>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------
    st.markdown("""
    ### Executive Summary

    The LinkedIn prediction model highlights **which demographic profiles are most likely to adopt and actively use LinkedIn**.  
    These insights support the marketing team in:

    - Identifying high-value target segments  
    - Tailoring messaging and creative content by demographic  
    - Optimizing paid media and campaign allocation  
    - Expanding adoption among under-penetrated groups  

    The following sections break down adoption patterns across education, income, age, gender, marital status, and parenthood.
    """)

    # ------------------------------------------------------
    # EDUCATION INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 1. Education Level {info('Higher education is strongly associated with LinkedIn usage.')}",
        unsafe_allow_html=True
    )

    edu_rates = df.groupby("education")["sm_li"].mean().sort_index()
    edu_x = [education_labels.get(int(i), f"Other ({int(i)})") for i in edu_rates.index]

    st.plotly_chart(
        px.bar(
            x=edu_x,
            y=edu_rates.values,
            labels={"x": "Education Level", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Education Level"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    Users with **college, master’s, or professional degrees** exhibit the highest LinkedIn usage.  
    Messaging should emphasize **career advancement, networking, leadership**, and **professional credibility**.
    """)

    # ------------------------------------------------------
    # INCOME INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 2. Income Level {info('Income reflects seniority and professional engagement levels.')}",
        unsafe_allow_html=True
    )

    inc_rates = df.groupby("income")["sm_li"].mean().sort_index()
    inc_x = [income_labels.get(int(i), f"Other ({int(i)})") for i in inc_rates.index]

    st.plotly_chart(
        px.bar(
            x=inc_x,
            y=inc_rates.values,
            labels={"x": "Income Bracket", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Income"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    Higher-income users show the **strongest adoption**, making them ideal for targeted professional campaigns.  
    Position LinkedIn as a platform for **promotion opportunities, executive insights, and career acceleration**.
    """)

    # ------------------------------------------------------
    # AGE INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 3. Age Demographics {info('Age is one of the strongest predictors of LinkedIn adoption.')}",
        unsafe_allow_html=True
    )

    df_age = df.copy()
    df_age["age_group"] = pd.cut(
        df_age["age"],
        bins=[18, 25, 35, 45, 55, 65, 120],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
    )

    age_rates = df_age.groupby("age_group")["sm_li"].mean()

    st.plotly_chart(
        px.line(
            x=age_rates.index.astype(str),
            y=age_rates.values,
            markers=True,
            labels={"x": "Age Group", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Age Group"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    LinkedIn adoption peaks between **ages 26–45**, representing early-career and mid-career professionals.  
    Campaigns should target these users with **leadership pathways, promotions, salary growth**, and **network-building themes**.
    """)

    # ------------------------------------------------------
    # GENDER INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 4. Gender Differences {info('Gender-based engagement varies modestly.')}",
        unsafe_allow_html=True
    )

    gender_rates = df.groupby("female")["sm_li"].mean()

    st.plotly_chart(
        px.bar(
            x=["Male", "Female"],
            y=[gender_rates.get(0, np.nan), gender_rates.get(1, np.nan)],
            labels={"x": "Gender", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Gender"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    Women show slightly **higher LinkedIn usage**.  
    Effective messaging includes themes around **workplace empowerment, mentorship, and career equity**.
    """)

    # ------------------------------------------------------
    # MARITAL STATUS INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 5. Marital Status {info('Lifestyle segmentation reveals behavioral differences.')}",
        unsafe_allow_html=True
    )

    df["married_clean"] = df["married"].apply(lambda x: 1 if x == 1 else 0)
    mar_rates = df.groupby("married_clean")["sm_li"].mean().sort_index()

    st.plotly_chart(
        px.bar(
            x=["Not Married", "Married"],
            y=[mar_rates.get(0, 0), mar_rates.get(1, 0)],
            labels={"x": "Marital Status", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Marital Status"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    Non-married individuals display **higher adoption**, responding strongly to messaging around  
    **career mobility, independence, and accelerated growth opportunities**.
    """)

    # ------------------------------------------------------
    # PARENTHOOD INSIGHTS
    # ------------------------------------------------------
    st.markdown(
        f"## 6. Parenthood Status {info('Parental responsibilities influence content needs and messaging responsiveness.')}",
        unsafe_allow_html=True
    )

    df["parent_binary"] = df["parent"].apply(lambda x: 1 if x == 1 else 0)
    par_rates = df.groupby("parent_binary")["sm_li"].mean()

    st.plotly_chart(
        px.bar(
            x=["Not Parent", "Parent"],
            y=[par_rates.get(0, 0), par_rates.get(1, 0)],
            labels={"x": "Parent Status", "y": "LinkedIn Usage Rate"},
            title="LinkedIn Usage by Parenthood"
        ),
        use_container_width=True
    )

    st.markdown("""
    **Marketing Implication:**  
    Non-parents adopt LinkedIn at higher rates, but parent users can be activated through themes of  
    **flexibility, remote work, career stability**, and **balancing professional and personal life**.
    """)

    # ------------------------------------------------------
    # STRATEGIC MARKETING RECOMMENDATIONS
    # ------------------------------------------------------
    st.markdown("## Strategic Recommendations")

    st.write("""
    **1. Focus spend on high-probability adopters**  
    Target ages **26–45**, higher income brackets, and college-educated users.

    **2. Develop differentiated creative by segment**  
    - Women → empowerment, leadership, mentorship  
    - Parents → flexibility, remote-work opportunities  
    - Singles → accelerated career growth  

    **3. Optimize paid media targeting**  
    Use probability scores to identify high-ROI audience segments.

    **4. Expand penetration in low-usage groups**  
    Use awareness campaigns for older adults and lower-income brackets.

    **5. Integrate segmentation into campaign planning**  
    Build look-alike audiences, refine bidding strategies, and allocate spend dynamically.
    """)

    # ------------------------------------------------------
    # KEY TAKEAWAYS
    # ------------------------------------------------------
    st.markdown("## Key Takeaways")
    st.write("""
    - Education and income are the **strongest structural predictors** of LinkedIn usage  
    - Ages 26–45 form the **core engagement demographic**  
    - Women show slightly higher adoption  
    - Lifestyle factors (marriage, parenthood) provide **valuable segmentation opportunities**  
    - Insights directly support **media planning, creative strategy, and audience targeting**  
    """)



# ======================================================
# TAB 5 — MODEL PERFORMANCE
# ======================================================
with tab_perf:

    st.markdown("<h2 class='big-title'>Model Diagnostics</h2>", unsafe_allow_html=True)

    prob = lr.predict_proba(X)[:,1]

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------
    st.markdown(f"### Confusion Matrix {info('Shows correct vs incorrect classifications.')}", unsafe_allow_html=True)
    cm = confusion_matrix(y, lr.predict(X))
    fig = plt.figure(figsize=(10,6))       # Zoomed out
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    # ------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------
    st.markdown(f"### ROC Curve {info('Evaluates model’s ability to distinguish classes.')}", unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y, prob)
    fig = plt.figure(figsize=(10,6))       # Zoomed out
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y,prob):.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend()
    st.pyplot(fig)

    # ------------------------------------------------------
    # Odds Ratios
    # ------------------------------------------------------
    st.markdown(f"### Odds Ratios {info('Shows multiplicative effect each feature has on odds of LinkedIn use.')}", unsafe_allow_html=True)
    fig = plt.figure(figsize=(10,6))       # Zoomed out
    sns.barplot(x=np.exp(lr.coef_[0]), y=X.columns)
    st.pyplot(fig)

    # ------------------------------------------------------
    # Probability Distribution
    # ------------------------------------------------------
    st.markdown(f"### Probability Distribution {info('Shows model confidence across dataset?')}", unsafe_allow_html=True)
    fig = plt.figure(figsize=(10,6))       # Zoomed out
    sns.histplot(prob, bins=20, kde=True, color="cyan")
    st.pyplot(fig)

    # ------------------------------------------------------
    # Calibration Curve
    # ------------------------------------------------------
    st.markdown(f"### Calibration Curve {info('Tests probability calibration vs real outcomes.')}", unsafe_allow_html=True)
    true_prob, pred_prob = calibration_curve(y, prob, n_bins=10)
    fig = plt.figure(figsize=(10,6))       # Zoomed out
    plt.plot(pred_prob, true_prob, marker="o")
    plt.plot([0,1],[0,1],"--")
    st.pyplot(fig)

    # ------------------------------------------------------
    # Partial Dependence: Age
    # ------------------------------------------------------
    st.markdown(f"### Partial Dependence: Age {info('Shows isolated effect of age on prediction.')}", unsafe_allow_html=True)
    age_range = np.arange(18,98)
    X_temp = X.copy()
    pdp_vals = []

    for a in age_range:
        X_temp["age"] = a
        pdp_vals.append(lr.predict_proba(X_temp)[:,1].mean())

    fig = plt.figure(figsize=(10,6))       # Zoomed out
    plt.plot(age_range, pdp_vals)
    plt.xlabel("Age")
    plt.ylabel("Predicted Probability")
    st.pyplot(fig)


# -------------------------------------------------------
st.markdown("---")
st.caption("Developed by Conal Masters — Georgetown MSBA — Machine Learning Dashboard")


