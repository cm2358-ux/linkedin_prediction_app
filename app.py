# ============================================================
# LINKEDIN USER PREDICTION DASHBOARD — FINAL FULL VERSION
# ============================================================

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
.big-title { font-size: 40px; font-weight: 700; margin-bottom: 15px; }
.pred-box { padding: 20px; border-radius: 10px; font-size: 24px; text-align: center; }
.profile-box { padding: 15px; background: rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# Tooltip Helper
def info(text):
    return f"<span title='{text}' style='cursor: help;'> Info</span>"


# ------------------------------------------------------------
# Load Model and Dataset + Apply Cleaning
# ------------------------------------------------------------
try:
    lr = joblib.load("model.pkl")
except:
    st.error("model.pkl not found. Please place it in the working directory.")
    st.stop()

df = pd.read_csv("social_media_usage.csv")

# Target variable
df["sm_li"] = (df["web1h"] == 1).astype(int)

# Gender recode (female = 1, male = 0)
df["female"] = df["gender"].map({1: 0, 2: 1})

# Rename columns
df = df.rename(columns={"educ2": "education", "par": "parent", "marital": "married"})

# Convert parent and married to binary
df["parent"] = (df["parent"] == 1).astype(int)
df["married"] = (df["married"] == 1).astype(int)

# Valid ranges
df["income"] = df["income"].where(df["income"].between(1, 9))
df["education"] = df["education"].where(df["education"].between(1, 8))
df["age"] = df["age"].where(df["age"].between(1, 97))

# Drop missing
df = df.dropna(subset=["income","education","parent","married","female","age","sm_li"])

X = df[["income","education","parent","married","female","age"]]
y = df["sm_li"]


# ------------------------------------------------------------
# Label Dictionaries
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## Demographic Inputs")

    income = st.selectbox("Income", list(income_labels.keys()),
                           format_func=lambda x: income_labels[x])

    education = st.selectbox("Education", list(education_labels.keys()),
                              format_func=lambda x: education_labels[x])

    parent = st.selectbox("Parent", ["No", "Yes"])
    married = st.selectbox("Married", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 97, 42)

# Convert inputs to numeric
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
# Main Tabs (Final Order)
# ------------------------------------------------------------
tab_pred, tab_dynamic, tab_marketing, tab_shap, tab_perf = st.tabs(
    ["Prediction","Interactive Analytics","Marketing Insights","SHAP Explanation","Model Performance"]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================
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
        <span style='font-size:18px'>Probability: {proba:.3f} ({proba*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        f"### Probability Gauge {info('Visual gauge of predicted LinkedIn usage probability.')}",
        unsafe_allow_html=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,25],"color":"#8b0000"},
                {"range":[25,50],"color":"#ff6347"},
                {"range":[50,75],"color":"#ffd700"},
                {"range":[75,100],"color":"#2e8b57"},
            ],
            "bar":{"color":color}
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)



# ============================================================
# TAB 2 — INTERACTIVE ANALYTICS
# ============================================================
with tab_dynamic:

    st.markdown(
        f"## Interactive Analytics {info('Explore LinkedIn usage patterns across demographic groups.')}",
        unsafe_allow_html=True
    )

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

    df_filtered["li_label"] = df_filtered["sm_li"].map({0:"Non-User",1:"User"})

    if len(df_filtered) > 0:

        st.markdown("### LinkedIn Usage Breakdown")
        st.plotly_chart(
            px.pie(df_filtered, names="li_label",
                   color="li_label",
                   color_discrete_map={"Non-User":"red", "User":"green"}),
            use_container_width=True
        )

        st.markdown("### Age Distribution")
        st.plotly_chart(
            px.histogram(df_filtered, x="age", nbins=20,
                         color_discrete_sequence=["cyan"]),
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
        px.line(
            x=ages,
            y=lr.predict_proba(sweep)[:,1],
            labels={"x":"Age","y":"Predicted Probability"},
            title="Predicted Probability Across Age"
        ),
        use_container_width=True
    )



# ============================================================
# TAB 3 — MARKETING INSIGHTS (Version B)
# ============================================================
with tab_marketing:

    st.markdown("<h2 class='big-title'>Marketing Audience Insights</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### Executive Summary

    The LinkedIn prediction model highlights which demographic groups are
    most likely to use the platform. These insights help refine audience
    targeting, improve message testing, and support more effective media
    allocation. The findings emphasize strong adoption among higher-educated,
    higher-income users and early-to-mid career professionals, while also
    identifying segments where tailored messaging could expand engagement.
    """)

    df_seg = df.copy()

    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------
    st.markdown("## 1. Education Level")

    edu_rates = df_seg.groupby("education")["sm_li"].mean()
    edu_x = [education_labels[int(i)] for i in edu_rates.index]

    st.plotly_chart(
        px.bar(
            x=edu_x,
            y=edu_rates.values,
            labels={"x":"Education Level","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Education Level"
        ),
        use_container_width=True
    )

    st.markdown("""
    Users with college degrees and above show the strongest adoption.
    Messaging should emphasize professional credibility, advancement, and expertise.
    """)

    # ------------------------------------------------------
    # INCOME
    # ------------------------------------------------------
    st.markdown("## 2. Income Level")

    inc_rates = df_seg.groupby("income")["sm_li"].mean()
    inc_x = [income_labels[int(i)] for i in inc_rates.index]

    st.plotly_chart(
        px.bar(
            x=inc_x,
            y=inc_rates.values,
            labels={"x":"Income Bracket","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Income"
        ),
        use_container_width=True
    )

    st.markdown("""
    Adoption increases substantially with higher income.
    Messaging for these users should focus on leadership, influence, and high-level opportunities.
    """)

    # ------------------------------------------------------
    # AGE
    # ------------------------------------------------------
    st.markdown("## 3. Age Demographics")

    df_seg["age_group"] = pd.cut(
        df_seg["age"],
        bins=[18,25,35,45,55,65,120],
        labels=["18–25","26–35","36–45","46–55","56–65","65+"]
    )

    age_rates = df_seg.groupby("age_group")["sm_li"].mean()

    st.plotly_chart(
        px.line(
            x=age_rates.index.astype(str),
            y=age_rates.values,
            markers=True,
            labels={"x":"Age Group","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Age Group"
        ),
        use_container_width=True
    )

    st.markdown("""
    Usage peaks between ages 26–45. Messaging can highlight leadership development,
    career progression, and industry visibility.
    """)

    # ------------------------------------------------------
    # GENDER
    # ------------------------------------------------------
    st.markdown("## 4. Gender Differences")

    gender_rates = df_seg.groupby("female")["sm_li"].mean()

    st.plotly_chart(
        px.bar(
            x=["Male","Female"],
            y=[gender_rates.get(0), gender_rates.get(1)],
            labels={"x":"Gender","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Gender"
        ),
        use_container_width=True
    )

    st.markdown("""
    Women show slightly higher adoption. Messaging may emphasize community, mentorship,
    and equitable career growth.
    """)

    # ------------------------------------------------------
    # MARITAL STATUS
    # ------------------------------------------------------
    st.markdown("## 5. Marital Status")

    mar_rates = df_seg.groupby("married")["sm_li"].mean()

    st.plotly_chart(
        px.bar(
            x=["Not Married","Married"],
            y=[mar_rates.get(0), mar_rates.get(1)],
            labels={"x":"Marital Status","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Marital Status"
        ),
        use_container_width=True
    )

    st.markdown("""
    Non-married users adopt LinkedIn at higher rates, responding strongly to messages
    about mobility, opportunity, and accelerated growth.
    """)

    # ------------------------------------------------------
    # PARENTHOOD
    # ------------------------------------------------------
    st.markdown("## 6. Parenthood Status")

    par_rates = df_seg.groupby("parent")["sm_li"].mean()

    st.plotly_chart(
        px.bar(
            x=["Not Parent","Parent"],
            y=[par_rates.get(0), par_rates.get(1)],
            labels={"x":"Parent Status","y":"LinkedIn Usage Rate"},
            title="LinkedIn Usage by Parenthood"
        ),
        use_container_width=True
    )

    st.markdown("""
    Parents tend to respond well to messaging centered on flexibility, remote work options,
    and long-term professional stability.
    """)

    # ------------------------------------------------------
    # STRATEGIC RECOMMENDATIONS
    # ------------------------------------------------------
    st.markdown("## Strategic Recommendations")

    st.markdown("""
    1. Focus investment on high-probability adopters: ages 26–45, college-educated,
       and higher-income professionals.
    2. Tailor creative by demographic segment to improve engagement and relevance.
    3. Use predicted probabilities to build and refine high-ROI audience groups.
    4. Develop awareness campaigns for lower-income and older segments.
    5. Integrate segmentation insights into creative development and bidding strategy.
    """)

    st.markdown("## Key Takeaways")

    st.markdown("""
    - Education and income are the strongest structural predictors.  
    - Ages 26–45 form the core adoption cohort.  
    - Women adopt LinkedIn slightly more often than men.  
    - Marital and parenthood status provide valuable segmentation dimensions.  
    """)



# ============================================================
# TAB 4 — SHAP EXPLANATION
# ============================================================
with tab_shap:

    st.markdown("<h2 class='big-title'>SHAP Model Explanation</h2>", unsafe_allow_html=True)

    explainer = shap.Explainer(lr, X)
    shap_vals = explainer(X)

    st.markdown("### SHAP Feature Importance")
    fig1 = plt.figure(figsize=(7,4))
    shap.summary_plot(shap_vals.values, X, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.markdown("### SHAP Summary Plot")
    fig2 = plt.figure(figsize=(7,4))
    shap.summary_plot(shap_vals.values, X, show=False)
    st.pyplot(fig2)

    st.markdown("### SHAP Waterfall Plot — Selected Profile")
    shap_person = explainer(person)

    fig3 = plt.figure(figsize=(7,4))
    shap.waterfall_plot(shap_person[0], show=False)
    st.pyplot(fig3)



# ============================================================
# TAB 5 — MODEL PERFORMANCE
# ============================================================
with tab_perf:

    st.markdown("<h2 class='big-title'>Model Diagnostics</h2>", unsafe_allow_html=True)

    prob = lr.predict_proba(X)[:,1]
    preds = lr.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    fig = plt.figure(figsize=(7,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["Actual 0","Actual 1"])
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, prob)
    auc_score = roc_auc_score(y, prob)

    fig = plt.figure(figsize=(7,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(fig)

    # Odds Ratios
    odds = np.exp(lr.coef_[0])

    fig = plt.figure(figsize=(7,4))
    sns.barplot(x=odds, y=X.columns, orient="h")
    plt.title("Odds Ratios")
    st.pyplot(fig)

    # Probability Distribution
    fig = plt.figure(figsize=(7,4))
    sns.histplot(prob, bins=20, kde=True)
    plt.title("Distribution of Predicted Probabilities")
    st.pyplot(fig)

    # Calibration Curve
    true_prob, pred_prob = calibration_curve(y, prob, n_bins=10)
    fig = plt.figure(figsize=(7,4))
    plt.plot(pred_prob, true_prob, marker="o")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    st.pyplot(fig)

    # Partial Dependence Plot: Age
    age_range = np.arange(18,98)
    X_temp = X.copy()
    pdp_vals = []

    for a in age_range:
        X_temp["age"] = a
        pdp_vals.append(lr.predict_proba(X_temp)[:,1].mean())

    fig = plt.figure(figsize=(7,4))
    plt.plot(age_range, pdp_vals)
    plt.xlabel("Age")
    plt.ylabel("Predicted Probability")
    plt.title("Partial Dependence Plot: Age")
    st.pyplot(fig)


# Footer
st.markdown("---")
st.caption("Developed by Conal Masters — Georgetown MSBA — Machine Learning Dashboard")




