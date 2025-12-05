import streamlit as st
import requests
import pandas as pd

# URL de l'API Flask
API_URL = "http://127.0.0.1:5000/predict"

st.title("Prédiction avec le modèle RandomForest")

st.header("Informations numériques")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Duration", min_value=1, max_value=500, value=100)
campaign = st.number_input("Campaign", min_value=1, max_value=50, value=1)
pdays = st.number_input("Pdays", min_value=-1, max_value=999, value=-1)
previous = st.number_input("Previous", min_value=0, max_value=50, value=0)
emp_var_rate = st.number_input("Emp.var.rate", min_value=-5.0, max_value=5.0, value=1.1)
cons_price_idx = st.number_input("Cons.price.idx", min_value=90.0, max_value=100.0, value=93.994)
cons_conf_idx = st.number_input("Cons.conf.idx", min_value=-50.0, max_value=0.0, value=-36.4)
euribor3m = st.number_input("Euribor3m", min_value=0.0, max_value=6.0, value=4.857)
nr_employed = st.number_input("Nr employed", min_value=4000, max_value=7000, value=5195)

st.header("Informations catégorielles")
job = st.selectbox("Job", ["blue-collar","entrepreneur","housemaid","management","retired","self-employed",
                           "services","student","technician","unemployed","unknown"])
marital = st.selectbox("Marital", ["married","single","unknown"])
education = st.selectbox("Education", ["basic.6y","basic.9y","high.school","illiterate","professional.course",
                                       "university.degree","unknown"])
default = st.radio("Has credit default?", ["unknown","yes"])
housing = st.radio("Has housing loan?", ["unknown","yes"])
loan = st.radio("Has personal loan?", ["unknown","yes"])
contact = st.radio("Contact method", ["telephone"])
month = st.selectbox("Month", ["aug","dec","jul","jun","mar","may","nov","oct","sep"])
poutcome = st.selectbox("Previous outcome", ["nonexistent","success"])
day_of_week = st.selectbox("Day of the week", ["mon","thu","tue","wed"])

# Préparer le dictionnaire avec toutes les colonnes que le modèle attend
def make_features():
    features = {
        "age": age,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    # Colonnes binaires du modèle
    categorical_cols = [
        "job_blue-collar","job_entrepreneur","job_housemaid","job_management","job_retired",
        "job_self-employed","job_services","job_student","job_technician","job_unemployed","job_unknown",
        "marital_married","marital_single","marital_unknown",
        "education_basic.6y","education_basic.9y","education_high.school","education_illiterate",
        "education_professional.course","education_university.degree","education_unknown",
        "default_unknown","default_yes","housing_unknown","housing_yes","loan_unknown","loan_yes",
        "contact_telephone","month_aug","month_dec","month_jul","month_jun","month_mar","month_may",
        "month_nov","month_oct","month_sep","poutcome_nonexistent","poutcome_success",
        "day_of_week_mon","day_of_week_thu","day_of_week_tue","day_of_week_wed"
    ]

    # Initialiser toutes les colonnes binaires à 0
    for col in categorical_cols:
        features[col] = 0

    # Activer celles correspondant aux choix de l'utilisateur
    features[f"job_{job}"] = 1
    features[f"marital_{marital}"] = 1
    features[f"education_{education}"] = 1
    features[f"default_{default}"] = 1
    features[f"housing_{housing}"] = 1
    features[f"loan_{loan}"] = 1
    features[f"contact_{contact}"] = 1
    features[f"month_{month}"] = 1
    features[f"poutcome_{poutcome}"] = 1
    features[f"day_of_week_{day_of_week}"] = 1

    return features

if st.button("Prédire"):
    features = make_features()
    try:
        response = requests.post(API_URL, json=features)
        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            probability = data["probability"]

            st.success(f"Prédiction : {prediction}")
            # Affichage de la probabilité en bleu
            st.markdown(f"<span style='color:blue'>Probabilité d'avoir 1 : {probability:.2f}</span>", unsafe_allow_html=True)
        else:
            st.error(f"Erreur : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")