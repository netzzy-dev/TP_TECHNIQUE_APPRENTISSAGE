import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.title("Prédiction du prix automobile")

st.write(
    "Entrez les caractéristiques du véhicule pour obtenir une estimation de son prix."
)

with st.form("vehicle_form"):

    year = st.number_input(
        "Année",
        min_value=1900,
        max_value=2030,
        value=2022
    )

    km = st.number_input(
        "Kilométrage",
        min_value=0,
        value=55098
    )

    bodydoors = st.number_input(
        "Nombre de portes",
        min_value=2,
        max_value=6,
        value=5
    )

    passengers = st.number_input(
        "Nombre de passagers",
        min_value=1,
        max_value=10,
        value=7
    )

    make = st.text_input("Marque", value="Jeep")
    model = st.text_input("Modèle", value="Grand Cherokee L")
    bodydescf = st.text_input("Type de carrosserie", value="VUS")
    drivetrain = st.text_input("Transmission motrice", value="4 roues motrices")
    fuel = st.text_input("Carburant", value="Essence")
    transdescf = st.text_input("Transmission", value="Automatique")
    colorf = st.text_input("Couleur", value="Charbon")

    submitted = st.form_submit_button("Prédire le prix")

if submitted:

    vehicle_data = {
        "YEAR": year,
        "KM": km,
        "BODYDOORS": bodydoors,
        "PASSENGERS": passengers,
        "MAKE": make,
        "MODEL": model,
        "BODYDESCF": bodydescf,
        "DRIVETRAIN": drivetrain,
        "FUEL": fuel,
        "TRANSDESCF": transdescf,
        "COLORF": colorf
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=vehicle_data
    )

    result = response.json()

    st.success(
    f"Prix prédit : {result['prediction']:,.2f} $"
    )

    st.info(
    f"Modèle utilisé : {result['model_name']} — alpha = {result['alpha']}"
    )