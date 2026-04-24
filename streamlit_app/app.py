# streamlit_app/app.py
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Air Paradis - Analyse de sentiment")
st.write("Entrez un tweet pour predire le sentiment associe.")

tweet = st.text_area("Tweet", placeholder="Tapez votre tweet ici...")

if st.button("Predire"):
    if not tweet.strip():
        st.warning("Veuillez entrer un tweet.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": tweet},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            sentiment = data["sentiment"]
            confidence = data["confidence"]

            if sentiment == "positif":
                st.success(f"Sentiment : {sentiment} (confiance : {confidence:.2%})")
            else:
                st.error(f"Sentiment : {sentiment} (confiance : {confidence:.2%})")

            st.session_state["last_prediction"] = {
                "tweet": tweet,
                "sentiment": sentiment,
                "confidence": confidence,
            }

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion a l'API : {e}")

if "last_prediction" in st.session_state:
    st.write("---")
    st.write("Cette prediction est-elle correcte ?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Bonne prediction"):
            st.success("Merci pour votre retour !")
            del st.session_state["last_prediction"]

    with col2:
        if st.button("Mauvaise prediction"):
            pred = st.session_state["last_prediction"]
            try:
                requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "tweet": pred["tweet"],
                        "predicted_sentiment": pred["sentiment"],
                        "confidence": pred["confidence"],
                    },
                    timeout=5,
                )
            except requests.exceptions.RequestException:
                pass
            st.warning("Merci ! Votre retour a ete enregistre pour amelioration.")
            del st.session_state["last_prediction"]
