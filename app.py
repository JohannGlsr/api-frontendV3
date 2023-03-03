import streamlit as st
import requests
import nltk
import string
from nltk.stem import WordNetLemmatizer

# Initialisation des stopwords /  words / lemmatization
stopwords = nltk.corpus.stopwords.words('english') # Mots à supprimer
words = set(nltk.corpus.words.words()) # Totalités des mots de la langue
lemmatizer = WordNetLemmatizer() # Pour préserver la racine du mots

def Preprocess_Sentence(Sentence):        
    # Enlever la ponctuation
    Sentence = "".join([i.lower() for i in Sentence if i not in string.punctuation])
    # Enlever les chiffres
    Sentence = ''.join(i for i in Sentence if not i.isdigit())
    # Tokenization : Transformer les phrases en liste de tokens (en liste de mots)
    Sentence = nltk.tokenize.word_tokenize(Sentence)
    # Enlever les stopwords
    Sentence = [i for i in Sentence if i not in stopwords]
    # Enlever les majuscules
    Sentence = ' '.join(w for w in Sentence if w.lower() in words or not w.isalpha())

    return Sentence 

# Définir le titre de la page
st.set_page_config(page_title="Analyseur de sentiments")

# Titre de l'application
st.title("Analyse de sentiment par IA")

# Ajouter une image de fond
st.image("fond.png")

# Afficher une présentation
st.write("Bienvenue dans notre analyseur de sentiments !\n"
         "Celui-ci fonctionne avec un modèle de machine learning et a été entraîné sur 1 600 000 tweets.")

# Ajouter un champ de saisie pour la phrase
phrase = st.text_input("Entrez une phrase :")

sequence = Preprocess_Sentence(phrase)

# Ajouter un bouton pour lancer l'analyse
if st.button("Analyser"):
    # Envoyer la phrase à l'API et récupérer la prédiction
    response = requests.post("https://api-backendv3.herokuapp.com/prediction", data={'phrase': phrase})
    
    # Traiter la réponse de l'API
    if response.status_code == 200:
        result = response.json()
        if float(result['prediction']) < 0.2:
            st.write("La phrase est positive")
        elif float(result['prediction']) < 0.4:
            st.write("La phrase semble être positive")
        elif float(result['prediction']) < 0.6:
            st.write("La phrase semble être neutre")
        elif float(result['prediction']) < 0.8:
            st.write("La phrase semble être négative")
        else:
            st.write("La phrase est négative")
    else:
        st.write("Une erreur s'est produite lors de la requête à l'API.")