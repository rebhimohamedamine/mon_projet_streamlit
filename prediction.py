"""
Fichier pour charger le modèle et faire des prédictions
"""
import joblib
import numpy as np

def load_model():
    """
    Charge le modèle sauvegardé
    
    Returns:
        model: Modèle de classification chargé
        class_names: Noms des classes
    """
    try:
        model = joblib.load('iris_model.pkl')
        class_names = joblib.load('class_names.pkl')
        return model, class_names
    except FileNotFoundError:
        raise FileNotFoundError(
            "Le modèle n'a pas été trouvé. "
            "Veuillez d'abord exécuter 'python model.py' pour entraîner et sauvegarder le modèle."
        )

def predict(sepal_length, sepal_width, petal_length, petal_width):
    """
    Fait une prédiction basée sur les caractéristiques de la fleur
    
    Args:
        sepal_length (float): Longueur du sépale (cm)
        sepal_width (float): Largeur du sépale (cm)
        petal_length (float): Longueur du pétale (cm)
        petal_width (float): Largeur du pétale (cm)
    
    Returns:
        tuple: (classe prédite, probabilités pour chaque classe)
    """
    # Charger le modèle
    model, class_names = load_model()
    
    # Préparer les données d'entrée
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Faire la prédiction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Obtenir le nom de la classe prédite
    predicted_class = class_names[prediction]
    
    # Créer un dictionnaire avec les probabilités pour chaque classe
    prob_dict = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    
    return predicted_class, prob_dict