"""
Fichier pour charger le modèle et faire des prédictions
"""
import joblib
import numpy as np
import sklearn
import warnings

warnings.filterwarnings('ignore')

def load_model():
    """
    Charge le modèle sauvegardé
    
    Returns:
        model: Modèle de classification chargé
        class_names: Noms des classes
    """
    try:
        # Essayer de charger le nouveau format (avec métadonnées)
        model_data = joblib.load('iris_model.pkl')
        
        if isinstance(model_data, dict):
            model = model_data['model']
            class_names = model_data['target_names']
            saved_version = model_data.get('sklearn_version', 'unknown')
            
            current_version = sklearn.__version__
            
            # Vérifier la compatibilité des versions
            if saved_version != current_version:
                print(f"⚠️  Attention: Le modèle a été entraîné avec scikit-learn {saved_version}, "
                      f"mais vous utilisez la version {current_version}")
        else:
            # Format ancien (juste le modèle)
            model = model_data
            class_names = joblib.load('class_names.pkl')
        
        return model, class_names
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Le modèle n'a pas été trouvé. "
            "Veuillez d'abord exécuter 'python model.py' pour entraîner et sauvegarder le modèle."
        )
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")

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
    try:
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
        
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction: {str(e)}")