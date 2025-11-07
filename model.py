"""
Fichier pour entraîner et sauvegarder le modèle de classification Iris
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_save_model():
    """
    Charge le dataset Iris, entraîne un modèle RandomForest et le sauvegarde
    """
    # 1. Charger le dataset Iris
    # Si vous n'avez pas le fichier CSV, on peut le générer depuis sklearn
    from sklearn.datasets import load_iris
    
    # Créer le dossier data s'il n'existe pas
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Charger les données depuis sklearn
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    
    # Sauvegarder le CSV
    df.to_csv('data/iris.csv', index=False)
    
    print("Dataset Iris chargé avec succès!")
    print(f"Dimensions: {df.shape}")
    print(f"\nPremières lignes:\n{df.head()}")
    
    # Préparer les données
    X = df.drop('species', axis=1)
    y = df['species']
    
    # 2. Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTaille ensemble d'entraînement: {len(X_train)}")
    print(f"Taille ensemble de test: {len(X_test)}")
    
    # 3. Entraîner le modèle RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    
    print("\nEntraînement du modèle en cours...")
    model.fit(X_train, y_train)
    
    # 4. Évaluer la précision
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"RÉSULTATS DE L'ENTRAÎNEMENT")
    print(f"{'='*50}")
    print(f"Précision sur l'ensemble de test: {accuracy * 100:.2f}%")
    print(f"\nRapport de classification:")
    print(classification_report(
        y_test, y_pred,
        target_names=iris.target_names
    ))
    
    # 5. Sauvegarder le modèle
    joblib.dump(model, 'iris_model.pkl')
    print(f"\nModèle sauvegardé avec succès dans 'iris_model.pkl'")
    
    # Sauvegarder également les noms des classes
    joblib.dump(iris.target_names, 'class_names.pkl')
    
    return model, accuracy

if __name__ == "__main__":
    train_and_save_model()