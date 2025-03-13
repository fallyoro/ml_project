import dill
import os
from src import CustomException
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def save_object(obj, path):
    try:
        dir_path= os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        with open(path, 'wb') as file:
            dill.dump(obj, file)

       
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, X_test, y_train, y_test, models) -> dict:
    """
    Entraîne et évalue plusieurs modèles de régression.
    
    Paramètres :
    - X : Features (données d'entrée)
    - y : Target (valeurs à prédire)
    - models : Dictionnaire contenant les modèles à évaluer
    
    Retourne :
    - Un dictionnaire contenant les scores R² de chaque modèle
    """
    # Division des données en ensemble d'entraînement et de test
  
   
    results = {}

    for name, model in models.items():
     
        model.fit(X_train, y_train)

        y_train_pred= model.predict(X_train)

        y_pred_test = model.predict(X_test)

        
        test_score = r2_score(y_test, y_pred_test)
        train_score = r2_score(y_train,y_train_pred)
        
        # Stockage du résultat
        results[name] = test_score

        print(f"{name} - R² Score: {test_score:.4f}")

    return results