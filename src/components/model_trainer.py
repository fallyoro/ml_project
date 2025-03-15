import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src import logging, CustomException, save_object
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    train_model_path:str = os.path.join('artifacts', 'model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info('Splitting train and test data')
            X_train,y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[: , -1],
                test_arr[:, :-1],
                test_arr[: , -1]
            )

            logging.info('Splitting train and test data completed')

            models = {
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "RandomForest": RandomForestRegressor(),
                "SGD": SGDRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)  # Désactiver l'affichage des logs de CatBoost
            }

            model_result:dict = evaluate_models(X_train, X_test, y_train, y_test, models)

            best_model_score = max(sorted(model_result.values()))
            best_model_name = next((key for key, value in model_result.items() if value == best_model_score ), None)
            bests_model= models[best_model_name]

            if best_model_score <=0.6:
                raise CustomException('The best model is BAD')
            
            logging.info('The best model is found')

            save_object(
                obj=bests_model,
                path= self.model_trainer_config.train_model_path
            )

        except Exception as e:
            raise CustomException(e,sys)

